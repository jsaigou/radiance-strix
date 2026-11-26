/* hostbench_test.cpp -- the core's host hot paths, held to budgets, and the exactness of the
 * optimisations that bought them.
 *
 * Everything the engine does on the CPU between kernel launches is host overhead a decode step
 * cannot overlap once the device catches up: the batch build (KV slot mapping), the sampler's
 * staging, the prefix hash, detokenisation's UTF-8 hygiene, the token sink, the host backend's
 * stream queues. The perf_ cases here measure those paths at the shapes production uses and
 * hold them to ceilings an order of magnitude above a healthy run (best of several, like
 * tokenizer_perf_test), so a regression to a per-element loop or a per-call allocation fails
 * loudly instead of arriving as lost tokens a second.
 *
 * The non-perf cases are the correctness half: every dispatched SIMD level (util/simd.h) against
 * a scalar reference written here, and the batch slot walk (KVGeom::slots) against the per-token
 * slot() it replaces, over shapes that include the truncating-division corners.
 */
#include "rad_test.h"

#include "rad_internal.h"
#include "device/device.h"
#include "mem/prefix.h"
#include "sample/gbnf.h"
#include "sample/vocab_view.h"
#include "sched/mem_iface.h"
#include "server/json_partial.h"
#include "server/sink.h"
#include "text/chat.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"
#include "text/tokenizer.h"
#include "util/json.h"
#include "mm/imaging.h"
#include "server/oai.h"
#include "tokenizer_fixture.h"
#include "util/simd.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <random>

using namespace rad;

namespace {

double best_s(int reps, const std::function<void()>& f) {
    double b = 1e30;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        b = std::min(b, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return b;
}

volatile int64_t g_sink = 0;

/* Run f with each SIMD level the CPU can execute forced in turn, restoring the detected level. */
void for_each_level(const std::function<void(int)>& f) {
    const int top = simd::detect_level();
    for (int lvl = 0; lvl <= top; ++lvl) {
        simd::force_level(lvl);
        f(lvl);
    }
    simd::force_level(top);
}

}  /* namespace */

/* ================================================================== SIMD exactness */

TEST(simd_levels_agree_with_scalar_reference) {
    std::mt19937_64 rng(20261105);
    for_each_level([&](int lvl) {
        fprintf(stderr, "    level %s\n", simd::level_name(lvl));
        for (size_t n : { (size_t)0, (size_t)1, (size_t)3, (size_t)7, (size_t)8, (size_t)15,
                          (size_t)16, (size_t)17, (size_t)31, (size_t)63, (size_t)64,
                          (size_t)65, (size_t)100, (size_t)256, (size_t)1000 }) {
            std::vector<int32_t> v(n);
            for (auto& x : v) x = (int32_t)(rng() % 200001) - 100000;
            if (n >= 2) { v[0] = INT32_MIN; v[n - 1] = INT32_MAX; }

            int32_t want_max = INT32_MIN;
            for (int32_t x : v) want_max = std::max(want_max, x);
            CHECK_EQ(simd::max_i32(v.data(), n), want_max);

            for (int32_t probe : { -100000, 0, 7, INT32_MAX, INT32_MIN }) {
                size_t want = 0;
                for (int32_t x : v) want += x == probe;
                CHECK_EQ(simd::count_eq_i32(v.data(), n, probe), want);
            }

            std::vector<int64_t> w(n);
            for (auto& x : w) x = (int64_t)(rng() % 1000);
            for (int64_t probe : { (int64_t)-1, (int64_t)0, (int64_t)500, (int64_t)999,
                                   (int64_t)1000000 }) {
                int64_t want = -1;
                for (size_t i = 0; i < n; ++i)
                    if (w[i] == probe) { want = (int64_t)i; break; }
                CHECK_EQ(simd::find_eq_i64(w.data(), n, probe), want);
            }

            std::string s(n, 'a');
            if (n) s[rng() % n] = '\x80';
            size_t want_na = n;
            for (size_t i = 0; i < n; ++i)
                if ((unsigned char)s[i] >= 0x80) { want_na = i; break; }
            CHECK_EQ(simd::first_non_ascii(s.data(), n), want_na);
            std::string plain(n, 'z');
            CHECK_EQ(simd::first_non_ascii(plain.data(), n), n);

            /* The JSON escaper's byte class: controls, quote, backslash, non-ASCII. */
            std::string js = plain;
            size_t want_js = n;
            if (n) {
                const size_t at = rng() % n;
                js[at] = "\"\\\x01\x1f\x80\xff"[rng() % 6];
                want_js = at;
            }
            CHECK_EQ(simd::first_json_special(js.data(), n), want_js);
            CHECK_EQ(simd::first_json_special(plain.data(), n), n);
        }

        /* level bitmaps: every length around a word, NaN, signed zeros, infinities, subnormals,
         * values at the level and at the band's edge */
        for (size_t n : { (size_t)0, (size_t)1, (size_t)5, (size_t)63, (size_t)64, (size_t)65,
                          (size_t)127, (size_t)129, (size_t)513 }) {
            const float specials[] = { std::nanf(""), 0.0f, -0.0f, INFINITY, -INFINITY, 1e-40f,
                                       -1e-40f, 0.25f, 0.25f + 0.95f, 0.25f - 0.95f };
            std::vector<float> v(n);
            for (auto& x : v) x = rng() % 3 == 0 ? specials[rng() % 10] : (float)((int)(rng() % 4001) - 2000) / 1000.0f;
            const size_t words = (n + 63) / 64;
            std::vector<uint64_t> g(words + 1, ~0ull), l(words + 1, ~0ull), q(words + 1, ~0ull), r(words + 1, ~0ull);
            simd::level_bits_f32(v.data(), n, 0.25f, 0.95f, g.data(), l.data(), q.data(), r.data());
            for (size_t w = 0; w < words; ++w) {
                uint64_t wg = 0, wl = 0, wq = 0, wr = 0;
                for (size_t b = 0; b < 64 && w * 64 + b < n; ++b) {
                    const float x = v[w * 64 + b];
                    wg |= (uint64_t)(x > 0.25f) << b;
                    wl |= (uint64_t)(x < 0.25f) << b;
                    wq |= (uint64_t)std::isnan(x) << b;
                    wr |= (uint64_t)(std::fabs(x - 0.25f) < 0.95f) << b;
                }
                CHECK_EQ(g[w], wg);
                CHECK_EQ(l[w], wl);
                CHECK_EQ(q[w], wq);
                CHECK_EQ(r[w], wr);
            }
            CHECK_EQ(g[words], ~0ull);   /* nothing written past the last word */
        }

        /* bitstream shift, every shift */
        for (unsigned sh = 1; sh < 32; ++sh) {
            for (size_t words : { (size_t)1, (size_t)2, (size_t)5, (size_t)16, (size_t)17,
                                  (size_t)40, (size_t)257 }) {
                std::vector<uint32_t> src(words), got(words, 0xdeadbeefu), want(words);
                for (auto& x : src) x = (uint32_t)rng();
                for (size_t i = 0; i < words; ++i) {
                    const uint32_t hi = i + 1 < words ? src[i + 1] : 0u;
                    want[i] = (src[i] >> sh) | (hi << (32 - sh));
                }
                simd::bitstream_shr(got.data(), src.data(), words, sh);
                CHECK(got == want);
            }
        }

        /* bitmap marks, including out-of-range ids */
        {
            std::vector<uint32_t> bits(40);
            for (auto& x : bits) x = (uint32_t)rng();
            std::vector<int32_t> ids = { -5, -1, 0, 1, 31, 32, 33, 100, 1279, 1280, 5000,
                                         INT32_MAX, 64, 65 };
            for (int i = 0; i < 200; ++i) ids.push_back((int32_t)(rng() % 1400) - 50);
            std::vector<uint8_t> got(ids.size(), 0xee), want(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) {
                const int32_t id = ids[i];
                want[i] = id >= 0 && (size_t)(id >> 5) < bits.size()
                              ? (uint8_t)((bits[(size_t)(id >> 5)] >> (id & 31)) & 1u) : 0;
            }
            simd::bitmap_marks(got.data(), ids.data(), ids.size(), bits.data(), bits.size());
            CHECK(got == want);
            std::vector<uint8_t> z(ids.size(), 0xee);
            simd::bitmap_marks(z.data(), ids.data(), ids.size(), bits.data(), 0);
            for (uint8_t b : z) CHECK_EQ(b, (uint8_t)0);
        }

        /* byte ORs: three rows, a missing row, in place; and a line with its neighbours */
        for (size_t n : { (size_t)0, (size_t)1, (size_t)2, (size_t)3, (size_t)15, (size_t)16,
                          (size_t)17, (size_t)31, (size_t)32, (size_t)33, (size_t)63, (size_t)64,
                          (size_t)65, (size_t)66, (size_t)129, (size_t)257, (size_t)1000 }) {
            std::vector<uint8_t> a(n), b(n), c(n), got(n, 0xee), want(n);
            for (size_t i = 0; i < n; ++i) {
                a[i] = (uint8_t)(rng() % 7 == 0 ? rng() : 0);
                b[i] = (uint8_t)rng();
                c[i] = (uint8_t)(1u << (rng() % 8));
            }
            for (size_t i = 0; i < n; ++i) want[i] = (uint8_t)(a[i] | b[i] | c[i]);
            simd::or3_u8(got.data(), a.data(), b.data(), c.data(), n);
            CHECK(got == want);
            for (size_t i = 0; i < n; ++i) want[i] = (uint8_t)(a[i] | c[i]);
            simd::or3_u8(got.data(), a.data(), nullptr, c.data(), n);
            CHECK(got == want);
            std::vector<uint8_t> in = a;
            for (size_t i = 0; i < n; ++i) want[i] = (uint8_t)(a[i] | b[i]);
            simd::or3_u8(in.data(), in.data(), b.data(), nullptr, n);
            CHECK(in == want);
            for (size_t i = 0; i < n; ++i)
                want[i] = (uint8_t)((i > 0 ? a[i - 1] : 0) | a[i] | (i + 1 < n ? a[i + 1] : 0));
            std::fill(got.begin(), got.end(), (uint8_t)0xee);
            simd::or_adjacent_u8(got.data(), a.data(), n);
            CHECK(got == want);
        }

        /* the filter's vertical taps: int32 rows times a coefficient, summed in int64 */
        for (size_t n : { (size_t)0, (size_t)1, (size_t)3, (size_t)4, (size_t)5, (size_t)7,
                          (size_t)8, (size_t)9, (size_t)17, (size_t)1536, (size_t)1539 }) {
            std::vector<int32_t> h(n);
            for (auto& x : h) x = (int32_t)(rng() % 2000001) - 1000000;
            if (n >= 2) { h[0] = INT32_MIN; h[n - 1] = INT32_MAX; }
            for (int32_t c : { 0, 1, -1, 2048, -300, INT32_MAX, INT32_MIN }) {
                std::vector<int64_t> acc(n), want(n);
                for (size_t i = 0; i < n; ++i) acc[i] = want[i] = (int64_t)(rng() % 1000) - 500;
                for (size_t i = 0; i < n; ++i) want[i] += (int64_t)h[i] * c;
                simd::madd_i32_i64(acc.data(), h.data(), c, n);
                CHECK(acc == want);
            }
        }

        /* base64 runs: whatever a level consumes must be the reference's encoding of exactly
         * those bytes, and its decoding must stop before the first non-digit */
        {
            static const char kAlpha[] =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            auto value = [](unsigned char c) -> int {
                if (c >= 'A' && c <= 'Z') return c - 'A';
                if (c >= 'a' && c <= 'z') return c - 'a' + 26;
                if (c >= '0' && c <= '9') return c - '0' + 52;
                return c == '+' ? 62 : c == '/' ? 63 : -1;
            };
            for (size_t n : { (size_t)0, (size_t)11, (size_t)12, (size_t)16, (size_t)27,
                              (size_t)28, (size_t)51, (size_t)52, (size_t)53, (size_t)100,
                              (size_t)3000 }) {
                std::vector<uint8_t> raw(n);
                for (auto& x : raw) x = (uint8_t)rng();
                std::string enc(n / 3 * 4 + 4, '?');
                const size_t used = simd::b64_encode_run(raw.data(), n, enc.data());
                CHECK(used % 3 == 0);
                CHECK(used <= n);
                for (size_t g = 0; g < used / 3; ++g) {
                    const uint32_t v = (uint32_t)raw[3 * g] << 16 | (uint32_t)raw[3 * g + 1] << 8 |
                                       raw[3 * g + 2];
                    for (int k = 0; k < 4; ++k)
                        CHECK_EQ(enc[4 * g + (size_t)k], kAlpha[(v >> (18 - 6 * k)) & 63]);
                }
                /* the full reference encoding, then one byte made each of a few non-digits */
                std::string full;
                for (size_t g = 0; g + 3 <= n; g += 3) {
                    const uint32_t v = (uint32_t)raw[g] << 16 | (uint32_t)raw[g + 1] << 8 | raw[g + 2];
                    for (int k = 0; k < 4; ++k) full += kAlpha[(v >> (18 - 6 * k)) & 63];
                }
                for (int trial = 0; trial < 6; ++trial) {
                    std::string t = full;
                    if (trial > 0 && !t.empty())
                        t[rng() % t.size()] = "=\n \x80\xff-"[trial - 1];
                    std::vector<uint8_t> out(t.size() / 4 * 3 + 64, 0xee);
                    const size_t took = simd::b64_decode_run((const uint8_t*)t.data(), t.size(), out.data());
                    CHECK(took % 4 == 0);
                    CHECK(took <= t.size());
                    for (size_t i = 0; i < took; ++i) CHECK(value((unsigned char)t[i]) >= 0);
                    for (size_t g = 0; g < took / 4; ++g) {
                        uint32_t v = 0;
                        for (int k = 0; k < 4; ++k) v = v << 6 | (uint32_t)value((unsigned char)t[4 * g + (size_t)k]);
                        CHECK_EQ(out[3 * g], (uint8_t)(v >> 16));
                        CHECK_EQ(out[3 * g + 1], (uint8_t)(v >> 8));
                        CHECK_EQ(out[3 * g + 2], (uint8_t)v);
                    }
                }
            }
        }

        /* the striped content hash: every level against a scalar restatement, and two stripes
         * swapped within a block must not accumulate alike */
        {
            auto key = [](int i) {
                uint64_t z = 0x5241444943414e45ull + 0x9e3779b97f4a7c15ull * (uint64_t)(i + 1);
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
                return z ^ (z >> 31);
            };
            auto ref = [&](uint64_t* acc, const uint8_t* p, size_t stripes) {
                for (size_t s2 = 0; s2 < stripes; ++s2) {
                    const size_t r = s2 % 16;
                    for (int j = 0; j < 8; ++j) {
                        uint64_t d;
                        std::memcpy(&d, p + 64 * s2 + 8 * (size_t)j, 8);
                        const uint64_t k = d ^ key((int)r + j);
                        acc[j] += (k & 0xffffffffull) * (k >> 32);
                        acc[j ^ 1] += d;
                    }
                    if (r == 15)
                        for (int j = 0; j < 8; ++j)
                            acc[j] = (acc[j] ^ (acc[j] >> 47) ^ key(16 + j)) * 0x9E3779B1ull;
                }
            };
            for (size_t stripes : { (size_t)0, (size_t)1, (size_t)15, (size_t)16, (size_t)17,
                                    (size_t)33, (size_t)200 }) {
                std::vector<uint8_t> buf(64 * stripes + 7);
                for (auto& x : buf) x = (uint8_t)rng();
                uint64_t got[8], want[8];
                for (int j = 0; j < 8; ++j) got[j] = want[j] = rng();
                simd::stripe_acc_u64x8(got, buf.data() + 7 * (stripes & 1), stripes);
                ref(want, buf.data() + 7 * (stripes & 1), stripes);
                for (int j = 0; j < 8; ++j) CHECK_EQ(got[j], want[j]);
                if (stripes >= 16) {
                    std::vector<uint8_t> sw = buf;
                    std::swap_ranges(sw.begin() + 64 * 2, sw.begin() + 64 * 3, sw.begin() + 64 * 5);
                    uint64_t x[8] = {}, y[8] = {};
                    simd::stripe_acc_u64x8(x, buf.data(), stripes);
                    simd::stripe_acc_u64x8(y, sw.data(), stripes);
                    bool differ = false;
                    for (int j = 0; j < 8; ++j) differ |= x[j] != y[j];
                    CHECK(differ);
                }
            }
        }
    });
}

/* ================================================================== the slot walk */

TEST(batch_slot_walk_matches_per_token_slot) {
    std::mt19937_64 rng(977);
    auto geom = [](int64_t bs) {
        KVGeom kg;
        KVGeom::G g;
        g.block_size = bs;
        g.paged = true;
        kg.g.push_back(g);
        return kg;
    };
    for (int64_t bs : { (int64_t)1, (int64_t)2, (int64_t)3, (int64_t)16, (int64_t)48,
                        (int64_t)128, (int64_t)256 }) {
        KVGeom kg = geom(bs);
        std::vector<int32_t> bt(300);
        for (auto& x : bt) x = (int32_t)(rng() % 100000);
        for (int64_t first : { (int64_t)0, (int64_t)1, (int64_t)7, bs, 2 * bs + 1, (int64_t)1000,
                               (int64_t)-3, -bs }) {
            for (int64_t pos0 : { (int64_t)-2 * bs - 1, (int64_t)-1, (int64_t)0, (int64_t)1,
                                  first - 1, first, first + 1, (int64_t)999, (int64_t)4096 }) {
                const int64_t n = 3 * bs + 5;
                std::vector<int32_t> got((size_t)n, -777), want((size_t)n);
                kg.slots(0, bt.data(), bt.size(), first, pos0, n, got.data());
                for (int64_t j = 0; j < n; ++j)
                    want[(size_t)j] = kg.slot(0, bt, first, pos0 + j);
                if (got != want) {
                    fprintf(stderr, "    bs=%lld first=%lld pos0=%lld\n", (long long)bs,
                            (long long)first, (long long)pos0);
                    CHECK(got == want);
                }
                /* The ring mapping, armed and not. */
                for (int64_t ring_blocks : { (int64_t)1, (int64_t)2, (int64_t)5 }) {
                    SeqWindow w;
                    w.set = true;
                    w.ring_pos = 4 * bs;
                    w.ring_blocks = ring_blocks;
                    std::vector<int32_t> g2((size_t)n, -777), w2((size_t)n);
                    kg.slots_ring(0, bt.data(), bt.size(), first, pos0, n, w, g2.data());
                    for (int64_t j = 0; j < n; ++j)
                        w2[(size_t)j] = kg.slot_ring(0, bt, first, pos0 + j, w);
                    if (g2 != w2) {
                        fprintf(stderr, "    ring bs=%lld first=%lld pos0=%lld blocks=%lld\n",
                                (long long)bs, (long long)first, (long long)pos0,
                                (long long)ring_blocks);
                        CHECK(g2 == w2);
                    }
                }
            }
        }
    }
}

/* ================================================================== perf budgets */

TEST(perf_slot_fill_costs_a_fraction_of_a_division_per_token) {
    KVGeom kg;
    KVGeom::G g;
    g.block_size = 16;
    g.paged = true;
    kg.g.push_back(g);
    std::vector<int32_t> bt(16384), out(8192);
    std::mt19937_64 rng(5);
    for (auto& x : bt) x = (int32_t)(rng() % 100000);
    const double t = best_s(10, [&] {
        kg.slots(0, bt.data(), bt.size(), 0, 100000, 8192, out.data());
    });
    g_sink += out[3];
    const double ns = t / 8192 * 1e9;
    fprintf(stderr, "    batch slot fill, 8192-token chunk: %.2f ns/token (two 64-bit divisions "
                    "a token measured ~3.3)\n", ns);
    CHECK(ns < 1.5);
}

TEST(perf_simd_primitives_at_production_shapes) {
    std::mt19937_64 rng(11);
    /* The seqused span maximum (256 rows) and the group maximum. */
    std::vector<int32_t> used(256);
    for (auto& x : used) x = (int32_t)(rng() % 130000);
    const double t_max = best_s(200, [&] { g_sink += simd::max_i32(used.data(), used.size()); });
    fprintf(stderr, "    max_i32 x256: %.0f ns (%s)\n", t_max * 1e9,
            simd::level_name(simd::level()));
    CHECK(t_max * 1e9 < 400.0);

    /* A 151936-token vocabulary mask window, unaligned shard (bitstream_shr). */
    std::vector<uint32_t> full(4748), dst(4748);
    for (auto& x : full) x = (uint32_t)rng();
    const double t_shr = best_s(100, [&] {
        simd::bitstream_shr(dst.data(), full.data(), full.size(), 17);
    });
    g_sink += dst[9];
    fprintf(stderr, "    bitstream_shr x4748 words: %.0f ns (per-bit loop measured ~435000)\n",
            t_shr * 1e9);
    CHECK(t_shr * 1e9 < 20000.0);

    /* DRY breaker marks over an 8k history window. */
    std::vector<int32_t> hist(8192);
    for (auto& x : hist) x = (int32_t)(rng() % 150000);
    std::vector<uint32_t> bits(4748);
    for (auto& x : bits) x = (uint32_t)rng();
    std::vector<uint8_t> marks(8192);
    const double t_mk = best_s(50, [&] {
        simd::bitmap_marks(marks.data(), hist.data(), hist.size(), bits.data(), bits.size());
    });
    g_sink += marks[5];
    fprintf(stderr, "    bitmap_marks x8192: %.0f ns\n", t_mk * 1e9);
    CHECK(t_mk * 1e9 < 30000.0);
}

TEST(perf_prefix_chain_hash_throughput) {
    std::mt19937_64 rng(3);
    std::vector<int32_t> toks(256);
    for (auto& x : toks) x = (int32_t)(rng() % 150000);
    BlockHash h{};
    const double t = best_s(10, [&] {
        BlockHash p = h;
        for (int b = 0; b < 512; ++b)
            p = chain_block_hash(p, toks.data(), 256, nullptr, 0, (int64_t)b * 256);
        h = p;
    });
    g_sink += (int64_t)h.lo;
    const double ns = t / (512 * 256) * 1e9;
    fprintf(stderr, "    chain_block_hash: %.2f ns/token (%.2f ms per 128k prompt)\n", ns,
            ns * 131072 / 1e6);
    CHECK(ns < 12.0);
}

TEST(perf_utf8_sanitize_ascii_throughput) {
    std::string big(64 * 1024, 'x');
    const double t = best_s(20, [&] {
        auto s = utf8_sanitize(big);
        g_sink += (int64_t)s.size();
    });
    const double mbps = (double)big.size() / t / 1e6;
    fprintf(stderr, "    utf8_sanitize ASCII: %.0f MB/s\n", mbps);
    CHECK(mbps > 1500.0);
}

TEST(perf_sink_push_cost_per_token) {
    server::Sink sink(8192);
    const int rounds = 20000;
    const double t = best_s(5, [&] {
        server::Sink s2(8192);
        /* One clock reading for all of them, as the scheduler pushes a step's tokens. */
        const int64_t now = server::sink_now_ns();
        for (int i = 0; i < rounds; ++i) {
            /* Keep the ring near-empty, as a live consumer leaves it: push one, read one. */
            if (!s2.push(i, nullptr, now)) break;
            server::SinkToken out[1];
            (void)s2.read(out, 1);
            g_sink += out[0].token;
        }
    });
    const double ns = t / rounds * 1e9;
    fprintf(stderr, "    sink push+read: %.0f ns/token\n", ns);
    CHECK(ns < 1000.0);
    (void)sink;
}

TEST(perf_host_stream_small_task_throughput) {
    if (!device_is_host()) {
        radtest::skip("host stream queues exist only on the host backend");
        return;
    }
    RadStream st = nullptr;
    REQUIRE(rad_stream_create(&st, 0) >= 0);
    void* dst = rad_dev_alloc(4096, RAD_MEM_HOST_PINNED);
    void* src = rad_dev_alloc(4096, RAD_MEM_HOST_PINNED);
    REQUIRE(dst && src);
    std::memset(src, 7, 4096);
    const int tasks = 20000;
    const double t = best_s(3, [&] {
        for (int i = 0; i < tasks; ++i) rad_memcpy_async(dst, src, 64, st);
        rad_stream_sync(st);
    });
    const double ns = t / tasks * 1e9;
    fprintf(stderr, "    host stream 64B memcpy+sync amortised: %.0f ns/task\n", ns);
    CHECK(ns < 3000.0);
    CHECK_EQ(((unsigned char*)dst)[63], (unsigned char)7);
    rad_dev_free(dst, RAD_MEM_HOST_PINNED);
    rad_dev_free(src, RAD_MEM_HOST_PINNED);
    rad_stream_destroy(st);
}

/* ================================================================== tokenizer stages
 *
 * The encoder as a whole is budgeted in tokenizer_perf_test; what a regression hunt needs is
 * where inside it the time goes. Same trained byte-level vocabulary and document, the step chain
 * varied: full (NFC + split + bytelevel), without the normaliser, and with neither (the BPE word
 * path over whole segments -- different ids, timed only to price the steps it skips). The
 * segment cache is off for the stage prices and on for the last row, which is the serving shape
 * for a re-sent conversation. */

namespace {

std::string bench_document(uint32_t seed, size_t bytes) {
    static const char* words[] = {
        "the", "cache", "is", "sharded", "by", "key", "and", "each", "shard", "has", "its", "own",
        "lock", "so", "that", "readers", "never", "wait", "on", "writers", "of", "another", "we",
        "measured", "it", "at", "requests", "per", "second", "with", "p99", "latency", "under",
        "load", "tokenizer", "encode", "returns", "ids", "for", "text", "value", "struct", "func",
        "return", "if", "else", "nil", "err", "string", "int", "len", "make", "map", "append",
        "更新", "缓存", "naïve", "café", "über", "Straße", "ünïcode", "don't", "it's", "we'll",
    };
    static const char* punct[] = { ".", ",", ";", ":", "(", ")", "{", "}", "[", "]", " =",
                                   " :=", " ==", " !=", " //", " ->", "\"", "'", "?", "!" };
    std::mt19937 rng(seed);
    std::string s;
    while (s.size() < bytes) {
        const int n = 3 + (int)(rng() % 14);
        for (int k = 0; k < n; ++k) {
            if (k) s += ' ';
            s += words[rng() % (sizeof(words) / sizeof(words[0]))];
            if (rng() % 5 == 0) s += punct[rng() % (sizeof(punct) / sizeof(punct[0]))];
            if (rng() % 17 == 0) s += std::to_string(rng() % 100000);
        }
        s += rng() % 3 ? ".\n" : (rng() % 2 ? "\n\t" : "\n\n    ");
    }
    return s;
}

std::shared_ptr<const Vocab> bench_vocab(std::vector<VocabStep> steps) {
    using namespace tokfix;
    VocabBuild vb = train_bytelevel(bench_document(1, 300000), k_qwen35, 1500);
    vb.steps = std::move(steps);
    add_specials(vb, { { "<|im_start|>", RAD_TT_CONTROL }, { "<|im_end|>", RAD_TT_CONTROL } });
    return load(std::move(vb));
}

}  /* namespace */

TEST(perf_tokenizer_stage_breakdown) {
    using namespace tokfix;
    const std::string doc = bench_document(7, 1 << 20);
    struct Variant {
        const char*           name;
        std::vector<VocabStep> steps;
        bool                  cache;
    };
    const Variant variants[] = {
        { "full (NFC+split+bytelevel)",
          { simple_step(RAD_NORM_NFC), split_step(k_qwen35), bytelevel_step() }, false },
        { "no normaliser", { split_step(k_qwen35), bytelevel_step() }, false },
        { "bytelevel only", { bytelevel_step() }, false },
        { "full, segment cache on",
          { simple_step(RAD_NORM_NFC), split_step(k_qwen35), bytelevel_step() }, true },
    };
    double full_mbps = 0;
    for (const Variant& var : variants) {
        auto v = bench_vocab(var.steps);
        REQUIRE(v != nullptr);
        Tokenizer t;
        REQUIRE(t.init(v) >= 0);
        t.set_segment_cache(var.cache);
        std::vector<int32_t> ids;
        if (var.cache) t.encode(doc, ids, false, true);   /* warm the cache: the re-sent shape */
        const double sec = best_s(3, [&] {
            ids.clear();
            t.encode(doc, ids, false, true);
        });
        g_sink += (int64_t)ids.size();
        const double mbps = (double)doc.size() / sec / 1e6;
        fprintf(stderr, "    encode %-28s %7.1f MB/s (%zu ids)\n", var.name, mbps, ids.size());
        if (!var.cache && full_mbps == 0) full_mbps = mbps;
    }
    /* tokenizer_perf_test holds the strict floor; this only keeps the breakdown honest. */
    CHECK(full_mbps > 10.0);
}

TEST(perf_reply_stream_feed_per_piece) {
    /* The streaming reply parser runs on the decode thread's output, one token piece at a
     * time. Feed a long reply (reasoning, prose, one tool call) in 4-byte pieces and price it
     * per piece; the whole-reply parse is the reference shape. */
    ChatFormat f;
    std::string refused, why;
    const std::string tmpl =
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message.role + '\\n' }}"
        "{% if message.content %}{{ message.content }}{% endif %}"
        "{% if message.tool_calls %}{% for call in message.tool_calls %}"
        "{{ '<tool_call>\\n<function=' + call.function.name + '>\\n' }}"
        "{% for k, v in call.function.arguments.items() %}"
        "{{ '<parameter=' + k + '>\\n' + v | string + '\\n</parameter>\\n' }}"
        "{% endfor %}{{ '</function>\\n</tool_call>' }}{% endfor %}{% endif %}"
        "{{ '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n' }}{% endif %}";
    REQUIRE(chat_format_from_template(tmpl, "", "<|im_end|>", &f, &refused, &why) >= 0);
    std::shared_ptr<const ReplyFormat> rf;
    REQUIRE(ReplyFormat::compile(f, &rf, &why) >= 0);
    using ojson = Json;
    ojson tools = ojson::array({ ojson{{"type", "function"},
        {"function", ojson{{"name", "edit_file"}, {"description", "edit"},
            {"parameters", ojson{{"type", "object"},
                {"properties", ojson{{"path", ojson{{"type", "string"}}},
                                      {"old_string", ojson{{"type", "string"}}}}},
                {"required", ojson::array({"path"})}}}}}} });
    ReplyOptions ro;
    ro.generation_prompt = "<|im_start|>assistant\n<think>\n";
    ro.tools = &tools;
    std::shared_ptr<const ReplyParser> rp;
    REQUIRE(ReplyParser::create(rf, ro, &rp, &why) >= 0);

    std::string reply = "Let me check the call sites first.\n</think>\n";
    while (reply.size() < 100000) reply += "The scheduler admits one more chunk and the sampler reads it back. ";
    reply += "<tool_call>\n<function=edit_file>\n<parameter=path>\ncore/x.cpp\n</parameter>\n"
             "<parameter=old_string>\n" + std::string(4000, 'x') + "\n</parameter>\n</function>\n</tool_call>Done.";

    {
        const auto m = reply_parse(*rp, reply);
        REQUIRE_EQ(m.calls.size(), 1u);
        CHECK_EQ(m.calls[0].name, std::string("edit_file"));
        CHECK(m.content.ends_with("Done."));
    }
    const size_t pieces = (reply.size() + 3) / 4;
    const double sec = best_s(5, [&] {
        auto stream = rp->open();
        std::vector<ReplyEvent> ev;
        for (size_t i = 0; i < reply.size(); i += 4) {
            stream->feed(std::string_view(reply).substr(i, 4), &ev);
            for (const auto& e : ev) g_sink += (int64_t)e.text.size();
            ev.clear();
        }
        stream->finish(&ev);
        g_sink += (int64_t)stream->message().content.size();
    });
    const double ns = sec / (double)pieces * 1e9;
    fprintf(stderr, "    reply stream feed: %.0f ns/piece over %zu pieces\n", ns, pieces);
    CHECK(ns < 500.0);
}

TEST(perf_sse_stream_push_per_piece) {
    /* The per-token SSE framing for each API shape, fed the same 4-byte pieces. The OpenAI chunk
     * is hand-written; Anthropic's and Responses' delta events were DOM dumps at ~600-800
     * ns/piece before they were written the same way. */
    using namespace server;
    OaiRequest r;
    r.id = "msg_bench";
    r.model = "bench-model";
    r.created = 1700000000;
    std::string text;
    while (text.size() < 100000) text += "The scheduler admits one more chunk and reads it back. ";
    const size_t pieces = (text.size() + 3) / 4;
    auto bench = [&](const char* name, auto make) {
        const double sec = best_s(5, [&] {
            auto s = make();
            for (size_t i = 0; i < text.size(); i += 4) {
                auto out = s->push(std::string_view(text).substr(i, 4), false);
                g_sink += (int64_t)out.size();
            }
            auto fin = s->final_chunk(Finish::Stop, TimingsOut(), UsageOut());
            g_sink += (int64_t)fin.size();
        });
        const double ns = sec / (double)pieces * 1e9;
        fprintf(stderr, "    sse %-22s %.0f ns/piece over %zu pieces\n", name, ns, pieces);
        CHECK(ns < 500.0);
    };
    bench("openai chat", [&] { return std::make_unique<ChatDeltaStream>(r, 0, nullptr); });
    bench("anthropic messages", [&] { return std::make_unique<AnthropicStream>(r, 0, nullptr); });
    bench("responses", [&] { return std::make_unique<ResponsesStream>(r, 0, nullptr); });
}

TEST(perf_detokenizer_push_throughput) {
    using namespace tokfix;
    auto v = bench_vocab({ simple_step(RAD_NORM_NFC), split_step(k_qwen35), bytelevel_step() });
    REQUIRE(v != nullptr);
    Tokenizer t;
    REQUIRE(t.init(v) >= 0);
    const std::string doc = bench_document(9, 1 << 18);
    std::vector<int32_t> ids;
    REQUIRE(t.encode(doc, ids, false, true) >= 0);
    REQUIRE(!ids.empty());
    const double sec = best_s(3, [&] {
        Detokenizer d(v);
        size_t got = 0;
        for (int32_t id : ids) got += d.push(id).size();
        got += d.flush().size();
        g_sink += (int64_t)got;
    });
    const double ns = sec / (double)ids.size() * 1e9;
    fprintf(stderr, "    detokenizer push: %.0f ns/token over %zu tokens\n", ns, ids.size());
    CHECK(ns < 2000.0);
}

/* ================================================================== grammar
 *
 * A JSON grammar over a realistic-width synthetic vocabulary: what admission pays to compile it
 * (once per distinct grammar text; the program cache is bypassed here by measuring the first
 * compile) and what a driven mask costs per token on the scheduler thread. The strict mask
 * budget lives in gbnf_test; this keeps the compile side and the drive shape under watch. */

TEST(perf_grammar_compile_and_mask) {
    SimpleVocab v;
    for (int c = 1; c < 256; ++c) v.add(std::string(1, (char)c));
    v.add("<|im_end|>", true);
    std::mt19937_64 rng(4242);
    static const char* frags[] = { "th", "he", "in", "er", "an", "ou", "re", "ed", "es", "st",
                                   "the", "ing", "and", "ion", "ent", " \"", "\":", "{\"" , "},",
                                   "true", "false", "null", "name", "value", "type", "items" };
    std::vector<std::string> pieces;
    for (const char* f : frags) pieces.emplace_back(f);
    while (v.n_tokens() < 48000) {
        std::string p;
        const int k = 1 + (int)(rng() % 3);
        for (int i = 0; i < k; ++i) p += frags[rng() % (sizeof(frags) / sizeof(frags[0]))];
        v.add(p);
    }

    static const char* kJson =
        "root ::= object\n"
        "object ::= \"{\" ws members? ws \"}\"\n"
        "members ::= member (\",\" ws member)*\n"
        "member ::= string ws \":\" ws value\n"
        "value ::= object | array | string | number | \"true\" | \"false\" | \"null\"\n"
        "array ::= \"[\" ws elements? ws \"]\"\n"
        "elements ::= value (\",\" ws value)*\n"
        "string ::= \"\\\"\" chars \"\\\"\"\n"
        "chars ::= char*\n"
        "char ::= [^\"\\\\\\x00-\\x1F] | \"\\\\\" esc\n"
        "esc ::= [\"\\\\/bfnrt] | \"u\" [0-9a-fA-F]{4}\n"
        "number ::= \"-\"? int frac? exp?\n"
        "int ::= \"0\" | [1-9] [0-9]*\n"
        "frac ::= \".\" [0-9]+\n"
        "exp ::= [eE] [+-]? [0-9]+\n"
        "ws ::= [ \\t\\n\\r]*\n";

    const auto t0 = std::chrono::steady_clock::now();
    GbnfVocabInfo vi;
    CHECK_OK(gbnf_prepare_vocab(&v, &vi));
    const double index_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const auto t1 = std::chrono::steady_clock::now();
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    REQUIRE(GbnfGrammar::create(kJson, "root", &v, false, {}, {}, &g, &err) >= 0);
    const double compile_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();

    const int64_t n_words = (v.n_tokens() + 31) / 32;
    std::vector<uint32_t> mask((size_t)n_words);
    const std::string text =
        "{\"name\":\"radiance\",\"items\":[1,2.5,true,null,{\"x\":\"y\\nz\"}],\"n\":-3e2}";
    std::vector<int32_t> toks;
    REQUIRE(v.tokenize(text, &toks) >= 0);
    REQUIRE(!toks.empty());
    const auto t2 = std::chrono::steady_clock::now();
    int driven = 0;
    for (int rep = 0; rep < 20; ++rep) {
        std::unique_ptr<GbnfGrammar> d;
        REQUIRE(GbnfGrammar::create(kJson, "root", &v, false, {}, {}, &d, &err) >= 0);
        for (int32_t tok : toks) {
            REQUIRE(d->fill_mask(mask.data(), n_words) >= 0);
            g_sink += mask[0];
            if (d->accept_token(tok) < 0) break;
            ++driven;
        }
    }
    const double drive_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
    fprintf(stderr,
            "    grammar over %d tokens: vocab index %.0f ms (reported %.0f), compile %.0f ms, "
            "mask+accept %.3f ms/token over %d driven\n",
            (int)v.n_tokens(), index_ms, vi.build_ms, compile_ms,
            driven ? drive_ms / driven : 0.0, driven);
    CHECK(compile_ms < 5000.0);
    CHECK(driven > 0);
    CHECK(drive_ms / std::max(driven, 1) < 1.0);
}

/* Complex and deep grammars: a 300-rule delegation chain, an 800-way literal enum with shared
 * prefixes, 150-deep nesting driven token by token, and a recursive-descent expression grammar.
 * Real tool grammars are wide rather than deep, but a schema can be both, and compile cost that
 * is superlinear in rules (or mask cost that grows with live stack depth) only shows here. */
TEST(perf_deep_and_complex_grammars) {
    SimpleVocab v;
    for (int c = 1; c < 256; ++c) v.add(std::string(1, (char)c));
    v.add("<|im_end|>", true);
    std::mt19937_64 rng(4242);
    static const char* frags[] = { "th", "he", "in", "er", "an", "ou", "re", "ed", "es", "st",
                                   "the", "ing", "and", "ion", "ent", " \"", "\":", "{\"", "},",
                                   "true", "false", "null", "name", "value", "type", "items" };
    while (v.n_tokens() < 48000) {
        std::string p;
        const int k = 1 + (int)(rng() % 3);
        for (int i = 0; i < k; ++i) p += frags[rng() % (sizeof(frags) / sizeof(frags[0]))];
        v.add(p);
    }
    GbnfVocabInfo vi;
    CHECK_OK(gbnf_prepare_vocab(&v, &vi));
    const int64_t n_words = (v.n_tokens() + 31) / 32;
    std::vector<uint32_t> mask((size_t)n_words);

    auto compile = [&](const std::string& text, const char* name,
                       std::unique_ptr<GbnfGrammar>* out) {
        const auto t0 = std::chrono::steady_clock::now();
        std::string err;
        const int rc = GbnfGrammar::create(text.c_str(), "root", &v, false, {}, {}, out, &err);
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "    grammar %-22s compile %8.1f ms\n", name, ms);
        REQUIRE(rc >= 0);
        CHECK(ms < 10000.0);
    };
    auto drive = [&](const std::string& gtxt, std::unique_ptr<GbnfGrammar> first,
                     const std::string& text, const char* name) {
        std::vector<int32_t> toks;
        REQUIRE(v.tokenize(text, &toks) >= 0);
        REQUIRE(!toks.empty());
        /* Repeat short drives from a fresh grammar on the same program: the first drive pays for
         * discovering mask states, and an average over six tokens would call that the per-token
         * cost of every later turn. At least 2,000 driven tokens makes the figure production's. */
        const int reps = std::max(1, (int)(2000 / toks.size()));
        double fill_ms = 0, accept_ms = 0;
        int driven = 0;
        for (int rep = 0; rep < reps; ++rep) {
            std::unique_ptr<GbnfGrammar> g = std::move(first);
            if (!g) compile(gtxt, "(repeat)", &g);
            for (int32_t tok : toks) {
                auto t0 = std::chrono::steady_clock::now();
                REQUIRE(g->fill_mask(mask.data(), n_words) >= 0);
                const auto t1 = std::chrono::steady_clock::now();
                fill_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                g_sink += mask[0];
                if (g->accept_token(tok) < 0) break;
                accept_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t1).count();
                ++driven;
            }
        }
        const double ms = fill_ms + accept_ms;
        fprintf(stderr, "    drive   %-22s %d tokens, %.3f ms/token (fill %.3f, accept %.3f)\n",
                name, driven, driven ? ms / driven : 0.0,
                driven ? fill_ms / driven : 0.0, driven ? accept_ms / driven : 0.0);
        CHECK(driven == reps * (int)toks.size());
        CHECK(ms / std::max(driven, 1) < 5.0);
    };

    {   /* A 300-rule delegation chain. */
        std::string gtxt = "root ::= r1\n";
        for (int i = 1; i < 300; ++i)
            gtxt += "r" + std::to_string(i) + " ::= r" + std::to_string(i + 1) + "\n";
        gtxt += "r300 ::= \"z\"\n";
        std::unique_ptr<GbnfGrammar> g; compile(gtxt, "300-rule chain", &g);
        drive(gtxt, std::move(g), "z", "300-rule chain");
    }
    {   /* An 800-way enum of literals sharing long prefixes. */
        std::string gtxt = "root ::= ";
        for (int i = 0; i < 800; ++i) {
            if (i) gtxt += " | ";
            char buf[16];
            std::snprintf(buf, sizeof buf, "\"tok%03d\"", i);
            gtxt += buf;
        }
        gtxt += "\n";
        std::unique_ptr<GbnfGrammar> g; compile(gtxt, "800-way enum", &g);
        drive(gtxt, std::move(g), "tok517", "800-way enum");
    }
    {   /* 150-deep nesting, driven at depth. */
        static const char* kJson =
            "root ::= value\n"
            "value ::= object | array | string | number | \"true\" | \"false\" | \"null\"\n"
            "object ::= \"{\" ws members? ws \"}\"\n"
            "members ::= member (\",\" ws member)*\n"
            "member ::= string ws \":\" ws value\n"
            "array ::= \"[\" ws elements? ws \"]\"\n"
            "elements ::= value (\",\" ws value)*\n"
            "string ::= \"\\\"\" chars \"\\\"\"\n"
            "chars ::= char*\n"
            "char ::= [^\"\\\\\\x00-\\x1F] | \"\\\\\" esc\n"
            "esc ::= [\"\\\\/bfnrt] | \"u\" [0-9a-fA-F]{4}\n"
            "number ::= \"-\"? int frac? exp?\n"
            "int ::= \"0\" | [1-9] [0-9]*\n"
            "frac ::= \".\" [0-9]+\n"
            "exp ::= [eE] [+-]? [0-9]+\n"
            "ws ::= [ \\t\\n\\r]*\n";
        std::unique_ptr<GbnfGrammar> g; compile(kJson, "json (deep drive)", &g);
        drive(kJson, std::move(g), std::string(150, '[') + "1" + std::string(150, ']'),
              "150-deep array");
    }
    {   /* Recursive descent with repetition at every level. */
        static const char* kExpr =
            "root ::= expr\n"
            "expr ::= term ((\"+\" | \"-\") term)*\n"
            "term ::= factor ((\"*\" | \"/\") factor)*\n"
            "factor ::= number | ident | \"(\" expr \")\"\n"
            "number ::= [0-9]+ (\".\" [0-9]+)?\n"
            "ident ::= [a-z] [a-z0-9]*\n";
        std::unique_ptr<GbnfGrammar> g; compile(kExpr, "expression", &g);
        std::string text;
        for (int i = 0; i < 40; ++i) text += "(alpha+12*beta3-(";
        text += "7";
        for (int i = 0; i < 40; ++i) text += ")/2)";
        drive(kExpr, std::move(g), text, "40-deep expression");
    }
}

/* ================================================================== request path
 *
 * The per-token JSON escape on the streaming path and the per-request chat render: an agent
 * re-sends its whole conversation every turn, so the render is priced at conversation scale. */

TEST(perf_request_body_parse) {
    /* The whole conversation arrives as one body every turn; parse_json_bounded turns it into
     * the DOM the request parser walks. Reading it with the previous DOM library measured ~340 MB/s;
     * the yyjson read plus the converting walk is priced here so a reader swap or a fallback
     * that starts swallowing the fast path shows up as lost milliseconds per turn. */
    using ojson = Json;
    ojson body;
    body["model"] = "bench";
    ojson msgs = ojson::array();
    std::string chunk;
    while (chunk.size() < 5000)
        chunk += "fn main() { let window = compute(17, 0.5); println!(\"{:?}\", window); }\n";
    for (int i = 0; i < 360; ++i) {
        ojson m;
        m["role"] = (i % 3 == 0) ? "user" : (i % 3 == 1) ? "assistant" : "tool";
        m["content"] = chunk + std::to_string(i);
        msgs.push_back(std::move(m));
    }
    body["messages"] = std::move(msgs);
    const std::string text = body.dump();
    const double sec = best_s(10, [&] {
        bool over = false, deep = false;
        auto j = server::parse_json_bounded(text, 131072, &over, &deep);
        REQUIRE(!over && !deep && !j.is_discarded());
        g_sink += (int64_t)j["messages"].size();
    });
    const double mbps = (double)text.size() / sec / 1e6;
    fprintf(stderr, "    request body parse, %.2f MiB: %.2f ms (%.0f MB/s)\n",
            text.size() / 1048576.0, sec * 1000.0, mbps);
    CHECK(mbps > 1000.0);
}

TEST(perf_flat_object_parse) {
    /* The other body shape: one huge flat object -- a full logit_bias at admission, a
     * tokenizer.json vocab at startup. Member insertion by linear find made this quadratic
     * (a 262k-member vocab parsed in 27 s); the floor below is far under a healthy read and
     * far above that regime. */
    std::string text = "{\"logit_bias\":{";
    for (int i = 0; i < 65536; ++i) {
        if (i) text += ',';
        text += '"' + std::to_string(i) + "\":0.5";
    }
    text += "}}";
    const double sec = best_s(3, [&] {
        bool over = false, deep = false;
        auto j = server::parse_json_bounded(text, 131072, &over, &deep);
        REQUIRE(!over && !deep && !j.is_discarded());
        g_sink += (int64_t)j["logit_bias"].size();
    });
    const double mbps = (double)text.size() / sec / 1e6;
    fprintf(stderr, "    flat object parse, %.2f MiB / 65536 members: %.2f ms (%.0f MB/s)\n",
            text.size() / 1048576.0, sec * 1000.0, mbps);
    CHECK(mbps > 25.0);
}

TEST(perf_json_quote_throughput) {
    const std::string doc = bench_document(11, 1 << 20);
    std::string out;
    const double sec = best_s(10, [&] {
        out.clear();
        /* Chunk by chunk, as deltas arrive: many small strings, not one 1 MiB quote. */
        for (size_t i = 0; i < doc.size(); i += 7)
            if (!server::json_quote_append(out, std::string_view(doc).substr(i, 7)))
                out += "\"?\"";
        g_sink += (int64_t)out.size();
    });
    const double mbps = (double)doc.size() / sec / 1e6;
    fprintf(stderr, "    json_quote_append, 7-byte chunks: %.0f MB/s\n", mbps);
    CHECK(mbps > 100.0);
}

TEST(perf_embeddings_body_serialization) {
    /* A batch answer is hundreds of thousands of floats; building a DOM node per float and
     * dumping it afterwards is pure response-path latency. The writer in embeddings_body is
     * checked here for float-exact round-trip and priced per float. */
    server::OaiRequest r;
    r.model = "emb-bench";
    server::UsageOut u;
    u.prompt = 4096;
    {
        const std::vector<std::vector<float>> edge = {
            { 0.0f, -0.0f, 1e-30f, 3.0e38f, std::numeric_limits<float>::quiet_NaN(),
              std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity() } };
        const auto o = Json::parse(server::embeddings_body(r, edge, u));
        const auto& e = o["data"][0]["embedding"];
        REQUIRE_EQ(e.size(), edge[0].size());
        for (size_t j = 0; j < 4; ++j) CHECK_EQ(e[j].get<float>(), edge[0][j]);
        CHECK(e[4].is_null());
        CHECK(e[5].is_null());
        CHECK(e[6].is_null());
    }
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::vector<float>> vecs(64, std::vector<float>(2048));
    for (auto& v : vecs) for (float& x : v) x = dist(rng);
    const std::string body = server::embeddings_body(r, vecs, u);
    const auto o = Json::parse(body);
    for (size_t i = 0; i < vecs.size(); i += 13)
        for (size_t j = 0; j < vecs[i].size(); j += 257)
            CHECK_EQ(o["data"][i]["embedding"][j].get<float>(), vecs[i][j]);
    const double sec = best_s(5, [&] {
        g_sink += (int64_t)server::embeddings_body(r, vecs, u).size();
    });
    const double ns = sec * 1e9 / (double)(vecs.size() * vecs[0].size());
    fprintf(stderr, "    embeddings_body, 64x2048 floats: %.0f ns/float (%.2f MiB out)\n", ns,
            body.size() / 1048576.0);
    CHECK(ns < 120.0);
}

TEST(perf_chat_template_apply_at_conversation_scale) {
    /* A ChatML-shaped template with the loops real ones have: system, per-message dispatch,
     * tool calls with argument dumps. Inline so the bench needs no fixture file. */
    static const char* kTmpl =
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message.role + '\\n' }}"
        "{% if message.content %}{{ message.content }}{% endif %}"
        "{% if message.tool_calls %}{% for call in message.tool_calls %}"
        "{{ '<tool_call>\\n<function=' + call.function.name + '>\\n' }}"
        "{% for k, v in call.function.arguments.items() %}"
        "{{ '<parameter=' + k + '>\\n' + v | string + '\\n</parameter>\\n' }}"
        "{% endfor %}{{ '</function>\\n</tool_call>' }}{% endfor %}{% endif %}"
        "{{ '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
    ChatTemplate ct;
    REQUIRE(ct.load(kTmpl, "", "<|im_end|>") >= 0);
    using ojson = Json;
    ojson messages = ojson::array();
    messages.push_back(ojson{ { "role", "system" },
                               { "content", "You are a careful coding assistant." } });
    const std::string para = bench_document(3, 4000);
    for (int i = 0; i < 120; ++i) {
        messages.push_back(ojson{ { "role", "user" }, { "content", para } });
        ojson call{ { "id", "call_1" }, { "type", "function" },
                    { "function", ojson{ { "name", "edit_file" },
                                         { "arguments", ojson{ { "path", "core/x.cpp" },
                                                                { "old_string", para },
                                                                { "new_string", para } } } } } };
        messages.push_back(ojson{ { "role", "assistant" }, { "content", "" },
                                  { "tool_calls", ojson::array({ call }) } });
        messages.push_back(ojson{ { "role", "tool" }, { "tool_call_id", "call_1" },
                                  { "content", para } });
    }
    /* No tools array: this prices the render, not grammar generation (perf_grammar_*). */
    const ojson tools = ojson::array();
    ChatPrompt cp;
    REQUIRE(ct.apply(messages, tools, ChatOptions(), &cp) >= 0);
    REQUIRE(cp.prompt.size() > 100000);
    const double sec = best_s(3, [&] {
        cp = ChatPrompt();
        if (ct.apply(messages, tools, ChatOptions(), &cp) >= 0)
            g_sink += (int64_t)cp.prompt.size();
    });
    const double in_mbps = (double)messages.dump().size() / sec / 1e6;
    fprintf(stderr, "    chat apply, %zu messages -> %zu prompt bytes: %.1f ms (%.0f MB/s in)\n",
            messages.size(), cp.prompt.size(), sec * 1e3, in_mbps);
    CHECK(sec * 1e3 < 1000.0);
}

TEST(perf_media_ingress_base64_and_resize) {
    /* A picture in a chat request is host work before prefill: megabytes of base64 to
     * decode, then a PIL-exact Lanczos resize to the tower's size. The goldens in mm_test
     * pin the resize's bytes; this prices both at a phone-photo scale so a per-byte or
     * single-threaded regression shows up as admission latency. */
    std::mt19937_64 rng(7);
    std::vector<uint8_t> raw(4 << 20);
    for (auto& b : raw) b = (uint8_t)rng();
    std::string enc;
    const double t_enc = best_s(3, [&] { enc = server::base64_encode(raw.data(), raw.size()); });
    std::vector<uint8_t> dec;
    const double t_dec = best_s(3, [&] {
        REQUIRE(server::base64_decode(enc, &dec));
    });
    REQUIRE(dec == raw);
    fprintf(stderr, "    base64 encode %.0f MiB/s, decode %.0f MiB/s (4 MiB payload)\n",
            4.0 / t_enc, 4.0 / t_dec);
    CHECK(4.0 / t_enc > 200.0);
    CHECK(4.0 / t_dec > 400.0);

    mm::FrameRGBA src;
    src.w = 2048;
    src.h = 1536;
    src.rgba.resize((size_t)src.w * src.h * 4);
    for (auto& b : src.rgba) b = (uint8_t)rng();
    mm::FrameRGBA out;
    const double t_rsz = best_s(3, [&] { mm::resize_lanczos_rgba(src, 512, 384, &out); });
    REQUIRE(out.w == 512 && out.h == 384 && out.rgba.size() == (size_t)512 * 384 * 4);
    g_sink += out.rgba[(size_t)100 * 512 * 4 + 40];
    fprintf(stderr, "    lanczos 2048x1536 -> 512x384: %.1f ms\n", t_rsz * 1e3);
    /* One thread measured ~40 ms at this shape; where cores exist the row split must stay
     * well under it. A one- or two-core box gets the scalar budget instead. */
    CHECK(t_rsz * 1e3 < (std::thread::hardware_concurrency() >= 4 ? 25.0 : 200.0));
}

RAD_TEST_MAIN()
