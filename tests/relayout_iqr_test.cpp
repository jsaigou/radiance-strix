/* relayout_iqr_test.cpp -- the IQR weights' stored form (libr4d/r4d_iqr.h) from the relayout hook,
 * against a reference written from the header's description, on every form the kernels are built
 * for; and back through the inverse.
 *
 * The relayout reads each lane's codes of a row as one field and writes a segment as a stream of
 * dwords; the reference here places every code alone, at the bit r4d_iqr.h names for it, so the
 * two share no code. The inverse (r4d_rad_unrelayout_iqr) is a third walk of the same order. A
 * code the form cannot read must refuse the weight.
 *
 * libr4d's layouts are host C++, compiled into this test (tests/CMakeLists.txt).
 */
#include "rad_test.h"
#include "rad_plugin.h"
#include "rad_encoding.h"
#include "r4d_iqr.h"
#include "r4d_hsimd.h"
#include "r4d_fp8_frag.h"


#include <cstring>
#include <vector>

extern "C" int r4d_rad_layout_iqr(const RadParam*, int, int, const RadEncoding*, const int*,
                                  const RadTensor*, int, RadLayout*);
extern "C" int r4d_rad_relayout_iqr(const RadParam*, int, int, const RadEncoding*, const int*,
                                    const RadTensor*, int, void*, int64_t);
extern "C" int r4d_rad_unrelayout_iqr(const RadParam*, int, int, const RadEncoding*, const int*,
                                      const void*, int64_t, const RadTensor*, int);

namespace {

using namespace r4d_iqr;

RadParam pint(const char* k, long long v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
RadParam pstr(const char* k, const char* v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}

uint64_t g_z = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {
    g_z += 0x9E3779B97F4A7C15ull;
    uint64_t x = g_z;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

const uint32_t kU[] = { 0, RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6, RAD_U7,
                        RAD_U8, RAD_U9, RAD_U10, RAD_U11, RAD_U12 };

/* One weight's planes, as the loader hands them over. */
struct W {
    RadEncoding enc{};
    std::vector<std::vector<uint8_t>> data;
    std::vector<int64_t> rows, cols;
    int find(const char* r) const { return rad_enc_find(&enc, r); }
    RadTensor t(int i) {
        RadTensor x;
        std::memset(&x, 0, sizeof x);
        x.data = data[(size_t)i].data();
        x.dtype = enc.plane[i].dtype;
        x.rank = 2;
        x.shape[0] = rows[(size_t)i];
        x.shape[1] = cols[(size_t)i];
        rad_tensor_pack(&x);
        return x;
    }
    int64_t row_bytes(int i) const { return rad_enc_row_bytes(enc.plane[i].dtype, cols[(size_t)i]); }
    void size(int64_t n, int64_t k) {
        data.assign((size_t)enc.n_planes, {});
        rows.assign((size_t)enc.n_planes, 0);
        cols.assign((size_t)enc.n_planes, 0);
        for (int i = 0; i < enc.n_planes; ++i) {
            rad_enc_plane_dims(&enc.plane[i], n, k, &rows[(size_t)i], &cols[(size_t)i]);
            data[(size_t)i].assign((size_t)(rows[(size_t)i] * row_bytes(i)), 0);
        }
    }
};

/* kind as r4d_iqr.h's Kind; V the grid's vector; L the trellis' state bits. */
struct Case { uint32_t kind; int B, V, L; bool rot; };

/* The planes of a weight of form `c`, codes at random and the auxiliary plane valid. */
W make(const Case& c, int64_t N, int64_t K) {
    W w;
    const bool tr = is_trellis(c.kind);
    if (tr) {
        rad_enc_clear(&w.enc);
        rad_enc_copy_str(w.enc.scheme, "trellis");
        rad_enc_add_plane(&w.enc, "codes", kU[c.B], 1, 1);
        rad_enc_add_plane(&w.enc, "scale", RAD_F8E4M3, 1, 128);
        rad_enc_add_plane(&w.enc, "scale.1", RAD_F32, 0, 0);
        const int t = rad_enc_add_table(&w.enc, "trellis", RAD_I8, (int64_t)1 << c.L, 1);
        w.enc.plane[t].block[0] = 256;
        w.enc.plane[t].block[1] = c.L == 12 ? 0 : 32;
        if (c.kind == kTrellisMad) rad_enc_add_table(&w.enc, "trellis.mad", RAD_U32, 2, 1);
    } else {
        w.enc = rad_enc_affine(kU[c.B], RAD_BF16, 1, 128);
        if (c.kind == kGrid) {
            w.enc.plane[0].block[1] = c.V;
            rad_enc_add_table(&w.enc, "grid", RAD_I8, (int64_t)1 << c.B, c.V);
        } else if (c.kind == kTable) {
            rad_enc_add_table(&w.enc, "table", RAD_I8, 1, (int64_t)1 << c.B);
        } else if (c.kind == kZero) {
            rad_enc_add_plane(&w.enc, "zero", RAD_U6, 1, 128);
        }
    }
    if (c.rot) rad_enc_copy_str(w.enc.transform, "fwht128");
    w.size(N, K);
    const int ic = w.find("codes");
    for (auto& b : w.data[(size_t)ic]) b = (uint8_t)rnd();
    const int ia = tr ? w.find("trellis") : c.kind == kGrid ? w.find("grid")
                 : c.kind == kTable ? w.find("table") : c.kind == kZero ? w.find("zero") : -1;
    if (c.kind == kZero) {
        /* six-bit zeros, a row's unused high bits clear as the inverse leaves them */
        for (int64_t r = 0; r < N; ++r)
            for (int64_t kb = 0; kb < K / 128; ++kb)
                rad_store_bits(w.data[(size_t)ia].data() + r * w.row_bytes(ia), 6, kb,
                               (uint32_t)(rnd() & 63u));
    } else if (c.kind == kTrellisMad) {
        const uint32_t A = (uint32_t)(rnd() & 0xFFFFFFu) | 1u, Bc = (uint32_t)rnd();
        uint32_t* ab = (uint32_t*)w.data[(size_t)w.find("trellis.mad")].data();
        ab[0] = A;
        ab[1] = Bc;
        for (uint32_t st = 0; st < (1u << c.L); ++st) {
            const uint32_t h = ((st & 0xFFFFFFu) * A + Bc) & 0x3F3F3F3Fu;
            w.data[(size_t)ia][st] = (uint8_t)(int8_t)((int)((h & 0xFFu) + ((h >> 8) & 0xFFu) +
                                                             ((h >> 16) & 0xFFu) + (h >> 24)) - 126);
        }
    } else if (ia >= 0) {
        for (auto& b : w.data[(size_t)ia]) b = (uint8_t)(int8_t)((int)(rnd() % 255) - 127);
    }
    return w;
}

const char* aux_role(const Case& c) {
    return is_trellis(c.kind) ? "trellis" : c.kind == kGrid ? "grid" : c.kind == kTable ? "table"
         : c.kind == kZero ? "zero" : nullptr;
}

/* The selected planes: codes, the auxiliary plane, A and B for a computed trellis. */
struct Sel {
    std::vector<int> sel;
    std::vector<RadTensor> t;
};
Sel select(W& w, const Case& c) {
    Sel s;
    std::vector<const char*> roles = { "codes" };
    if (const char* a = aux_role(c)) roles.push_back(a);
    if (c.kind == kTrellisMad) roles.push_back("trellis.mad");
    for (const char* r : roles) {
        s.sel.push_back(w.find(r));
        s.t.push_back(w.t(s.sel.back()));
    }
    return s;
}

/* r4d_iqr.h's stored weight, every code placed alone. */
std::vector<uint8_t> reference(W& w, const Case& c, int64_t N, int64_t K, int64_t bytes) {
    std::vector<uint8_t> d((size_t)bytes, 0);
    Form f{};
    f.kind = c.kind;
    f.bits = (uint32_t)c.B;
    f.lv = c.kind == kGrid ? (c.V == 4 ? 2u : 1u) : 0u;
    const int ic = w.find("codes");
    const int64_t nkb = K / 128, rbc = w.row_bytes(ic);
    const int64_t ub = unit_bytes(f) * 8, lb = lane_bits(f);
    const int per = c.kind == kGrid ? 4 / c.V : 4;
    for (int64_t rt = 0; rt < N / 16; ++rt)
        for (int64_t kb = 0; kb < nkb; ++kb)
            for (int lane = 0; lane < 32; ++lane)
                for (int r = 0; r < 16; ++r)
                    for (int e = 0; e < per; ++e) {
                        const int64_t col = c.kind == kGrid ? (kb * 128 + 4 * lane) / c.V + e
                                                            : kb * 128 + 4 * lane + e;
                        const uint32_t code = rad_load_bits(w.data[(size_t)ic].data() +
                                                            (rt * 16 + r) * rbc, c.B, col);
                        const int64_t idx = is_trellis(c.kind) ? 16 * e + 15 - r : per * r + e;
                        const int64_t bit = (rt * nkb + kb) * ub + lane * lb + idx * c.B;
                        for (int b = 0; b < c.B; ++b)
                            if (code >> b & 1u) d[(size_t)((bit + b) >> 3)] |= (uint8_t)(1u << ((bit + b) & 7));
                    }
    uint8_t* aux = d.data() + aux_off(f, N, K);
    const int ia = w.find(aux_role(c) ? aux_role(c) : "codes");
    if (c.kind == kZero) {
        for (int64_t r = 0; r < N; ++r)
            for (int64_t kb = 0; kb < nkb; ++kb)
                aux[tile_major(r, kb, nkb)] =
                    (uint8_t)rad_load_bits(w.data[(size_t)ia].data() + r * w.row_bytes(ia), 6, kb);
    } else if (aux_role(c)) {
        const std::vector<uint8_t>& t = w.data[(size_t)ia];
        /* A computed trellis keeps A and B alone; every other form its table. */
        if (c.kind == kTrellisMad) std::memcpy(aux, w.data[(size_t)w.find("trellis.mad")].data(), 8);
        else std::memcpy(aux, t.data(), t.size());
    }
    return d;
}

std::vector<RadParam> params(int64_t N, int64_t K, bool rot) {
    return { pint("N", N), pint("K", K), pint("group", 128), pstr("dtype", rot ? "i8a8h" : "i8a8") };
}

/* Relayout, reference and inverse of one form at one size. */
void check_form(const Case& c, int64_t N, int64_t K) {
    W w = make(c, N, K);
    Sel s = select(w, c);
    const std::vector<RadParam> lp = params(N, K, c.rot);
    RadLayout L;
    std::memset(&L, 0, sizeof L);
    REQUIRE_EQ(r4d_rad_layout_iqr(lp.data(), (int)lp.size(), 2, &w.enc, s.sel.data(), s.t.data(),
                                  (int)s.t.size(), &L), RAD_OK);
    std::vector<uint8_t> got((size_t)L.bytes, 0xA5);
    REQUIRE_EQ(r4d_rad_relayout_iqr(lp.data(), (int)lp.size(), 2, &w.enc, s.sel.data(), s.t.data(),
                                    (int)s.t.size(), got.data(), L.bytes), RAD_OK);
    const std::vector<uint8_t> want = reference(w, c, N, K, L.bytes);
    size_t first = got.size();
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i]) { first = i; break; }
    if (first != got.size())
        std::fprintf(stderr, "kind %u B=%d V=%d L=%d N=%lld K=%lld: first differing byte %zu of %zu\n",
                     c.kind, c.B, c.V, c.L, (long long)N, (long long)K, first, got.size());
    CHECK_EQ(first, got.size());

    /* back: fresh planes of the same shape, filled by the inverse */
    W back = w;
    for (auto& p : back.data) std::fill(p.begin(), p.end(), (uint8_t)0x5A);
    Sel sb = select(back, c);
    REQUIRE_EQ(r4d_rad_unrelayout_iqr(lp.data(), (int)lp.size(), 2, &back.enc, sb.sel.data(),
                                      got.data(), L.bytes, sb.t.data(), (int)sb.t.size()), RAD_OK);
    for (size_t i = 0; i < s.sel.size(); ++i) {
        const bool same = back.data[(size_t)s.sel[i]] == w.data[(size_t)s.sel[i]];
        if (!same)
            std::fprintf(stderr, "kind %u B=%d: plane %s does not come back\n", c.kind, c.B,
                         w.enc.plane[s.sel[i]].role);
        CHECK(same);
    }
}

const Case kForms[] = {
    { kTable, 2, 1, 0, false }, { kTable, 3, 1, 0, false }, { kTable, 4, 1, 0, true },
    { kTable, 5, 1, 0, false },
    { kGrid, 8, 4, 0, false }, { kGrid, 9, 4, 0, false }, { kGrid, 10, 4, 0, true },
    { kGrid, 12, 4, 0, false },
    { kZero, 6, 1, 0, false },
    { kTrellis, 2, 1, 12, true }, { kTrellis, 3, 1, 12, true }, { kTrellis, 5, 1, 12, false },
    { kTrellis, 7, 1, 12, true }, { kTrellis, 4, 1, 14, true }, { kTrellis, 6, 1, 14, false },
    { kTrellisMad, 2, 1, 16, true }, { kTrellisMad, 3, 1, 16, true }, { kTrellisMad, 4, 1, 16, true },
    { kTrellisMad, 5, 1, 16, true }, { kTrellisMad, 6, 1, 16, true }, { kTrellisMad, 7, 1, 16, true },
};

}  // namespace

TEST(iqr_relayout_every_form_matches_the_header_and_inverts) {
    for (const Case& c : kForms) check_form(c, 512, 384);
}

/* The trellis tile at every CPU level this host has: each level's vector unpack, transpose and
 * pack against the same reference. */
TEST(iqr_relayout_trellis_matches_at_every_cpu_level) {
    const int top = r4d_hsimd::detect_level();
    for (int lvl = 0; lvl <= top; ++lvl) {
        REQUIRE_EQ(r4d_hsimd::force_level(lvl), lvl);
        for (const Case& c : kForms)
            if (is_trellis(c.kind)) check_form(c, 512, 384);
        check_form({ kTrellisMad, 7, 1, 16, true }, 256, 1024);
    }
    r4d_hsimd::force_level(top);
}

/* The MoE w4 nibble shuffle at every level against the scalar level, both ways, odd tails
 * included, and its inverse. */
TEST(moew4_permute_matches_at_every_cpu_level) {
    const int top = r4d_hsimd::detect_level();
    std::vector<uint8_t> src(4100), want(4100), got(4100), back(4100);
    for (auto& b : src) b = (uint8_t)rnd();
    for (int64_t n : { (int64_t)4, (int64_t)12, (int64_t)60, (int64_t)68, (int64_t)132, (int64_t)4100 })
        for (int fwd = 0; fwd < 2; ++fwd) {
            r4d_hsimd::force_level(0);
            r4d_hsimd::moew4_permute(src.data(), want.data(), n, fwd);
            for (int lvl = 1; lvl <= top; ++lvl) {
                r4d_hsimd::force_level(lvl);
                std::fill(got.begin(), got.end(), 0);
                r4d_hsimd::moew4_permute(src.data(), got.data(), n, fwd);
                CHECK(std::memcmp(got.data(), want.data(), (size_t)n) == 0);
                r4d_hsimd::moew4_permute(got.data(), back.data(), n, !fwd);
                CHECK(std::memcmp(back.data(), src.data(), (size_t)n) == 0);
            }
        }
    /* Forward is the canonical-to-kernel order: byte b of a dword holds element b low and b + 4
     * high. */
    const uint8_t dw[4] = { 0x10, 0x32, 0x54, 0x76 };   /* elements 0..7 in order */
    uint8_t out[4];
    for (int lvl = 0; lvl <= top; ++lvl) {
        r4d_hsimd::force_level(lvl);
        r4d_hsimd::moew4_permute(dw, out, 4, true);
        CHECK_EQ(out[0], 0x40);
        CHECK_EQ(out[1], 0x51);
        CHECK_EQ(out[2], 0x62);
        CHECK_EQ(out[3], 0x73);
    }
    r4d_hsimd::force_level(top);
}

/* Large enough that the relayout splits the weight over threads. */
TEST(iqr_relayout_threaded_weight_matches_the_header) {
    check_form({ kTrellisMad, 4, 1, 16, true }, 4096, 16384);
    check_form({ kTrellisMad, 3, 1, 16, true }, 4096, 16384);
}

/* A code no entry answers, or one standing for -128, refuses the weight. */
TEST(iqr_relayout_refuses_unreadable_codes) {
    const int64_t N = 32, K = 256;
    {
        const Case c{ kTable, 3, 1, 0, false };
        W w = make(c, N, K);
        const int ia = w.find("table");
        w.data[(size_t)ia][5] = (uint8_t)(int8_t)-128;
        rad_store_bits(w.data[(size_t)w.find("codes")].data() + 7 * w.row_bytes(w.find("codes")), 3, 200, 5);
        Sel s = select(w, c);
        const std::vector<RadParam> lp = params(N, K, false);
        RadLayout L;
        REQUIRE_EQ(r4d_rad_layout_iqr(lp.data(), (int)lp.size(), 2, &w.enc, s.sel.data(), s.t.data(),
                                      (int)s.t.size(), &L), RAD_OK);
        std::vector<uint8_t> d((size_t)L.bytes);
        CHECK_EQ(r4d_rad_relayout_iqr(lp.data(), (int)lp.size(), 2, &w.enc, s.sel.data(), s.t.data(),
                                      (int)s.t.size(), d.data(), L.bytes), RAD_E_FORMAT);
    }
    {
        /* a table of 6 entries for 3-bit codes: code 7 has none */
        const Case c{ kTable, 3, 1, 0, false };
        W w = make(c, N, K);
        const int ia = w.find("table");
        w.enc.plane[ia].extent[1] = 6;
        w.cols[(size_t)ia] = 6;
        w.data[(size_t)ia].resize(6);
        uint8_t* codes = w.data[(size_t)w.find("codes")].data();
        for (int64_t i = 0; i < N * K; ++i) rad_store_bits(codes, 3, i, (uint32_t)(i % 6));
        rad_store_bits(codes, 3, 31 * K + 255, 7);
        Sel s = select(w, c);
        const std::vector<RadParam> lp = params(N, K, false);
        RadLayout L;
        REQUIRE_EQ(r4d_rad_layout_iqr(lp.data(), (int)lp.size(), 2, &w.enc, s.sel.data(), s.t.data(),
                                      (int)s.t.size(), &L), RAD_OK);
        std::vector<uint8_t> d((size_t)L.bytes);
        CHECK_EQ(r4d_rad_relayout_iqr(lp.data(), (int)lp.size(), 2, &w.enc, s.sel.data(), s.t.data(),
                                      (int)s.t.size(), d.data(), L.bytes), RAD_E_FORMAT);
        rad_store_bits(codes, 3, 31 * K + 255, 2);
        CHECK_EQ(r4d_rad_relayout_iqr(lp.data(), (int)lp.size(), 2, &w.enc, s.sel.data(), s.t.data(),
                                      (int)s.t.size(), d.data(), L.bytes), RAD_OK);
    }
}

RAD_TEST_MAIN()

/* w2a8's stored order, a code at a time, as r4d_layout.cpp's w2_slot. */
static void w2a8_reference(const uint8_t* codes, int64_t N, int64_t K, uint32_t* wq) {
    const int64_t kpb = K / 128;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const int64_t t = n / 16, r = n % 16, kb = k / 128, kin = k % 128;
            const int64_t step = kin / 16, d = step / 2, h = step % 2;
            const int64_t kk = kin % 16, lh = (kk % 8) / 4, e = (kk / 8) * 4 + (kk % 4);
            const int f = (int)(4 * (e % 4) + (e / 4) + 2 * h);
            const int64_t w = ((t * kpb + kb) * 32 + lh * 16 + r) * 4 + d;
            wq[w] |= (uint32_t)((codes[n * (K / 4) + (k >> 2)] >> (2 * (k & 3))) & 3) << (2 * f);
        }
}

TEST(w2a8_tiles_match_the_slot_order_at_every_cpu_level) {
    const int top = r4d_hsimd::detect_level();
    /* An odd unit count, so the widest level's two-unit step leaves one. */
    for (const int64_t K : { (int64_t)128, (int64_t)384, (int64_t)1024 }) {
        const int64_t N = 48, rb = K / 4;
        std::vector<uint8_t> codes((size_t)(N * rb));
        uint32_t x = 2463534242u;
        for (uint8_t& c : codes) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; c = (uint8_t)x; }
        std::vector<uint32_t> want((size_t)(N * K / 16), 0);
        w2a8_reference(codes.data(), N, K, want.data());
        for (int lvl = 0; lvl <= top; ++lvl) {
            REQUIRE_EQ(r4d_hsimd::force_level(lvl), lvl);
            std::vector<uint32_t> got(want.size(), 0xDEADBEEFu);
            for (int64_t t = 0; t < N / 16; ++t)
                r4d_hsimd::w2a8_tile(codes.data() + t * 16 * rb, rb, K / 128,
                                     (uint8_t*)got.data() + t * 16 * rb);
            CHECK_EQ(std::memcmp(got.data(), want.data(), want.size() * 4), 0);
        }
    }
    r4d_hsimd::force_level(top);
}

TEST(dword_transpose16_matches_at_every_cpu_level) {
    const int top = r4d_hsimd::detect_level();
    /* Column counts that leave a tail at every vector width. */
    for (const int64_t cols : { (int64_t)1, (int64_t)7, (int64_t)12, (int64_t)36, (int64_t)640 }) {
        const int64_t stride = cols + 3;
        std::vector<uint32_t> in((size_t)(16 * stride));
        for (size_t i = 0; i < in.size(); ++i) in[i] = (uint32_t)(i * 2654435761u);
        std::vector<uint32_t> want((size_t)(16 * cols));
        for (int64_t c = 0; c < cols; ++c)
            for (int r = 0; r < 16; ++r) want[(size_t)(16 * c + r)] = in[(size_t)(r * stride + c)];
        for (int lvl = 0; lvl <= top; ++lvl) {
            REQUIRE_EQ(r4d_hsimd::force_level(lvl), lvl);
            std::vector<uint32_t> got(want.size(), 0xDEADBEEFu);
            r4d_hsimd::dword_transpose16(in.data(), stride, cols, got.data());
            CHECK(got == want);
        }
    }
    r4d_hsimd::force_level(top);
}

TEST(fp8_frag_tiles_match_the_fragment_order_at_every_cpu_level) {
    const int top = r4d_hsimd::detect_level();
    for (const int64_t K : { (int64_t)64, (int64_t)192, (int64_t)5120 }) {
        const int64_t ksuper = K / r4d_fp8::kSuperK;
        std::vector<uint8_t> in((size_t)(16 * K));
        for (size_t i = 0; i < in.size(); ++i) in[i] = (uint8_t)(i * 2654435761u >> 13);
        std::vector<uint8_t> want(in.size());
        int64_t at = 0;
        for (int64_t ks = 0; ks < ksuper; ++ks)
            for (int lane = 0; lane < 32; ++lane)
                for (int t = 0; t < 4; ++t)
                    for (int j = 0; j < 8; ++j, ++at) {
                        int64_t row, k;
                        r4d_fp8::frag_src(0, ks, lane, t, j, &row, &k);
                        want[(size_t)at] = in[(size_t)(row * K + k)];
                    }
        for (int lvl = 0; lvl <= top; ++lvl) {
            REQUIRE_EQ(r4d_hsimd::force_level(lvl), lvl);
            std::vector<uint8_t> got(in.size(), 0xA5);
            r4d_hsimd::fp8_frag_tile(in.data(), K, ksuper, got.data());
            CHECK(got == want);
        }
    }
    r4d_hsimd::force_level(top);
}

TEST(byte_transpose_matches_at_every_cpu_level) {
    const int top = r4d_hsimd::detect_level();
    for (const auto& [rows, cols] : std::vector<std::pair<int64_t, int64_t>>{
             { 1, 1 }, { 5, 40 }, { 16, 16 }, { 48, 160 }, { 37, 70 }, { 272, 33 } }) {
        std::vector<uint8_t> in((size_t)(rows * cols));
        for (size_t i = 0; i < in.size(); ++i) in[i] = (uint8_t)(i * 167 + 13);
        std::vector<uint8_t> want(in.size());
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t c = 0; c < cols; ++c) want[(size_t)(c * rows + r)] = in[(size_t)(r * cols + c)];
        for (int lvl = 0; lvl <= top; ++lvl) {
            REQUIRE_EQ(r4d_hsimd::force_level(lvl), lvl);
            std::vector<uint8_t> got(in.size(), 0xA5);
            r4d_hsimd::byte_transpose(in.data(), rows, cols, got.data());
            CHECK(got == want);
        }
    }
    r4d_hsimd::force_level(top);
}
