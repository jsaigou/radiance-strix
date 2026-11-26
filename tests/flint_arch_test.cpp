/* Flint's declaration and step (arch/flint), without a GPU or checkpoint: the containers its
 * recipes make, both KV caches, one and two ranks. */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "rad_plugin.h"
#include "../arch/flint/flint.cpp"

using namespace rad_fake;

namespace {

/* Recursive Flint's configuration as rad-convert records it. */
MetaKV flint_meta() {
    RadModelMeta m{};
    m.arch_id = "flint";
    m.name = "test-flint";
    m.quant = "";
    m.n_layers = 49;
    m.n_embd = 2048;
    m.n_head = 32;
    m.n_head_kv = 4;
    m.head_dim = 128;
    m.n_ff = 6144;
    m.n_vocab = 150144;
    m.n_ctx_train = 262144;
    m.rms_eps = 1e-6f;
    m.rope_theta = 10000.0f;
    m.n_expert = 256;
    m.n_expert_used = 6;
    MetaKV k(m);
    k.set("sliding_window", "513").set("num_experts", "256").set("num_experts_per_tok", "6")
     .set("moe_intermediate_size", "512").set("shared_expert_intermediate_size", "1024")
     .set("intermediate_size", "6144");
    return k;
}

/* IQ8R as libquant states it: int8 codes rotated by fwht128, a u8 scale per [1 x 256] of K under
 * a bf16 scale a row. */
RadEncoding iq8r() {
    RadEncoding e = rad_enc_affine(RAD_I8, RAD_U8, 1, 256);
    rad_enc_add_plane(&e, "scale.1", RAD_BF16, 1, 0);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}

/* Int8 rotated by fwht128 at a bf16 scale a (row, 128 of K): the experts' form. */
RadEncoding i8h() {
    RadEncoding e = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}

/* data/recipes/flint-iq8r.recipe's encodings by declared name. The head comes last because
 * `output.weight` is in `attn_output.weight`. */
void iq8r_encs(RadBuilder* b) {
    b->encs.clear();
    for (const char* t : { "ffn_gate_up_exps", "ffn_down_exps", "ffn_down_shexp" })
        b->encs.push_back({ t, i8h() });
    for (const char* t : { "attn_q.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ffn_gate_up.weight", "ffn_down.weight",
                           "ffn_gate_up_shexp" })
        b->encs.push_back({ t, iq8r() });
    b->encs.push_back({ "output.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 128) });
}

/* data/recipes/flint-bf16-q4.recipe's: u4 attention under an f16 scale and a u8 zero a 128,
 * plain int8 dense and shared linears and head, signed int4 experts. */
void q4_encs(RadBuilder* b) {
    RadEncoding u4 = rad_enc_affine(RAD_U4, RAD_F16, 1, 128);
    rad_enc_add_plane(&u4, "zero", RAD_U8, 1, 128);
    const RadEncoding w8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    const RadEncoding i4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
    b->encs.clear();
    b->encs.push_back({ "ffn_gate_up_exps", i4 });
    b->encs.push_back({ "ffn_down_exps", i4 });
    for (const char* t : { "attn_q.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight" })
        b->encs.push_back({ t, u4 });
    for (const char* t : { "ffn_gate_up_shexp", "ffn_down_shexp", "ffn_gate_up.weight",
                           "ffn_down.weight", "output.weight" })
        b->encs.push_back({ t, w8 });
}

bool is_full(int l) { return l == 0 || l == 1 || l == 48 || (l >= 5 && l <= 45 && (l - 5) % 4 == 0); }

/* Decode, a fused-prologue prefill and one past the fused band, each lint clean. */
void steps_lint_clean(RadBuilder& b, int rank, int world) {
    RadBatch d = decode_batch();
    RadCtx c1{.b = &b, .batch = &d, .rank = rank, .world = world};
    flint::step(&c1, &d);
    CHECK_LINT(b, &c1);
    CHECK(c1.step_fail.empty());
    RadBatch p = prefill_batch();
    RadCtx c2{.b = &b, .batch = &p, .rank = rank, .world = world};
    flint::step(&c2, &p);
    CHECK_LINT(b, &c2);
    CHECK(c2.step_fail.empty());
    RadBatch w = prefill_batch();
    w.n_tok = 200;
    w.max_q_len = 100;
    RadCtx c3{.b = &b, .batch = &w, .rank = rank, .world = world};
    flint::step(&c3, &w);
    CHECK_LINT(b, &c3);
    CHECK(c3.step_fail.empty());
}

}  /* namespace */

/* The attention of every layer: the sink op, the 513 window on the sliding layers, and the fused
 * prologue with the rotary on the sliding layers and none (`rotary_dim` 0) on the full ones, on
 * the bf16 and the int8 cache; the two KV groups take the cache's dtype. */
TEST(attention_takes_the_sink_window_and_nope_prologue_on_both_caches) {
    for (const bool i8 : { false, true }) {
        MetaKV mk = flint_meta();
        RadBuilder b;
        iq8r_encs(&b);
        RadBuildCtx ctx = build_ctx();
        if (i8) ctx.kv_dtype = RAD_I8;
        REQUIRE_EQ(flint::declare(&b, mk.meta(), &ctx), RAD_OK);
        const char* kvn = i8 ? "i8" : "bf16";
        REQUIRE_EQ(b.kv_groups.size(), (size_t)2);
        for (const auto& g : b.kv_groups) CHECK_EQ(g.d.dtype, (uint32_t)(i8 ? RAD_I8 : RAD_BF16));
        REQUIRE_EQ(count_op(b, "attn_paged_sink"), 49);
        REQUIRE_EQ(count_op(b, "qkv_norm_rope_kv_store"), 49);
        CHECK_EQ(count_op(b, "attn_paged"), 0);
        CHECK_EQ(count_op(b, "rope"), 2 * 35);
        for (int l = 0; l < 49; ++l) {
            const RecOp* a = find_op(b, "attn_paged_sink", l);
            const RecOp* f = find_op(b, "qkv_norm_rope_kv_store", l);
            const RecOp* s = find_op(b, "kv_store", l);
            REQUIRE(a && f && s);
            CHECK_EQ(pint(a, "window"), is_full(l) ? 0 : 513);
            CHECK_EQ(pint(a, "gqa"), 8);
            CHECK_EQ(pint(a, "causal"), 1);
            CHECK_EQ(pstr(a, "kv_dtype"), std::string(kvn));
            CHECK_EQ(pint(f, "rotary_dim"), is_full(l) ? 0 : 128);
            CHECK_EQ(pstr(f, "kv_dtype"), std::string(kvn));
            CHECK_EQ(pstr(s, "kv_dtype"), std::string(kvn));
            CHECK(param(s, "k_scale") == nullptr);
        }
        steps_lint_clean(b, 0, 1);
    }
}

/* The IQ8R container: every trunk linear is an i8a8h GEMM, every grouped GEMM the int8 row, gate_up
 * reading the block input's own rotated codes at IQ8R's 256 and down the gated product's at a 128;
 * the norms, the gated products and the attention output write those codes, so no linear declares
 * a quantiser and no E4M3 code is written. One rank and both ranks of two. */
TEST(iq8r_container_reads_the_trunks_rotated_codes_everywhere) {
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            MetaKV mk = flint_meta();
            RadBuilder b;
            iq8r_encs(&b);
            RadBuildCtx ctx = build_ctx(rank, world);
            REQUIRE_EQ(flint::declare(&b, mk.meta(), &ctx), RAD_OK);
            const flint::Model& m = flint::g_model[rank];
            CHECK(m.trunk_rot);
            CHECK_EQ(m.trunk_g8, 256);
            const rad_buf hq = (rad_buf)m.h.q8, fq = (rad_buf)m.mw.eff.q8;
            REQUIRE(hq != 0);
            REQUIRE(fq != 0);
            int gu = 0, dn = 0;
            for (const auto& o : b.ops) {
                if (o.op == "gemm_nt_q") CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
                if (o.op != "moe_gemm_q") continue;
                CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
                const bool token = pstr(&o, "a_order") == "token";
                CHECK_EQ(pint(&o, "a_group"), token ? 256 : 128);
                CHECK(in(o.reads, token ? hq : fq));
                (token ? gu : dn) += 1;
            }
            CHECK_EQ(gu, 47);
            CHECK_EQ(dn, 47);
            /* q|k|v, o, the dense pair twice, the shared pair 47 times. */
            CHECK_EQ(count_op(b, "gemm_nt_q"), 49 * 4 + 2 * 2 + 47 * 2);
            CHECK_EQ(count_op(b, "gated_had_quant_i8g"), 2 + 47 + 47);
            CHECK_EQ(count_op(b, "had_quant_act_i8g"), 49);
            for (const char* op : { "gated_quant_fp8", "quant_act_fp8", "rmsnorm_quant_fp8",
                                    "ar_rmsnorm_quant_fp8", "quant_act_i8g" })
                CHECK_EQ(count_op(b, op), 0);
            CHECK_EQ(count_op(b, world == 2 ? "ar_rmsnorm_had_quant_i8g" : "rmsnorm_had_quant_i8g") > 0,
                     true);
            for (const char* a : { "h.q", "moe_ffn.q", "moe_hr.q8", "shared_ffn.q", "attn_out.q" })
                CHECK(find_b(b, a) == nullptr);
            /* Expert parallel: each rank owns 128 whole experts. */
            CHECK(find_w(b, rank == 0 ? "blk.2.ffn_gate_up_exps.0.weight"
                                      : "blk.2.ffn_gate_up_exps.128.weight") != nullptr);
            steps_lint_clean(b, rank, world);
        }
}

/* Where the rotating norm is not served (the CPU kernels), every linear quantises `h` itself and
 * the routed experts read a rotated int8 copy that had_quant_act_i8g writes once a layer. */
TEST(iq8r_container_quantises_the_expert_input_without_the_rotating_norm) {
    MetaKV mk = flint_meta();
    RadBuilder b;
    for (const char* op : { "rmsnorm_had_quant_i8g", "ar_rmsnorm_had_quant_i8g" })
        b.undeclared.insert(op);
    iq8r_encs(&b);
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(flint::declare(&b, mk.meta(), &ctx), RAD_OK);
    const RecBuf* hr = find_b(b, "moe_hr.q8");
    REQUIRE(hr != nullptr);
    const rad_buf hrb = (rad_buf)flint::g_model[0].hr.q8;
    int reads = 0;
    for (const auto& o : b.ops)
        if (o.op == "moe_gemm_q" && pstr(&o, "a_order") == "token") {
            CHECK_EQ(pint(&o, "a_group"), 128);
            reads += in(o.reads, hrb);
        }
    CHECK_EQ(reads, 47);
    steps_lint_clean(b, 0, 1);
}

/* The four-bit container: u4 attention over the bf16 plane, four-bit experts over the block
 * input's E4M3 pair (w4a8), plain int8 dense and shared linears each quantising its own input,
 * an int8 head. */
TEST(q4_container_serves_four_bit_experts_and_plain_int8_linears) {
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            MetaKV mk = flint_meta();
            RadBuilder b;
            q4_encs(&b);
            RadBuildCtx ctx = build_ctx(rank, world);
            REQUIRE_EQ(flint::declare(&b, mk.meta(), &ctx), RAD_OK);
            CHECK(!flint::g_model[rank].trunk_rot);
            int moe = 0, asym = 0, i8 = 0;
            for (const auto& o : b.ops) {
                if (o.op == "moe_gemm_q") {
                    CHECK_EQ(pstr(&o, "dtype"), std::string("w4a8"));
                    ++moe;
                }
                if (o.op == "gemm_nt_q") {
                    const std::string dt = pstr(&o, "dtype");
                    asym += dt == "w4a16_asym" || dt == "w4a8i8_asym";
                    i8 += dt == "i8a8";
                }
            }
            CHECK_EQ(moe, 2 * 47);
            CHECK(asym >= 4 * 49);
            CHECK_EQ(i8, 2 * 2 + 2 * 47);
            CHECK(count_op(b, world == 2 ? "ar_rmsnorm_quant_fp8" : "rmsnorm_quant_fp8") > 0);
            /* No attention output quantiser: the u4 out projection reads the bf16 plane. */
            CHECK_EQ(count_op(b, "quant_act_fp8"), 0);
            const RecWeight* head = find_w(b, "output.weight");
            REQUIRE(head != nullptr);
            CHECK_EQ(head->d.dtype, (uint32_t)RAD_I8);
            steps_lint_clean(b, rank, world);
        }
}

/* What the plugin refuses by name: a container whose experts are plain (unrotated) int8, a rotated
 * trunk over unrotated four-bit experts, rotated int8 experts under an unrotated trunk, and a
 * cache other than bf16 or int8. */
TEST(refuses_mismatched_forms_and_other_caches) {
    {
        MetaKV mk = flint_meta();
        RadBuilder b;
        iq8r_encs(&b);
        const RadEncoding w8 = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
        for (auto& e : b.encs) e.second = w8;
        RadBuildCtx ctx = build_ctx();
        CHECK(stderr_of_rc([&] { return flint::declare(&b, mk.meta(), &ctx); }) < 0);
    }
    {
        MetaKV mk = flint_meta();
        RadBuilder b;
        iq8r_encs(&b);
        const RadEncoding i4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);
        for (auto& e : b.encs)
            if (e.first == "ffn_gate_up_exps" || e.first == "ffn_down_exps") e.second = i4;
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(stderr_of_rc([&] { return flint::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {
        MetaKV mk = flint_meta();
        RadBuilder b;
        q4_encs(&b);
        for (auto& e : b.encs)
            if (e.first == "ffn_gate_up_exps" || e.first == "ffn_down_exps") e.second = i8h();
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(stderr_of_rc([&] { return flint::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {
        MetaKV mk = flint_meta();
        RadBuilder b;
        iq8r_encs(&b);
        RadBuildCtx ctx = build_ctx();
        ctx.kv_dtype = RAD_F8E4M3;
        CHECK_EQ(stderr_of_rc([&] { return flint::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
}

/* Per-instance state lives in the declaring instance's slot, so a second engine in the process
 * leaves the first one's model alone. */
TEST(declares_into_its_own_slot) {
    MetaKV mk = flint_meta();
    RadBuilder b0, b1;
    iq8r_encs(&b0);
    q4_encs(&b1);
    RadBuildCtx c0 = build_ctx();
    RadBuildCtx c1 = build_ctx();
    c1.slot_base = 4;
    REQUIRE_EQ(flint::declare(&b0, mk.meta(), &c0), RAD_OK);
    REQUIRE_EQ(flint::declare(&b1, mk.meta(), &c1), RAD_OK);
    RadArchProbe probe{};
    CHECK_OK(flint::probe(mk.meta(), &probe));
    CHECK_EQ(probe.draft_depth, 0);
    CHECK(flint::g_model[0].trunk_rot);
    CHECK(!flint::g_model[4].trunk_rot);
    CHECK_EQ(flint::g_model[4].g.n_layer, 49);
}

RAD_TEST_MAIN()
