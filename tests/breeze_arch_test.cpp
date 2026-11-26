/* Recording tests of Breeze's reference encoder and prompt assembly; no weights or GPU. */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include <dlfcn.h>
#include "../arch/breeze/breeze.cpp"

extern "C" int rad_declare_audio_generator(RadBuilder*, const RadAudioGenDecl*) { return RAD_OK; }

namespace {
struct Fixture {
    RadBuilder b;
    breeze::Model m;
    Fixture() {
        m.gl.max_tok = 1024;
        m.gl.n_embd = 2048;
        m.ge = m.gl;
        m.b_x = rad::arch::decl_b(&b, "lm.x", RAD_BF16, {1024, 2048});
        m.w_cb = rad::arch::decl_w(&b, "frame.codebooks.weight", RAD_BF16, {16 * 2051, 2048},
                                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, rad::arch::grp_layer(0));
        const char* keys[] = {"text_encoder_config.layer_types", "text_encoder_config.num_hidden_layers",
            "audio_tokenizer.encoder_config.hidden_size", "audio_tokenizer.encoder_config.num_hidden_layers",
            "audio_tokenizer.encoder_config.num_filters"};
        const char* vals[] = {"full_attention", "1", "512", "8", "64"};
        RadModelMeta meta{};
        meta.n_kv = 5; meta.kv_key = keys; meta.kv_val = vals;
        CHECK_OK(breeze::declare_encoder(&b, &meta, m));
        CHECK_OK(breeze::declare_reference(&b, &meta, m));
    }
};
}

TEST(breeze_clone_encodes_text_segments_independently_and_audio_only_once) {
    Fixture f;
    /* Three text rows, two reference frames, one audio EOS, three new text rows, twice. */
    int32_t tokens[18]{}, positions[18]{}, cu[] = {0, 9, 18}, ref_cu[] = {0, 3840};
    int32_t map[] = {-1, -1, -1, 0, 1, -1, -1, -1, -1,
                     -1, -1, -1, 0, 1, -1, -1, -1, -1};
    int32_t last[] = {0};
    float audio = 0;
    RadAudioGenBatch A{};
    A.n_lane = 1; A.last_h = last; A.n_ref = 1; A.ref_samples = 3840;
    A.ref_cu = ref_cu; A.ref_cu_h = ref_cu; A.ref_audio = &audio; A.ref_frame_h = map;
    A.prompt_cu_h = cu;
    RadBatch batch{};
    batch.gen = RAD_GEN_PASS_PROMPT; batch.n_tok = 18; batch.n_seq = 2;
    batch.token_ids = tokens; batch.positions = positions; batch.cu_seqlens = cu; batch.agen_b = &A;
    RadCtx c; c.b = &f.b; c.batch = &batch;
    breeze::step_prompt(&c, &batch, f.m);
    int text_segments = 0, reference_encodes = 0, quantisers = 0, embeddings = 0;
    std::vector<int64_t> output_offsets;
    for (const auto& issue : c.issues) {
        if (issue.op == f.m.op_eembed) {
            ++text_segments;
            CHECK_EQ(issue.n, (int64_t)3);
        }
        if (issue.op == f.m.enc[0].op_qknr)
            CHECK_EQ(issue.opd[1].handle, f.m.ref.text_pos);
        if (issue.op == f.m.op_eproj) output_offsets.push_back(issue.opd.back().offset / 2048);
        if (issue.op == f.m.ref.first.op) ++reference_encodes;
        if (f.b.ops[issue.op - 1].op == "vq_euclidean") {
            ++quantisers; CHECK_EQ(issue.n, (int64_t)2);
        }
        if (issue.op == f.m.ref.embed) {
            ++embeddings;
            /* rvq_embed's second table starts at codebook 1, not at codebook 0.
             * The same slice is required for the reference frames and their audio EOS. */
            CHECK_EQ(issue.opd[1].handle, f.m.w_cb);
            CHECK_EQ(issue.opd[1].offset, (int64_t)0);
            CHECK_EQ(issue.opd[2].handle, f.m.w_cb);
            CHECK_EQ(issue.opd[2].offset, (int64_t)2051 * 2048);
            CHECK_EQ(issue.opd[2].rows, (int64_t)15 * 2051);
        }
    }
    CHECK_EQ(text_segments, 4);
    CHECK_EQ(reference_encodes, 1);
    CHECK_EQ(quantisers, 16);
    CHECK_EQ(embeddings, 4);  /* a clip and its EOS in each branch */
    CHECK(output_offsets == std::vector<int64_t>({0, 6, 9, 15}));
    CHECK_LINT(f.b, &c);
}

TEST(breeze_reference_transcripts_cannot_inject_audio_or_template_markers) {
    char out[128];
    CHECK_EQ(breeze::text_prepare(RAD_TEXT_REFERENCE, "the reference words", out, sizeof(out)), 19);
    for (const char* marker : {"<bos>", "<eos>", "<|AUDIO|>", "<|audio_eos|>", "<ins_bos>", "<ins_eos>"})
        CHECK_EQ(breeze::text_prepare(RAD_TEXT_REFERENCE, marker, out, sizeof(out)), (int64_t)RAD_E_INVAL);
}

TEST(breeze_reference_direction_uses_actual_unequal_branch_boundaries) {
    Fixture f;
    int32_t tokens[22]{}, positions[22]{}, cu[] = {0, 13, 22}, ref_cu[] = {0, 3840};
    int32_t map[] = {-1, -1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, 0, 1, -1, -1, -1, -1};
    int32_t last[] = {0}; float audio = 0;
    RadAudioGenBatch A{};
    A.n_lane = 1; A.last_h = last; A.n_ref = 1; A.ref_samples = 3840;
    A.ref_cu = ref_cu; A.ref_cu_h = ref_cu; A.ref_audio = &audio;
    A.ref_frame_h = map; A.prompt_cu_h = cu;
    RadBatch batch{};
    batch.gen = RAD_GEN_PASS_PROMPT; batch.n_tok = 22; batch.n_seq = 2;
    batch.token_ids = tokens; batch.positions = positions; batch.cu_seqlens = cu; batch.agen_b = &A;
    RadCtx c; c.b = &f.b; c.batch = &batch;
    breeze::step_prompt(&c, &batch, f.m);
    std::vector<int64_t> lengths, offsets;
    int encodes = 0;
    for (const auto& issue : c.issues) {
        if (issue.op == f.m.op_eembed) lengths.push_back(issue.n);
        if (issue.op == f.m.op_eproj) offsets.push_back(issue.opd.back().offset / 2048);
        if (issue.op == f.m.ref.first.op) ++encodes;
    }
    CHECK(lengths == std::vector<int64_t>({3, 7, 3, 3}));
    CHECK(offsets == std::vector<int64_t>({0, 6, 13, 19}));
    CHECK_EQ(encodes, 1);
    CHECK_LINT(f.b, &c);
}

TEST(breeze_reference_declarations_use_the_real_kernel_vocabulary) {
    const char* home = std::getenv("RADIANCE_HOME");
    REQUIRE(home != nullptr);
    void* lib = dlopen((std::string(home) + "/kernels/libref.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    REQUIRE(lib != nullptr);
    auto count = (int (*)())dlsym(lib, "rad_kernel_schema_count");
    auto at = (const RadOpSchema* (*)(int))dlsym(lib, "rad_kernel_schema_at");
    REQUIRE(count && at);
    std::map<std::string, const RadOpSchema*> schemas;
    for (int i = 0; i < count(); ++i) schemas[at(i)->op] = at(i);
    Fixture f;
    for (const auto& op : f.b.ops) {
        REQUIRE(schemas.count(op.op) != 0);
        const auto* s = schemas.at(op.op);
        for (const auto& p : op.p) {
            bool known = false;
            for (int i = 0; i < s->n_params; ++i) known |= p.key == s->params[i].key;
            CHECK(known);
        }
        for (int i = 0; i < s->n_params; ++i) {
            if (!s->params[i].required) continue;
            CHECK(std::any_of(op.p.begin(), op.p.end(), [&](const auto& p) { return p.key == s->params[i].key; }));
        }
        size_t min_w = 0, max_w = 0;
        for (int i = 0; i < s->n_operands; ++i) if (s->operands[i].role == RAD_OPD_WEIGHT) {
            ++max_w; if (!s->operands[i].optional) ++min_w;
        }
        CHECK(op.w.size() >= min_w && op.w.size() <= max_w);
        for (auto w : op.w) CHECK(w != 0 && w <= f.b.weights.size());
    }
    dlclose(lib);
}

RAD_TEST_MAIN()
