/* Breeze's Qwen3-TTS/Mimi reference encoder. Included inside namespace breeze after the
 * ordinary networks. Kept in f32, as the released audio tokenizer encodes references. */
static rad_op ref_rows(RadBuilder* b, const char* op, int64_t rows, int64_t n,
                       std::initializer_list<rad_buf> in, std::initializer_list<rad_buf> out,
                       const char* dtype = "f32") {
    return rw(b, RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                         RAD_STR("dtype", dtype)), RAD_NOWEIGHTS), in, out);
}

static rad_op ref_gemm(RadBuilder* b, int64_t rows, int64_t n, int64_t k, rad_weight w,
                       rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "gemm_nt", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", n),
                        RAD_INT("K", k), RAD_STR("dtype", "f32")), RAD_WEIGHTS(w)), {x}, {y});
}

static int ref_conv(RadBuilder* b, Model& m, RefConv& C, const char* name, const char* src,
                    int64_t ci, int64_t co, int64_t k, int64_t st, int64_t rate, int64_t rows,
                    rad_buf x, rad_buf res, rad_buf y, bool bias = true) {
    /* The first convolution's weight is mapped by reference_stored when that ran. */
    C.w = (m.ref_stored && &C == &m.ref.first)
              ? decl_w(b, m.nm.f("ref.%s.weight", name), RAD_F32, {co, ci, k}, RAD_ACCESS_PER_TOKEN,
                       RAD_SHARD_NONE, grp_layer(kGrpCodec + 1))
              : wcopy(b, m.nm, m.nm.f("ref.%s.weight", name), m.nm.ckpt("%s.weight", src),
                      RAD_F32, {co, ci, k}, kGrpCodec + 1);
    if (bias) C.b = wcopy(b, m.nm, m.nm.f("ref.%s.bias", name), m.nm.ckpt("%s.bias", src),
                          RAD_F32, {co}, kGrpCodec + 1);
    const rad_weight weights[] = {C.w, C.b};
    C.op = rw(b, RAD_OP(b, "conv1d", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("cin", ci),
                        RAD_INT("cout", co), RAD_INT("k", k), RAD_INT("dil", 1),
                        RAD_INT("causal", 1), RAD_INT("stride", st), RAD_INT("cu_div", rate),
                        RAD_STR("dtype", "f32")), weights, bias ? 2 : 1),
              {x, m.ref.cu, res}, {y});
    return C.w && (!bias || C.b) && C.op ? RAD_OK : RAD_E_INVAL;
}

/* Whether the model holds the reference encoder, asked of its first convolution through the name
 * map entry declared here (the one ref_conv would declare). A builder with no model behind it
 * answers no. */
static bool reference_stored(RadBuilder* b, Model& m) {
    const std::string dn = m.nm.f("ref.first.weight");
    if (map_copy(b, dn.c_str(),
                 m.nm.ckpt("%s.weight", "audio_tokenizer.encoder.encoder.layers.0.conv")) < 0)
        return false;
    RadEncoding enc{};
    return weight_enc(b, dn.c_str(), &enc);
}

static int declare_reference(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Reference& E = m.ref;
    Names& nm = m.nm;
    const char* ec = "audio_tokenizer.encoder_config";
    if (rad_meta_geti(meta, nm.f("%s.hidden_size", ec), 0) != 512 ||
        rad_meta_geti(meta, nm.f("%s.num_hidden_layers", ec), 0) != 8 ||
        rad_meta_geti(meta, nm.f("%s.num_filters", ec), 0) != 64 ||
        rad_meta_geti(meta, nm.f("%s.use_causal_conv", ec), 1) != 1 ||
        rad_meta_geti(meta, nm.f("%s.use_conv_shortcut", ec), 0) != 0) {
        fprintf(stderr, "radiance: breeze: reference encoder is not the released Qwen3-TTS encoder\n");
        return RAD_E_UNSUPPORTED;
    }
    constexpr int64_t R = kMaxRefSamples / 960;  /* SEANet's 25 Hz output */
    constexpr int64_t NI = 2 * kMaxPrompt + 3;
    E.index = decl_b(b, nm.f("ref.index"), RAD_I32, {NI}, RAD_BUF_PERSIST);
    E.indices = iota_op(b, NI, 0, 1, E.index);
    E.cu = decl_b(b, nm.f("ref.cu"), RAD_I32, {2}, RAD_BUF_PERSIST);
    E.text_pos = decl_b(b, nm.f("ref.text_pos"), RAD_I32, {kMaxPrompt}, RAD_BUF_PERSIST);
    E.text_cu = decl_b(b, nm.f("ref.text_cu"), RAD_I32, {2}, RAD_BUF_PERSIST);
    E.text_indices = iota_op(b, kMaxPrompt, 0, 1, E.text_pos);
    E.copy_i = rw(b, RAD_OP(b, "int_affine", RAD_PARAMS(RAD_INT("M", 1), RAD_STR("src", "0"),
                    RAD_STR("mul", "1"), RAD_STR("add", "0")), RAD_NOWEIGHTS),
                  {E.index}, {E.cu, E.text_cu});
    E.copy_f = rw(b, RAD_OP(b, "cast", RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", 512),
                    RAD_STR("from", "f32"), RAD_STR("to", "f32")), RAD_NOWEIGHTS), {}, {});
    int64_t rows = kMaxRefSamples, ch = 64, rate = 1;
    static const int64_t stride[] = {4, 5, 6, 8};
    for (int i = 0; i < 5; ++i) {
        E.x[i] = decl_b(b, nm.f("ref.x%d", i), RAD_F32, {rows, ch});
        E.a[i] = decl_b(b, nm.f("ref.a%d", i), RAD_F32, {rows, ch});
        E.elu[i] = ref_rows(b, "elu", rows, ch, {E.x[i]}, {E.a[i]});
        if (i < 4) {
            E.h[i] = decl_b(b, nm.f("ref.h%d", i), RAD_F32, {rows, ch / 2});
            E.elu_h[i] = ref_rows(b, "elu", rows, ch / 2, {E.h[i]}, {E.h[i]});
            rows /= stride[i]; ch *= 2;
        }
    }
    RAD_ARCH_TRY(ref_conv(b, m, E.first, "first", "audio_tokenizer.encoder.encoder.layers.0.conv",
                          1, 64, 7, 1, 1, kMaxRefSamples, 0, 0, E.x[0]));
    rows = kMaxRefSamples; ch = 64; rate = 1;
    for (int i = 0; i < 4; ++i) {
        const int li = 1 + 3 * i;
        RAD_ARCH_TRY(ref_conv(b, m, E.res1[i], nm.f("res%d.1", i),
                              nm.ckpt("audio_tokenizer.encoder.encoder.layers.%d.block.1.conv", li),
                              ch, ch / 2, 3, 1, rate, rows, E.a[i], 0, E.h[i]));
        RAD_ARCH_TRY(ref_conv(b, m, E.res2[i], nm.f("res%d.2", i),
                              nm.ckpt("audio_tokenizer.encoder.encoder.layers.%d.block.3.conv", li),
                              ch / 2, ch, 1, 1, rate, rows, E.h[i], E.x[i], E.x[i]));
        RAD_ARCH_TRY(ref_conv(b, m, E.down[i], nm.f("down%d", i),
                              nm.ckpt("audio_tokenizer.encoder.encoder.layers.%d.conv", li + 2),
                              ch, 2 * ch, 2 * stride[i], stride[i], rate, rows,
                              E.a[i], 0, E.x[i + 1]));
        rows /= stride[i]; rate *= stride[i]; ch *= 2;
    }
    E.z = decl_b(b, nm.f("ref.z"), RAD_F32, {R, 512});
    RAD_ARCH_TRY(ref_conv(b, m, E.last, "last", "audio_tokenizer.encoder.encoder.layers.14.conv",
                          1024, 512, 3, 1, 960, R, E.a[4], 0, E.z));
    E.th = decl_b(b, nm.f("ref.th"), RAD_F32, {R, 512});
    E.qkv = decl_b(b, nm.f("ref.qkv"), RAD_F32, {R, 1536});
    E.attn = decl_b(b, nm.f("ref.attn"), RAD_F32, {R, 512});
    E.o = decl_b(b, nm.f("ref.o"), RAD_F32, {R, 512});
    E.ff = decl_b(b, nm.f("ref.ff"), RAD_F32, {R, 2048});
    E.act = decl_b(b, nm.f("ref.act"), RAD_F32, {R, 2048});
    for (int i = 0; i < 8; ++i) {
        RefLayer& L = E.layer[i];
        auto w = [&](const char* dn, const char* sn, std::initializer_list<int64_t> sh) {
            return wcopy(b, nm, nm.f("ref.t%d.%s", i, dn),
                          nm.ckpt("audio_tokenizer.encoder.encoder_transformer.layers.%d.%s", i, sn),
                          RAD_F32, sh, kGrpCodec + 1);
        };
        L.n1 = w("n1", "input_layernorm.weight", {512});
        L.b1 = w("b1", "input_layernorm.bias", {512});
        L.n2 = w("n2", "post_attention_layernorm.weight", {512});
        L.b2 = w("b2", "post_attention_layernorm.bias", {512});
        L.ls1 = w("ls1", "self_attn_layer_scale.scale", {512});
        L.ls2 = w("ls2", "mlp_layer_scale.scale", {512});
        const char* qn = nm.f("ref.t%d.qkv", i);
        RAD_ARCH_TRY(map_concat(b, qn, {
            nm.ckpt("audio_tokenizer.encoder.encoder_transformer.layers.%d.self_attn.q_proj.weight", i),
            nm.ckpt("audio_tokenizer.encoder.encoder_transformer.layers.%d.self_attn.k_proj.weight", i),
            nm.ckpt("audio_tokenizer.encoder.encoder_transformer.layers.%d.self_attn.v_proj.weight", i)}));
        L.qkv = decl_w(b, qn, RAD_F32, {1536, 512}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                        grp_layer(kGrpCodec + 1));
        L.o = w("out", "self_attn.o_proj.weight", {512, 512});
        L.fc1 = w("fc1", "mlp.fc1.weight", {2048, 512});
        L.fc2 = w("fc2", "mlp.fc2.weight", {512, 2048});
        auto ln = [&](rad_weight nw, rad_weight nb) {
            return rw(b, RAD_OP(b, "layernorm", RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", 512),
                          RAD_F64("eps", 1e-5), RAD_STR("dtype", "f32")), RAD_WEIGHTS(nw, nb)),
                      {E.z}, {E.th});
        };
        L.norm1 = ln(L.n1, L.b1); L.norm2 = ln(L.n2, L.b2);
        L.proj = ref_gemm(b, R, 1536, 512, L.qkv, E.th, E.qkv);
        L.rope = rw(b, RAD_OP(b, "rope", RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", 64),
                         RAD_INT("n_head", 8), RAD_INT("n_head_kv", 8), RAD_F64("theta", 10000),
                         RAD_F64("scale", 1), RAD_STR("mode", "neox")), RAD_NOWEIGHTS),
                    {E.qkv, E.index}, {E.qkv});
        L.attn = rw(b, RAD_OP(b, "attn_dense", RAD_PARAMS(RAD_RANGE("M", 1, R),
                         RAD_INT("head_dim", 64), RAD_INT("gqa", 1), RAD_INT("causal", 1),
                         RAD_F64("scale", 0.125), RAD_INT("window", 250), RAD_INT("max_seqlen", R),
                         RAD_STR("dtype", "f32")), RAD_NOWEIGHTS), {E.qkv, E.cu}, {E.attn});
        L.out = ref_gemm(b, R, 512, 512, L.o, E.attn, E.o);
        L.f1 = ref_gemm(b, R, 2048, 512, L.fc1, E.th, E.ff);
        L.act = ref_rows(b, "gelu", R, 2048, {E.ff}, {E.act});
        L.f2 = ref_gemm(b, R, 512, 2048, L.fc2, E.act, E.o);
        auto scale = [&](rad_weight sw) {
            return rw(b, RAD_OP(b, "mul_weight", RAD_PARAMS(RAD_RANGE("M", 1, R),
                          RAD_INT("n", 512), RAD_STR("dtype", "f32")), RAD_WEIGHTS(sw)), {E.o}, {E.o});
        };
        L.scale1 = scale(L.ls1); L.scale2 = scale(L.ls2);
        L.add1 = ref_rows(b, "add", R, 512, {E.z, E.o}, {E.z});
        L.add2 = ref_rows(b, "add", R, 512, {E.z, E.o}, {E.z});
        if (!L.n1 || !L.b1 || !L.n2 || !L.b2 || !L.qkv || !L.o || !L.fc1 || !L.fc2 ||
            !L.ls1 || !L.ls2 || !L.norm1 || !L.norm2 || !L.proj || !L.rope || !L.attn ||
            !L.out || !L.f1 || !L.act || !L.f2 || !L.scale1 || !L.scale2 || !L.add1 || !L.add2)
            return RAD_E_INVAL;
    }
    /* The last downsample uses replicate padding. Prepending two copies of the first row,
     * using the ordinary causal conv, then dropping its first output is exactly that pad. */
    E.pad = decl_b(b, nm.f("ref.pad"), RAD_F32, {R + 2, 512});
    E.sampled = decl_b(b, nm.f("ref.sampled"), RAD_F32, {kMaxRefFrames + 1, 512});
    RAD_ARCH_TRY(rad_op_reads(b, E.copy_f, &E.z, 1));
    RAD_ARCH_TRY(rad_op_writes(b, E.copy_f, &E.pad, 1));
    RAD_ARCH_TRY(ref_conv(b, m, E.sample, "sample", "audio_tokenizer.encoder.downsample.conv",
                          512, 512, 4, 2, 1, R + 2, E.pad, 0, E.sampled, false));
    E.codes = decl_b(b, nm.f("ref.codes"), RAD_I32, {16, kMaxRefFrames}, RAD_BUF_PERSIST);
    E.eos_codes = decl_b(b, nm.f("ref.eos_codes"), RAD_I32, {16, 1}, RAD_BUF_PERSIST);
    for (int part = 0; part < 2; ++part) {
        const char* qs = part ? "acoustic_residual_vector_quantizer" : "semantic_residual_vector_quantizer";
        E.q[part] = decl_b(b, nm.f("ref.q%d", part), RAD_F32, {kMaxRefFrames, 256});
        E.qi[part] = wcopy(b, nm, nm.f("ref.q%d.in", part),
                           nm.ckpt("audio_tokenizer.encoder.quantizer.%s.input_proj.weight", qs),
                           RAD_F32, {256, 512}, kGrpCodec + 1);
        /* A 1x1 Conv1d weight's trailing singleton is immaterial to GEMM's matrix view. */
        E.project[part] = ref_gemm(b, kMaxRefFrames, 256, 512, E.qi[part], E.sampled, E.q[part]);
        const int first = part ? 1 : 0, count = part ? 15 : 1;
        for (int j = 0; j < count; ++j) {
            const int k = first + j;
            const char* dn = nm.f("ref.cb%d", k);
            RAD_ARCH_TRY(map_ema(b, dn,
                nm.ckpt("audio_tokenizer.encoder.quantizer.%s.layers.%d.codebook.embed_sum", qs, j),
                nm.ckpt("audio_tokenizer.encoder.quantizer.%s.layers.%d.codebook.cluster_usage", qs, j)));
            E.cb[k] = decl_w(b, dn, RAD_F32, {2048, 256}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                             grp_layer(kGrpCodec + 1));
            E.quant[k] = rw(b, RAD_OP(b, "vq_euclidean", RAD_PARAMS(RAD_RANGE("M", 1, kMaxRefFrames),
                                   RAD_INT("n", 256), RAD_INT("bins", 2048), RAD_STR("dtype", "f32")),
                                   RAD_WEIGHTS(E.cb[k])), {E.q[part]}, {E.q[part], E.codes});
            if (!E.cb[k] || !E.quant[k]) return RAD_E_INVAL;
        }
        if (!E.qi[part] || !E.project[part]) return RAD_E_INVAL;
    }
    E.embed = rw(b, RAD_OP(b, "rvq_embed", RAD_PARAMS(RAD_RANGE("M", 1, kMaxRefFrames),
                    RAD_INT("n_embd", m.gl.n_embd), RAD_INT("n_codes", 16), RAD_INT("vocab", kVocab),
                    RAD_INT("off0", 0), RAD_F64("scale", 1), RAD_INT("round", 1),
                    RAD_STR("dtype", "bf16")), RAD_WEIGHTS(m.w_cb, m.w_cb)), {E.codes}, {m.b_x});
    E.eos = iota_op(b, 16, 0, 0, E.eos_codes);
    RAD_ARCH_TRY(rad_op_reads(b, E.embed, &E.eos_codes, 1));
    /* Both branches' text segments are encoded separately at position zero. These buffers must
     * remain live through each segment's encoder, projection and the backbone prefill. */
    for (const EncLayer& L : m.enc) {
        RAD_ARCH_TRY(rad_op_reads(b, L.op_qknr, &E.text_pos, 1));
        RAD_ARCH_TRY(rad_op_reads(b, L.op_attn, &E.text_cu, 1));
    }
    return E.indices && E.copy_i && E.copy_f && E.text_indices && E.embed && E.eos ? RAD_OK : RAD_E_INVAL;
}

/* Copy the endpoints of an iota into a two-entry cu, without a host upload or ABI change. */
static void ref_bounds(RadCtx* c, const Reference& E, rad_buf cu, int64_t len) {
    RAD_ISSUE_N(c, E.copy_i, 1, brow_slice(E.index, 0, 1, 1), brow_slice(cu, 0, 1, 1));
    RAD_ISSUE_N(c, E.copy_i, 1, brow_slice(E.index, len, 1, 1), brow_slice(cu, 1, 1, 1));
}

static void step_reference(RadCtx* c, const RadBatch* batch, const Model& m) {
    const Reference& E = m.ref;
    const auto* A = batch->agen_b;
    const int64_t L = A->ref_samples, R = L / 960, F = R / 2;
    RAD_ISSUE_N(c, E.indices, 2 * kMaxPrompt + 3, brows(E.index, 2 * kMaxPrompt + 3));
    const RadOperand raw_cu = praw(A->ref_cu, RAD_I32, 2);
    auto conv = [&](const RefConv& C, RadOperand x, RadOperand res, RadOperand y,
                    int64_t rows, RadOperand cu) {
        RAD_ISSUE_N(c, C.op, rows, x, RAD_W(C.w), cu, res, y, C.b ? RAD_W(C.b) : RAD_NONE);
    };
    conv(E.first, praw(A->ref_audio, RAD_F32, L), RAD_NONE, brows(E.x[0], L), L, raw_cu);
    static const int64_t stride[] = {4, 5, 6, 8};
    int64_t rows = L;
    for (int i = 0; i < 4; ++i) {
        RAD_ISSUE_N(c, E.elu[i], rows, brows(E.x[i], rows), brows(E.a[i], rows));
        conv(E.res1[i], brows(E.a[i], rows), RAD_NONE, brows(E.h[i], rows), rows, raw_cu);
        RAD_ISSUE_N(c, E.elu_h[i], rows, brows(E.h[i], rows), brows(E.h[i], rows));
        conv(E.res2[i], brows(E.h[i], rows), brows(E.x[i], rows), brows(E.x[i], rows), rows, raw_cu);
        RAD_ISSUE_N(c, E.elu[i], rows, brows(E.x[i], rows), brows(E.a[i], rows));
        conv(E.down[i], brows(E.a[i], rows), RAD_NONE, brows(E.x[i + 1], rows / stride[i]), rows, raw_cu);
        rows /= stride[i];
    }
    RAD_ISSUE_N(c, E.elu[4], R, brows(E.x[4], R), brows(E.a[4], R));
    conv(E.last, brows(E.a[4], R), RAD_NONE, brows(E.z, R), R, raw_cu);
    ref_bounds(c, E, E.cu, R);
    const RadOperand cu = brows(E.cu, 2), pos = brows(E.index, R);
    for (const RefLayer& T : E.layer) {
        auto norm = [&](rad_op op, rad_weight w, rad_weight bias) {
            RAD_ISSUE_N(c, op, R, brows(E.z, R), RAD_W(w), RAD_W(bias), brows(E.th, R));
        };
        auto gemm = [&](rad_op op, rad_buf x, rad_weight w, rad_buf y) {
            RAD_ISSUE_N(c, op, R, brows(x, R), RAD_W(w), brows(y, R));
        };
        norm(T.norm1, T.n1, T.b1);
        gemm(T.proj, E.th, T.qkv, E.qkv);
        RAD_ISSUE_N(c, T.rope, R, brows(E.qkv, R), pos);
        RAD_ISSUE_N(c, T.attn, R, bcol(E.qkv, 0, 512, R), bcol(E.qkv, 512, 512, R),
                    bcol(E.qkv, 1024, 512, R), cu, brows(E.attn, R));
        gemm(T.out, E.attn, T.o, E.o);
        RAD_ISSUE_N(c, T.scale1, R, brows(E.o, R), RAD_W(T.ls1), brows(E.o, R));
        RAD_ISSUE_N(c, T.add1, R, brows(E.z, R), brows(E.o, R), brows(E.z, R));
        norm(T.norm2, T.n2, T.b2);
        gemm(T.f1, E.th, T.fc1, E.ff);
        RAD_ISSUE_N(c, T.act, R, brows(E.ff, R), brows(E.act, R));
        gemm(T.f2, E.act, T.fc2, E.o);
        RAD_ISSUE_N(c, T.scale2, R, brows(E.o, R), RAD_W(T.ls2), brows(E.o, R));
        RAD_ISSUE_N(c, T.add2, R, brows(E.z, R), brows(E.o, R), brows(E.z, R));
    }
    RAD_ISSUE_N(c, E.copy_f, 1, brow_slice(E.z, 0, 1, 512), brow_slice(E.pad, 0, 1, 512));
    RAD_ISSUE_N(c, E.copy_f, 1, brow_slice(E.z, 0, 1, 512), brow_slice(E.pad, 1, 1, 512));
    RAD_ISSUE_N(c, E.copy_f, R, brows(E.z, R), brow_slice(E.pad, 2, R, 512));
    ref_bounds(c, E, E.cu, R + 2);
    conv(E.sample, brows(E.pad, R + 2), RAD_NONE, brows(E.sampled, F + 1), R + 2, cu);
    for (int p = 0; p < 2; ++p) {
        RAD_ISSUE_N(c, E.project[p], F, brow_slice(E.sampled, 1, F, 512), RAD_W(E.qi[p]), brows(E.q[p], F));
        for (int k = p ? 1 : 0; k < (p ? 16 : 1); ++k) {
            RadOperand codes = brow_slice(E.codes, k, 1, kMaxRefFrames);
            codes.cols = F;
            RAD_ISSUE_N(c, E.quant[k], F, brows(E.q[p], F), RAD_W(E.cb[k]), codes);
        }
    }
}
