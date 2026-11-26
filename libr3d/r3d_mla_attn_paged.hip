/* r3d_mla_attn_paged.hip -- mla_attn_paged: dense causal latent attention over a paged cache
 * whose token row is the normed latent followed by the rotated key part (Ling-3.0's MLA).
 *
 *     out[r, head, :] = softmax_j( scale * ( q[r, head, :] . kv[j, :512]
 *                                          + q_rope[r, head, :] . kv[j, 512:576] ) ) . kv[j, :512]
 *
 * over one page table per sequence, in libref's reading: sequence s owns query rows
 * [cu[s], cu[s+1]), its context after this step is seqused[s], and row i of it sits at position
 * seqused[s] - qlen + i and sees keys 0..position. A negative page (an unmapped block) hides its
 * tokens, as attn_paged reads the table. The value is the latent; the rotary part scores only.
 * With `gate`, out = bf16( bf16(out) * bf16(sigmoid(gate[r, head])) ).
 *
 * A wave is a (query row, head). Lane l holds latent columns 16l..16l+15 and rotary columns
 * 2l, 2l+1, so a score is one wave sum: the lane's latent pair of the 512-wide dot plus its rope
 * pair, wave_sum, then scale. The running softmax is online (a rescaling max/den update per
 * token), exactly the libr4d kernel's: a row's bytes do not depend on the batch it is in.
 *
 * A short batch splits its history over the z axis, one (value[512], max, den) f32 partial per
 * (row, head, split) in the scratch arena, and merge_paged joins the partials against the global
 * maximum before the single narrowing. The split count is a function of the head count alone
 * (paged_splits), not of the launch's rows, for the same reason the libr4d kernel states it: a
 * speculative verify scores the k + 1 rows of a sequence in one launch, and greedy output equals
 * the run without speculation only if each row gives the single-row decode at its position bit
 * for bit.
 *
 * Operands: q [M, heads, 512] bf16, q_rope [M, heads, 64] bf16, kv [pages, 1, bs, >= 576] bf16,
 * block_table [seqs, blocks] i32, seqused [seqs] i32, cu_seqlens [seqs + 1] i32, gate [M, >=
 * heads] bf16 (optional), out [M, heads, 512] bf16. A 2-D [M, heads*width] plane is viewed as
 * 3-D, as libr4d.
 *
 * The launcher (r3d_rad_mla_attn_paged) checks: the dtypes above; latent_dim 512, rope_dim 64;
 * heads 1..64; block_size 1..256; scale finite and positive; M 1..65535; the shapes, strides and
 * pitches as libr4d's (last axes contiguous, row pitches at least the dense ones, the table at
 * least as wide as seqused's rows, cu [seqs + 1] contiguous); RAD_E_INVAL, RAD_E_DTYPE,
 * RAD_E_SHAPE. A scratch arena too small for the split lowers the split, down to 1, which needs
 * none -- the band is never refused for lack of scratch.
 *
 * RDNA4 -> RDNA3: the kernel is a bandwidth-bound GEMV shape, so it runs plain vectorised f32
 * with the library's bf16 codecs -- no WMMA (RDNA3's wave matrix is f16/i8 only) and no fp8. The
 * wave sum is wave_sum, the same butterfly order as libr4d's, so a row's bytes are
 * bit-identical to the libr4d kernel. The aligned 16-byte latent loads keep their alignment
 * guard: a misaligned 16-byte load returns garbage on this hardware, and the scalar path takes
 * the misaligned row.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>

namespace {

constexpr int D = 512;          /* the latent width */
constexpr int RD = 64;          /* the rotary width */
constexpr int PART = D + 2;     /* a partial: value[512], max, den */

/* The rows the scratch serves at full split: a verify at depth 31. */
constexpr long kPagedRows = 32;

struct PagedMeta {
    long qr, qh, rr, rh, kp, kt, bp, bc, orr, oh, gr;
    int rows, heads, seqs, blocks, bs, pages, splits;
    float scale;
};

int paged_splits(long heads) { return (int)std::min(16l, (256 + heads - 1) / heads); }

int64_t paged_scratch_bytes(long rows, long heads) {
    return std::min(rows, kPagedRows) * heads * paged_splits(heads) * PART * 4;
}

/* Which sequence owns dense query row `r`, walking the descriptor once per wave. */
__device__ __forceinline__ int paged_seq(const int* cu, int seqs, int r) {
    int s = 0;
    while (s + 1 < seqs && cu[s + 1] <= r) ++s;
    return s;
}

/* Sixteen bf16 elements as f32: two 16-byte loads when the row is aligned, element-wise when it
 * is not (a misaligned 16-byte load returns garbage on this hardware). The 16-byte staging goes
 * through r3d_u32x4, the ext_vector_type the library keeps in registers (r3d_common.h). */
__device__ __forceinline__ void load_latent(const uint16_t* p, float (&v)[16]) {
    if (!((uintptr_t)p & 15)) {
        const r3d_u32x4 a =
            *(const __attribute__((address_space(1))) r3d_u32x4*)(const void*)p;
        const r3d_u32x4 b =
            *(const __attribute__((address_space(1))) r3d_u32x4*)(const void*)(p + 8);
        const uint32_t packed[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
#pragma unroll
        for (int j = 0; j < 16; ++j)
            v[j] = bf16_to_f32((uint16_t)(packed[j / 2] >> (16 * (j & 1))));
    } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = bf16_to_f32((uint16_t)ldg_u16(p + j));
    }
}

/* The narrowed output, with the optional head gate: bf16(bf16(out) * bf16(sigmoid(logit))). */
__device__ __forceinline__ void paged_store(uint16_t* out, const uint16_t* gate, const PagedMeta& m,
                                            int r, int head, int lane, const float (&value)[16],
                                            float den) {
    float g = 1.0f;
    if (gate)
        g = bf16_to_f32(f32_to_bf16(1.0f / (1.0f + expf(-bf16_to_f32((uint16_t)ldg_u16(
                                                                 gate + (long)r * m.gr + head))))));
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        float v = den > 0 ? value[j] / den : 0;
        if (gate) v = bf16_to_f32(f32_to_bf16(v)) * g;
        out[(long)r * m.orr + (long)head * m.oh + lane * 16 + j] = f32_to_bf16(v);
    }
}

__global__ __launch_bounds__(128) R3D_WAVES(4)
void attend_paged(const uint16_t* __restrict__ q, const uint16_t* __restrict__ qrot,
                  const uint16_t* __restrict__ kv, const int* __restrict__ bt,
                  const int* __restrict__ used, const int* __restrict__ cu,
                  const uint16_t* __restrict__ gate, uint16_t* __restrict__ out,
                  float* __restrict__ scratch, PagedMeta m) {
    const int lane = (int)threadIdx.x & 31, head = (int)blockIdx.y * 4 + ((int)threadIdx.x >> 5),
              r = (int)blockIdx.x, split = (int)blockIdx.z;
    if (head >= m.heads) return;
    const int s = paged_seq(cu, m.seqs, r);
    const int qlen = cu[s + 1] - cu[s], i = r - cu[s];
    const int live = std::min(std::max(used[s] - qlen + i + 1, 0), m.blocks * m.bs);
    const int begin = (int)((long)live * split / m.splits),
              end = (int)((long)live * (split + 1) / m.splits);
    float query[16], value[16] = {};
    load_latent(q + (long)r * m.qr + (long)head * m.qh + lane * 16, query);
    const uint16_t* const qp = qrot + (long)r * m.rr + (long)head * m.rh + lane * 2;
    const float rq0 = bf16_to_f32((uint16_t)ldg_u16(qp)),
                rq1 = bf16_to_f32((uint16_t)ldg_u16(qp + 1));
    float max_score = -INFINITY, den = 0;
    for (int token = begin; token < end; ++token) {
        const int page = bt[(long)s * m.bp + (token / m.bs) * m.bc];
        if (page < 0 || page >= m.pages) continue;
        const uint16_t* const row = kv + (long)page * m.kp + (token % m.bs) * m.kt;
        float key[16];
        load_latent(row + lane * 16, key);
        float score = rq0 * bf16_to_f32((uint16_t)ldg_u16(row + D + lane * 2)) +
                      rq1 * bf16_to_f32((uint16_t)ldg_u16(row + D + lane * 2 + 1));
#pragma unroll
        for (int j = 0; j < 16; ++j) score += query[j] * key[j];
        score = wave_sum(score) * m.scale;
        const bool higher = score > max_score;
        const float weight = expf(higher ? max_score - score : score - max_score);
        if (higher) {
#pragma unroll
            for (int j = 0; j < 16; ++j) value[j] = value[j] * weight + key[j];
            den = den * weight + 1;
            max_score = score;
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) value[j] += key[j] * weight;
            den += weight;
        }
    }
    if (m.splits == 1) {
        paged_store(out, gate, m, r, head, lane, value, den);
    } else {
        float* dst = scratch + (((long)r * m.heads + head) * m.splits + split) * PART;
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[lane * 16 + j] = value[j];
        if (!lane) {
            dst[D] = max_score;
            dst[D + 1] = den;
        }
    }
}

__global__ __launch_bounds__(128) R3D_WAVES(4)
void merge_paged(const float* __restrict__ scratch, const uint16_t* __restrict__ gate,
                 uint16_t* __restrict__ out, PagedMeta m) {
    const int lane = (int)threadIdx.x & 31, head = (int)blockIdx.y * 4 + ((int)threadIdx.x >> 5),
              r = (int)blockIdx.x;
    if (head >= m.heads) return;
    const float* src = scratch + ((long)r * m.heads + head) * m.splits * PART;
    float max_score = -INFINITY;
    for (int s = 0; s < m.splits; ++s)
        if (src[s * PART + D + 1] > 0) max_score = fmaxf(max_score, src[s * PART + D]);
    float value[16] = {}, den = 0;
    for (int s = 0; s < m.splits; ++s) {
        const float part_den = src[s * PART + D + 1];
        if (!(part_den > 0)) continue;
        const float w = expf(src[s * PART + D] - max_score);
        den += part_den * w;
#pragma unroll
        for (int j = 0; j < 16; ++j) value[j] += src[s * PART + lane * 16 + j] * w;
    }
    paged_store(out, gate, m, r, head, lane, value, den);
}

int paged_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 8) return R3D_NO(RAD_E_INVAL);
    RadTensor q = a->t[0], qr = a->t[1], o = a->t[7];
    const RadTensor& kv = a->t[2], &bt = a->t[3], &su = a->t[4], &cu = a->t[5];
    const RadTensor* gate = r3d_opt(a, 6);
    const long heads = r3d_geti_or(a, "heads", 0), bs = r3d_geti_or(a, "block_size", 0);
    const float scale = (float)r3d_getf(a, "scale", 0);
    auto heads_view = [&](RadTensor& t, long width) {
        if (heads > 0 && t.rank == 2 && t.shape[1] == heads * width) {
            t.rank = 3;
            t.shape[1] = heads;
            t.shape[2] = width;
            t.stride[2] = t.stride[1];
            t.stride[1] *= width;
        }
    };
    heads_view(q, D);
    heads_view(qr, RD);
    heads_view(o, D);
    if (q.dtype != RAD_BF16 || qr.dtype != RAD_BF16 || kv.dtype != RAD_BF16 ||
        o.dtype != RAD_BF16 || bt.dtype != RAD_I32 || su.dtype != RAD_I32 || cu.dtype != RAD_I32 ||
        (gate && gate->dtype != RAD_BF16))
        return R3D_NO(RAD_E_DTYPE);
    if (!q.data || !qr.data || !kv.data || !bt.data || !su.data || !cu.data || !o.data)
        return R3D_NO(RAD_E_INVAL);
    if (r3d_geti_or(a, "latent_dim", 0) != D || r3d_geti_or(a, "rope_dim", 0) != RD || heads < 1 ||
        heads > 64 || bs < 1 || bs > 256 || !std::isfinite(scale) || scale <= 0 || q.rank != 3 ||
        q.shape[0] < 1 || q.shape[0] > 65535 || q.shape[1] != heads || q.shape[2] != D ||
        q.stride[2] != 1 || q.stride[1] < D || q.stride[0] < heads * q.stride[1] || qr.rank != 3 ||
        qr.shape[0] != q.shape[0] || qr.shape[1] != heads || qr.shape[2] != RD ||
        qr.stride[2] != 1 || qr.stride[1] < RD || qr.stride[0] < heads * qr.stride[1] ||
        kv.rank != 4 || kv.shape[0] < 1 || kv.shape[0] > INT_MAX || kv.shape[1] != 1 ||
        kv.shape[2] != bs || kv.shape[3] < D + RD || kv.stride[3] != 1 || kv.stride[2] < D + RD ||
        kv.stride[0] < bs * kv.stride[2] || bt.rank != 2 || bt.shape[1] < 1 ||
        bt.shape[1] > INT_MAX / bs || bt.stride[1] < 1 || bt.stride[0] < bt.shape[1] * bt.stride[1] ||
        su.rank != 1 || su.shape[0] < 1 || bt.shape[0] < su.shape[0] || !r3d_contig(&su) ||
        !r3d_contig(&cu) || r3d_numel(&cu) != su.shape[0] + 1 || o.rank != 3 ||
        o.shape[0] != q.shape[0] || o.shape[1] != heads || o.shape[2] != D || o.stride[2] != 1 ||
        o.stride[1] < D || o.stride[0] < heads * o.stride[1] ||
        (gate && (gate->rank != 2 || gate->shape[0] != q.shape[0] || gate->shape[1] < heads ||
                  gate->stride[1] != 1)))
        return R3D_NO(RAD_E_SHAPE);
    /* The split comes from the head count alone (paged_splits). Only a batch whose partials the
     * scratch cannot hold takes fewer, as many as fit. */
    const long rows = q.shape[0], slots = a->scratch ? (long)(a->scratch_bytes / (PART * 4)) : 0;
    int splits = paged_splits(heads);
    if (rows * heads * splits > slots) splits = (int)std::max(1l, slots / (rows * heads));
    const PagedMeta m{ q.stride[0],  q.stride[1],     qr.stride[0],       qr.stride[1],
                       kv.stride[0], kv.stride[2],    bt.stride[0],       bt.stride[1],
                       o.stride[0],  o.stride[1],     gate ? (long)gate->stride[0] : 0L,
                       (int)q.shape[0], (int)heads,  (int)su.shape[0],   (int)bt.shape[1],
                       (int)bs,      (int)kv.shape[0], splits,           scale };
    const uint16_t* gp = gate ? (const uint16_t*)gate->data : nullptr;
    (void)hipGetLastError();
    R3D_KLAUNCH(attend_paged, dim3(m.rows, (m.heads + 3) / 4, m.splits), dim3(128),
                (hipStream_t)stream, (const uint16_t*)q.data, (const uint16_t*)qr.data,
                (const uint16_t*)kv.data, (const int*)bt.data, (const int*)su.data,
                (const int*)cu.data, gp, (uint16_t*)o.data, (float*)a->scratch, m);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return R3D_NO(RAD_E_DEVICE);
    if (splits > 1) {
        (void)hipGetLastError();
        R3D_KLAUNCH(merge_paged, dim3(m.rows, (m.heads + 3) / 4), dim3(128), (hipStream_t)stream,
                    (const float*)a->scratch, gp, (uint16_t*)o.data, m);
        e = hipGetLastError();
        if (e != hipSuccess) return R3D_NO(RAD_E_DEVICE);
    }
    return RAD_OK;
}

}  // namespace

extern "C" int r3d_rad_mla_attn_paged(const RadArgs* a, RadStream s) {
    return paged_body(a, s);
}

/* Sized at the band's top row count, at most kPagedRows (paged_splits). */
extern "C" int64_t r3d_rad_mla_attn_paged_scratch(const RadArgs* a) {
    const long heads = r3d_geti_or(a, "heads", 0), rows = r3d_geti_or(a, "M", kPagedRows);
    if (heads < 1 || heads > 64 || rows < 1) return R3D_NO(RAD_E_SHAPE);
    return paged_scratch_bytes(rows, heads);
}
