/* r3d_gemm_nt_f32.hip -- gemm_nt_f32: the router's and the index gate's F32 projection.

     x [M][K] bf16, w [N][K] f32  ->  y [M][N] f32,  y = x @ w^T

   Routing logits and index-head gates stay F32 projections even in a BF16 checkpoint (libref's
   ref_gemm_nt_f32), so nothing narrows: the accumulation is f32, the weight is read as f32 and the
   store writes f32. N is tiny -- a checkpoint's expert count, a handful of gate columns, at most
   512 -- against a hidden width K, so the op streams the weight plane and is bandwidth-bound, not
   compute-bound. There is no matrix tile here and RDNA3's wave matrix could not take it: the wave
   matrix is f16 and i8 only (r3d_common.h), an f32 weight has no fragment form. Plain f32 FMAs,
   one output element a lane-grid, is the arithmetic and the idiomatic libr3d shape for a GEMV-like
   projection.

   ---- the lane map --------------------------------------------------------------------------

   One wave computes R rows by C columns, so each loaded weight serves R rows and each input C
   columns. Every output has the same arithmetic: lane l folds k = l, l+32, ... ascending with fmaf,
   then the xor butterfly over 16, 8, 4, 2, 1. libr3d runs that butterfly with wave_sum (DPP16 and
   v_permlanex16, r3d_common.h), which is the same exchange in the same order as the __shfl_xor
   loop libr4d uses, so a wave's sum is bit-for-bit the sum libr4d's wave forms and the row's host
   arithmetic in r4d_selftest --perff32 write out. Every form below is byte-identical to every
   other: the wave count, the tile and the steps in flight change how the bytes are fetched, never
   how they add. Tail rows and columns read a clamped valid address and are never stored.

   A lane loads U steps of k before their FMAs. At a decode pass's few rows a wave streams its
   weight row with no other work to hide the load behind, and one step in flight leaves the router
   at a fraction of the card's bandwidth; the deep forms buy that latency back with registers.

   ---- what the launcher checks ---------------------------------------------------------------

   The three operands present, x bf16, w f32 and y f32 (RAD_E_DTYPE), 1 <= N <= 512 and
   1 <= K <= 16384, x, w and y rank 2 with x [M][K], w [N][K] and y [M][N] for x's own M
   (RAD_E_SHAPE), every row k-contiguous with a pitch of at least the row's width (the same
   RAD_E_SHAPE, as in libr4d), and M within the row-count grid, 1..65535. Nothing is staged in
   LDS, so RDNA3's 64 KiB a workgroup is not in play; the check below is written against the cap
   so a staged form added later cannot skip it.

   ---- RDNA4 -> RDNA3 -------------------------------------------------------------------------

   The tiling (R rows, C columns, U steps of k in flight), the four forms and the rule that picks
   one are libr4d's (r4d_glm.hip, gemm_nt_f32_tile_kernel), transcribed: the arithmetic is
   identical, so the choice between forms is speed only. The block stays four waves of 32 and the
   grid x * 4 * C columns by y * R rows. The reduction is wave_sum, not __shfl_xor: on gfx11 a
   shuffle is a ds_bpermute plus an LDS wait, and five of them per accumulator dominates a wave's
   tail. R3D_WAVES(8) states the occupancy against gfx11's 1024 VGPRs a SIMD at wave32: the
   U = 16 form carries 2 * 16 * 2 = 64 registers of loads in flight plus R * C accumulators, inside
   the 128 eight waves a SIMD allows. The form rule's thresholds are gfx1201's measurements as
   libr4d quotes them; they are kept as the starting point, and a tune on a gfx11 part is free to
   move them, because every form computes the same bytes.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <cmath>

namespace {

constexpr uint32_t kThreads = 128;              // four waves a block, as in libr4d
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kWavesPerSimd = 8;           // 1024 VGPRs a SIMD / 8 = 128 a wave

/* The geometry the shim admits, from the row's constraint table. */
constexpr long long kNMax = 512, kKMax = 16384, kMMax = 65535;

/* Dynamic LDS a launch asks for. Every form keeps its tile in registers, so the launch's shared
 * byte count is 0 and the cap below is the exemplar's, for the launcher to enforce. */
constexpr size_t kStageBytes = 0;

template <int R, int C, int U>
__global__ void __launch_bounds__(kThreads) R3D_WAVES(kWavesPerSimd)
    gemm_nt_f32_tile_kernel(const uint16_t* x, const float* w, float* y, long xp, long wp, long yp,
                            int m, int n, int k) {
    const int lane = (int)(threadIdx.x & (kWave - 1)), wave = (int)(threadIdx.x / kWave);
    const int c0 = (int)(blockIdx.x * kWaves + (unsigned)wave) * C, r0 = (int)blockIdx.y * R;
    if (c0 >= n) return;
    const uint16_t* xr[R];
    const float* wc[C];
#pragma unroll
    for (int q = 0; q < R; ++q) xr[q] = x + (long)(r0 + q < m ? r0 + q : m - 1) * xp;
#pragma unroll
    for (int c = 0; c < C; ++c) wc[c] = w + (long)(c0 + c < n ? c0 + c : n - 1) * wp;
    float sum[R][C];
#pragma unroll
    for (int q = 0; q < R; ++q)
#pragma unroll
        for (int c = 0; c < C; ++c) sum[q][c] = 0;
    int j = lane;
    for (; j + 32 * (U - 1) < k; j += 32 * U) {
        float weight[U][C];
        uint32_t in[U][R];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int c = 0; c < C; ++c) weight[u][c] = ldg_f32(wc[c] + j + 32 * u);
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < R; ++q) in[u][q] = ldg_u16(xr[q] + j + 32 * u);
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < R; ++q) {
                const float v = bf16_to_f32((uint16_t)in[u][q]);
#pragma unroll
                for (int c = 0; c < C; ++c) sum[q][c] = fmaf(v, weight[u][c], sum[q][c]);
            }
    }
    for (; j < k; j += 32) {
#pragma unroll
        for (int q = 0; q < R; ++q) {
            const float v = bf16_to_f32((uint16_t)ldg_u16(xr[q] + j));
#pragma unroll
            for (int c = 0; c < C; ++c) sum[q][c] = fmaf(v, ldg_f32(wc[c] + j), sum[q][c]);
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q)
#pragma unroll
        for (int c = 0; c < C; ++c) sum[q][c] = wave_sum(sum[q][c]);
    if (lane) return;
#pragma unroll
    for (int q = 0; q < R; ++q)
        if (r0 + q < m)
#pragma unroll
            for (int c = 0; c < C; ++c)
                if (c0 + c < n) y[(long)(r0 + q) * yp + c0 + c] = sum[q][c];
}

/* One tile form at the grid it owns: x * 4 * C columns, y * R rows, no LDS. */
template <int R, int C, int U = 1>
int gemm_f32_tile(RadStream s, const void* x, const void* w, void* y, long xp, long wp, long yp,
                  long m, long n, long k) {
    const dim3 grid((unsigned)((n + (long)kWaves * C - 1) / ((long)kWaves * C)),
                    (unsigned)((m + R - 1) / R)), blk(kThreads);
    R3D_KLAUNCH(R3D_KNAME((gemm_nt_f32_tile_kernel<R, C, U>)), grid, blk, (hipStream_t)s,
                (const uint16_t*)x, (const float*)w, (float*)y, xp, wp, yp, (int)m, (int)n, (int)k);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* The tile form, rows x columns a wave and steps of k in flight, by M; all forms are
 * byte-identical, so the choice is speed only. Measured on gfx1201 (r4d_selftest --perff32, cold)
 * at N 4/32/36/288, K 4096: 16x4 gains 3-5 % at N 288/32 above 64 rows but loses 16 % at N 36;
 * 8x4 holds at every width. Below 64 rows the steps of k in flight decide: 2x2 with 16 steps is 2
 * to 2.5 times as fast as one step at N 288 up to 16 rows, and about 3 times at N 32 at every
 * count below 64; at N 288 from 17 rows 8x1 with 8 steps is faster. A projection of a few columns
 * has too few waves for wide tiles at any M. The thresholds are libr4d's, kept as the starting
 * point for gfx11: the forms compute the same bytes, so a tune here moves them, not the kernel. */
int gemm_f32_body(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor *xt = &a->t[0], *wt = &a->t[1], *yt = &a->t[2];
    const long long n = r3d_geti_or(a, "N", 0), k = r3d_geti_or(a, "K", 0);
    if (xt->dtype != RAD_BF16 || wt->dtype != RAD_F32 || yt->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);
    if (!r3d_p(xt) || !r3d_p(wt) || !r3d_wp(yt)) return R3D_NO(RAD_E_INVAL);
    if (n < 1 || n > kNMax || k < 1 || k > kKMax || xt->rank != 2 || wt->rank != 2 ||
        yt->rank != 2 || xt->shape[0] < 1 || xt->shape[0] > kMMax || xt->shape[1] != k ||
        wt->shape[0] != n || wt->shape[1] != k || yt->shape[0] != xt->shape[0] ||
        yt->shape[1] != n || xt->stride[1] != 1 || wt->stride[1] != 1 || yt->stride[1] != 1 ||
        xt->stride[0] < k || wt->stride[0] < k || yt->stride[0] < n)
        return R3D_NO(RAD_E_SHAPE);
    /* Nothing is staged: a wave's whole tile is its register grid, so RDNA3's 64 KiB a workgroup
     * is not in play. The cap is asserted where the exemplar's launcher checks it, so a staged
     * form added to this unit cannot skip it. */
    static_assert(kStageBytes <= 60 * 1024, "a workgroup's stage must fit RDNA3's 64 KiB of LDS");

    const long m = (long)xt->shape[0];                     // the step's rows, not the band's bound
    const void* px = r3d_p(xt);
    const void* pw = r3d_p(wt);
    void* py = r3d_wp(yt);
    const long xp = (long)xt->stride[0], wp = (long)wt->stride[0], yp = (long)yt->stride[0];
    if (n <= 8) return gemm_f32_tile<4, 1>(s, px, pw, py, xp, wp, yp, m, n, k);
    if (m <= 16 || (m < 64 && n < 128)) return gemm_f32_tile<2, 2, 16>(s, px, pw, py, xp, wp, yp, m, n, k);
    if (m < 64) return gemm_f32_tile<8, 1, 8>(s, px, pw, py, xp, wp, yp, m, n, k);
    return gemm_f32_tile<8, 4>(s, px, pw, py, xp, wp, yp, m, n, k);
}

}  // namespace

/* ============================================================== the plugin ABI entry point
 *
 * gemm_nt_f32's operand list is [a, b, y] and its parameters M, N, K, dtype (libref's pGemm /
 * oGemm, libr4d's pGemmNt / oGemmNt). The row count comes from the operand: the band's `M` is an
 * upper bound, and `N` and `K` are read as the shim's extents, exactly as libr4d's shim reads them. */
extern "C" int r3d_rad_gemm_nt_f32(const RadArgs* a, RadStream s)
{
    return gemm_f32_body(a, s);
}
