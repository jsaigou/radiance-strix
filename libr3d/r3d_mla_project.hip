/* r3d_mla_project.hip -- mla_project: the per-head absorbed-MLA projections.
 *
 * An absorbed MLA model keeps one 512-wide latent per token and folds the per-head K/V
 * projections into two matrices:
 *
 *   mode "absorb"   y[M, heads, 512] = x[M, heads, head_dim] @ w[head, 0, :, :]^T per head
 *   mode "expand"   y[M, heads, head_dim] = x[M, heads, 512] @ w[head, 1, :, :]^T per head
 *
 * where w is the shared [heads, 2 * head_dim, 512] BF16 matrix (the K half, then the V half, of
 * each head's rows) and head_dim is 128 or 256. f32 arithmetic, one bf16 rounding at the store,
 * as libref.
 *
 * Absorption streams adjacent latent columns: one thread a column, R rows a thread, and each
 * weight element is loaded once for the R rows it serves. Expansion reduces a contiguous weight
 * row: one wave a column, R rows a wave, each lane its share of the 512-wide sum in ascending j,
 * then one wave sum. Every row's sum runs over j in the same order whatever R is, so a row's
 * bytes do not depend on the batch it is in. A tail group past the last row reads that row again
 * and stores nothing for it.
 *
 * Bandwidth-bound GEMV shapes: plain vectorised f32 is the idiomatic RDNA3 path here, as in the
 * skinny GEMM rows. No WMMA (RDNA3's wave matrix is f16/i8 only and adds a fragment layout for no
 * throughput at these shapes) and no fp8.
 *
 * The launcher (r3d_rad_mla_project) checks: bf16 in, weights and out; heads 1..64; head_dim
 * 128 or 256; latent_dim 512; x [M, heads, head_dim|512], w [heads, 2*head_dim, 512], y
 * [M, heads, 512|head_dim], M 1..65535, last axes contiguous and row pitches at least the dense
 * ones (a 2-D [M, heads*width] plane is viewed as 3-D); RAD_E_INVAL, RAD_E_DTYPE, RAD_E_SHAPE.
 *
 * RDNA4 -> RDNA3: the kernels use only bf16 loads, f32 FMAs and the wave-sum butterfly
 * (r3d_xor_sum<32>, the same order as libr4d's, so bit-identical). The element loads are the
 * library's ldg_u16 (a whole register, no register-half wait); everything else is unchanged.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

struct ProjectMeta {
    long xr, xh, wp, yr, yh;
    int rows, heads;
};

/* Absorption streams adjacent latent columns; expansion reduces a contiguous weight row. R rows
 * a thread (absorb) or a wave (expand): each weight element is loaded once for R rows. Every row's
 * sum runs over j in the same order whatever R is (expand: each lane's share in ascending j, then
 * r3d_xor_sum<32>), so a row's bytes do not depend on the batch it is in. A tail group past the
 * last row reads that row again and stores nothing for it. */
template <int R, int HD>
__global__ __launch_bounds__(128) R3D_WAVES(4)
void project_absorb_rows(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                         uint16_t* __restrict__ y, ProjectMeta m) {
    const int col = (int)blockIdx.x * 128 + (int)threadIdx.x, head = (int)blockIdx.y,
              r0 = (int)blockIdx.z * R;
    if (col >= 512) return;
    const uint16_t* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
        xr[r] = x + (long)std::min(r0 + r, m.rows - 1) * m.xr + (long)head * m.xh;
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0;
    for (int j = 0; j < HD; ++j) {
        const float wv = bf16_to_f32((uint16_t)ldg_u16(w + ((long)head * 2 * HD + j) * m.wp + col));
#pragma unroll
        for (int r = 0; r < R; ++r) sum[r] = fmaf(bf16_to_f32((uint16_t)ldg_u16(xr[r] + j)), wv, sum[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (r0 + r < m.rows) y[(long)(r0 + r) * m.yr + (long)head * m.yh + col] = f32_to_bf16(sum[r]);
}

template <int R, int HD>
__global__ __launch_bounds__(128) R3D_WAVES(4)
void project_expand_rows(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                         uint16_t* __restrict__ y, ProjectMeta m) {
    const int lane = (int)threadIdx.x & 31, col = (int)blockIdx.x * 4 + ((int)threadIdx.x >> 5),
              head = (int)blockIdx.y, r0 = (int)blockIdx.z * R;
    const uint16_t* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
        xr[r] = x + (long)std::min(r0 + r, m.rows - 1) * m.xr + (long)head * m.xh;
    const uint16_t* const wr = w + ((long)head * 2 * HD + HD + col) * m.wp;
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0;
    for (int j = lane; j < 512; j += 32) {
        const float wv = bf16_to_f32((uint16_t)ldg_u16(wr + j));
#pragma unroll
        for (int r = 0; r < R; ++r) sum[r] = fmaf(bf16_to_f32((uint16_t)ldg_u16(xr[r] + j)), wv, sum[r]);
    }
    float mine = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        sum[r] = r3d_xor_sum<32>(sum[r]);
        if (lane == r) mine = sum[r];
    }
    if (lane < R && r0 + lane < m.rows)
        y[(long)(r0 + lane) * m.yr + (long)head * m.yh + col] = f32_to_bf16(mine);
}

int project_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    RadTensor x = a->t[0], y = a->t[2];
    const RadTensor& w = a->t[1];
    const char* mode = r3d_gets(a, "mode");
    const bool absorb = mode && !std::strcmp(mode, "absorb"),
               expand = mode && !std::strcmp(mode, "expand");
    const long h = r3d_geti_or(a, "heads", 0), hd = r3d_geti_or(a, "head_dim", 0);
    auto heads_view = [&](RadTensor& t, long width) {
        if (h > 0 && t.rank == 2 && t.shape[1] == h * width) {
            t.rank = 3;
            t.shape[1] = h;
            t.shape[2] = width;
            t.stride[2] = t.stride[1];
            t.stride[1] *= width;
        }
    };
    heads_view(x, absorb ? hd : 512);
    heads_view(y, absorb ? 512 : hd);
    if (x.dtype != RAD_BF16 || w.dtype != RAD_BF16 || y.dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);
    if (!x.data || !w.data || !y.data) return R3D_NO(RAD_E_INVAL);
    if ((!absorb && !expand) || h < 1 || h > 64 || (hd != 128 && hd != 256) ||
        r3d_geti_or(a, "latent_dim", 0) != 512 || x.rank != 3 || w.rank != 2 || y.rank != 3 ||
        x.shape[0] < 1 || x.shape[0] > 65535 || x.shape[1] != h ||
        x.shape[2] != (absorb ? hd : 512) || w.shape[0] != h * 2 * hd || w.shape[1] != 512 ||
        y.shape[0] != x.shape[0] || y.shape[1] != h || y.shape[2] != (absorb ? 512 : hd) ||
        x.stride[2] != 1 || w.stride[1] != 1 || y.stride[2] != 1 || x.stride[1] < x.shape[2] ||
        x.stride[0] < h * x.stride[1] || w.stride[0] < 512 || y.stride[1] < y.shape[2] ||
        y.stride[0] < h * y.stride[1])
        return R3D_NO(RAD_E_SHAPE);
    const ProjectMeta m{ x.stride[0], x.stride[1], w.stride[0], y.stride[0],
                         y.stride[1], (int)x.shape[0], (int)h };
    const uint16_t* xp = (const uint16_t*)x.data;
    const uint16_t* wp = (const uint16_t*)w.data;
    uint16_t* yp = (uint16_t*)y.data;
    /* The largest of 8, 4, 2 and 1 rows a group that the batch fills. */
    const int R = m.rows >= 8 ? 8 : m.rows >= 4 ? 4 : m.rows >= 2 ? 2 : 1;
#define MLA_PROJECT_HD(RR, HD)                                                                     \
    do {                                                                                           \
        const unsigned groups = (unsigned)((m.rows + RR - 1) / RR);                                \
        (void)hipGetLastError();                                                                   \
        if (absorb)                                                                                \
            R3D_KLAUNCH((project_absorb_rows<RR, HD>), dim3(4, h, groups), dim3(128),              \
                        (hipStream_t)stream, xp, wp, yp, m);                                       \
        else                                                                                       \
            R3D_KLAUNCH((project_expand_rows<RR, HD>), dim3(HD / 4, h, groups), dim3(128),         \
                        (hipStream_t)stream, xp, wp, yp, m);                                       \
        const hipError_t e = hipGetLastError();                                                    \
        return e == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);                                    \
    } while (0)
#define MLA_PROJECT_ROWS(RR)                                                                       \
    do {                                                                                           \
        if (hd == 128) MLA_PROJECT_HD(RR, 128);                                                    \
        MLA_PROJECT_HD(RR, 256);                                                                   \
    } while (0)
    if (R == 8) MLA_PROJECT_ROWS(8);
    if (R == 4) MLA_PROJECT_ROWS(4);
    if (R == 2) MLA_PROJECT_ROWS(2);
    MLA_PROJECT_ROWS(1);
#undef MLA_PROJECT_ROWS
#undef MLA_PROJECT_HD
}

}  // namespace

extern "C" int r3d_rad_mla_project(const RadArgs* a, RadStream s) {
    return project_body(a, s);
}
