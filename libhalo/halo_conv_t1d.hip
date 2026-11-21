/* Derived from libr4d/r4d_audio.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_conv_t1d.hip -- a vocoder's transposed convolution, gfx1151: the bf16 conv_t1d row, which
// libr4d keeps inside r4d_audio.hip beside conv1d (libhalo's halo_audio.hip holds the rest of that
// unit; this one is extracted so the two lanes' slices stay apart).
//
// The implicit GEMM is libr4d's, with the two gfx11 fragment changes halo_gdn_wmma.h states:
// the fragment pointer loses `4 * (lane >> 4)` (a lane holds the whole 16-wide k run, 32 bytes) and
// the accumulator's element e is D[2e + (lane>>4)][lane & 15], so the blocked store goes through
// accPack -- four v_permlanex16 -- to hand each lane its own eight consecutive channels back.
// Arithmetic is untouched: f32 accumulation, one WMMA per 16 K ascending, bias after the product,
// RTNE bf16 stores (libref/ref_audio.cpp's conv_t1d).
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

// libr4d's r4d_gdn_wmma.h carried this name; libhalo's halo_gdn_wmma.h is the RDNA3.5 fragment
// layer and does not, so the units that need it define it locally (halo_audio.hip does the same).
__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

constexpr int kC1T = 256;

__device__ __forceinline__ void st8(uint16_t* p, const float* f) {
    uint4 u;
    u.x = (uint32_t)f32_to_bf16(f[0]) | ((uint32_t)f32_to_bf16(f[1]) << 16);
    u.y = (uint32_t)f32_to_bf16(f[2]) | ((uint32_t)f32_to_bf16(f[3]) << 16);
    u.z = (uint32_t)f32_to_bf16(f[4]) | ((uint32_t)f32_to_bf16(f[5]) << 16);
    u.w = (uint32_t)f32_to_bf16(f[6]) | ((uint32_t)f32_to_bf16(f[7]) << 16);
    *(uint4*)p = u;
}

// The sequence that owns row r: linear, because a vocoder pass has two (a stereo pair).
__device__ __forceinline__ int seq_of(const int* cu, int n_seq, int r) {
    int s = 0;
    while (s + 1 < n_seq && cu[s + 1] <= r) ++s;
    return s;
}

#define C1_BK 64
#define C1_PITCH (C1_BK + 8)

// ---------------------------------------------------------------------------- conv_t1d
// The transposed convolution of kernel `taps` s (two taps or one), stride s, padding P
// ((k - s) / 2, or 0 for the causal form) as s convolutions of `taps` taps, one a grid slice
// (blockIdx.z is the phase r): output row lo = q s + r - P of a sequence of L input rows reads
// x[q - t] for each tap t (PyTorch's tap r + t s). For a phase below P the q that land inside the
// sequence are 1..L, otherwise 0..L-1, so each phase has L output rows, one per input row: the M
// dimension is the input rows, as for conv1d. The weight is relayouted to [s][cout][taps][cin],
// which does not depend on P.
template <int TP, int TC>
__global__ __launch_bounds__(256) void conv_t1d_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ bias, const int* __restrict__ cu, uint16_t* __restrict__ y,
    int n_seq, int r_beg, int r_end, int cin, int cout, int s, int P, int taps, int x_ld,
    int y_ld) {
    constexpr int BP = 2 * TP * 16, BC = 4 * TC * 16;
    constexpr int CPR = C1_BK / 8;
    constexpr int AL = BP * CPR / 256, WL = BC * CPR / 256;
    static_assert(AL >= 1 && WL >= 1, "every thread stages whole chunks");
    __shared__ __attribute__((aligned(16))) uint16_t sA[BP * C1_PITCH];
    __shared__ __attribute__((aligned(16))) uint16_t sW[BC * C1_PITCH];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wp = wave / 4, wc = wave % 4;
    const int p0 = r_beg + blockIdx.x * BP, c0 = blockIdx.y * BC;
    const int ph = blockIdx.z, qoff = ph < P ? 1 : 0;
    const int K = taps * cin;
    const uint16_t* wph = w + (size_t)ph * cout * K;

    int a_q[AL], a_t[AL], a_ci[AL], a_s0[AL], a_len[AL];
    bool a_ok[AL];
    uint16_t* adst[AL];
    #pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int m = p0 + r;
        a_ok[i] = m < r_end;
        const int mm = a_ok[i] ? m : r_beg;
        const int sq = seq_of(cu, n_seq, mm);
        a_s0[i] = cu[sq];
        a_len[i] = cu[sq + 1] - cu[sq];
        a_q[i] = mm - cu[sq] + qoff;
        a_t[i] = e / cin;
        a_ci[i] = e % cin;
        adst[i] = sA + r * C1_PITCH + e;
    }
    size_t w_off[WL];
    int w_e[WL];
    bool w_ok[WL];
    uint16_t* wdst[WL];
    #pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int co = c0 + r;
        w_ok[i] = co < cout;
        w_off[i] = (size_t)(w_ok[i] ? co : 0) * K + e;
        w_e[i] = e;
        wdst[i] = sW + r * C1_PITCH + e;
    }
    const halo_u32x4 zero = {0u, 0u, 0u, 0u};
    auto loadA = [&](int i) -> halo_u32x4 {
        if (!a_ok[i] || a_t[i] >= taps) return zero;
        const int qi = a_q[i] - a_t[i];
        if (qi < 0 || qi >= a_len[i]) return zero;
        return *(const halo_u32x4*)(x + (size_t)(a_s0[i] + qi) * x_ld + a_ci[i]);
    };
    auto loadW = [&](int i, int kt) -> halo_u32x4 {
        if (!w_ok[i] || kt * C1_BK + w_e[i] >= K) return zero;
        return *(const halo_u32x4*)(wph + w_off[i] + (size_t)kt * C1_BK);
    };

    /* This lane's fragment row of each tile, at the un-split k offset (halo_gdn_wmma.h's rowP). */
    const uint16_t* fa = sA + (wp * TP * 16 + (lane & 15)) * C1_PITCH;
    const uint16_t* fw = sW + (wc * TC * 16 + (lane & 15)) * C1_PITCH;

    v8f acc[TC][TP];
    #pragma unroll
    for (int j = 0; j < TC; ++j)
        #pragma unroll
        for (int i = 0; i < TP; ++i) acc[j][i] = (v8f)(0.f);

    halo_u32x4 ra[AL], rw[WL];
    #pragma unroll
    for (int i = 0; i < AL; ++i) ra[i] = loadA(i);
    #pragma unroll
    for (int i = 0; i < WL; ++i) rw[i] = loadW(i, 0);

    const int nk = (K + C1_BK - 1) / C1_BK;
    for (int kt = 0; kt < nk; ++kt) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < AL; ++i) *(halo_u32x4*)adst[i] = ra[i];
        #pragma unroll
        for (int i = 0; i < WL; ++i) *(halo_u32x4*)wdst[i] = rw[i];
        __syncthreads();
        if (kt + 1 < nk) {
            #pragma unroll
            for (int i = 0; i < AL; ++i) {
                a_ci[i] += C1_BK;
                while (a_ci[i] >= cin) { a_ci[i] -= cin; ++a_t[i]; }
                ra[i] = loadA(i);
            }
            #pragma unroll
            for (int i = 0; i < WL; ++i) rw[i] = loadW(i, kt + 1);
        }
        #pragma unroll
        for (int ss = 0; ss < C1_BK / 16; ++ss) {
            v16bf af[TP], wf[TC];
            #pragma unroll
            for (int i = 0; i < TP; ++i) af[i] = fragP(fa + i * 16 * C1_PITCH + ss * 16);
            #pragma unroll
            for (int j = 0; j < TC; ++j) wf[j] = fragP(fw + j * 16 * C1_PITCH + ss * 16);
            #pragma unroll
            for (int j = 0; j < TC; ++j)
                #pragma unroll
                for (int i = 0; i < TP; ++i) acc[j][i] = mma(wf[j], af[i], acc[j][i]);
        }
    }

    /* Lane l's element e is channel cb + accRow(l, e) of input row accCol(l), so the bias is added
     * per element and accPack then reassembles the tile's sixteen consecutive channels. */
    #pragma unroll
    for (int j = 0; j < TC; ++j)
        #pragma unroll
        for (int i = 0; i < TP; ++i) {
            const int m = p0 + (wp * TP + i) * 16 + accCol(lane);
            if (m >= r_end) continue;
            const int cb = c0 + (wc * TC + j) * 16;
            const int sq = seq_of(cu, n_seq, m);
            const int q = m - cu[sq] + qoff;
            const int orow = cu[sq] * s + q * s + ph - P;
            uint16_t* dst = y + (size_t)orow * y_ld + cb;
            if (cb + 16 <= cout && (cout & 7) == 0) {
                float f[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) f[e] = acc[j][i][e] + bf2f(bias[cb + accRow(lane, e)]);
                *(uint4*)(dst + accPackRow(lane)) = accPack(f, lane);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int co = cb + accRow(lane, e);
                    if (co < cout) dst[co] = f32_to_bf16(acc[j][i][e] + bf2f(bias[co]));
                }
            }
        }
}

inline int c1_st_of(hipError_t e) { return e == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE); }
inline bool c1_is_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }
inline bool c1_is_i32(const RadTensor* t) { return t && t->dtype == RAD_I32; }

typedef __bf16 c1_bf162 __attribute__((ext_vector_type(2)));

// ---------------------------------------------------------------------------- conv_t1d small-row
// The decoder's transposed convolution is a few input rows against a large weight; the tiled kernel
// above stages the weight to LDS through 128 rows of x. This path is the m16 matrix-vector product
// per phase (blockIdx.z): a wave an output channel, K = taps * cin split SK ways and reduced in LDS
// in split order, the weight streamed coalesced once. Bias only -- the transposed conv has no
// residual or activation. MT is M rounded up to 1, 2, 4, 8 or 16; rows past M are computed on row
// M-1 and dropped.
template <int MT>
__global__ __launch_bounds__(1024) void conv_t1d_small_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ bias, const int* __restrict__ cu, uint16_t* __restrict__ y,
    int n_seq, int rx, int cin, int cout, int s, int P, int taps, int x_ld, int y_ld, int M,
    int K, int WV, int SK) {
    extern __shared__ float red[];
    constexpr int WAVE = 32, VEC = 8;
    const int tid = threadIdx.x, wave = tid / WAVE, lane = tid % WAVE;
    const int col_local = wave / SK, ksplit = wave % SK;
    const int j = blockIdx.x * WV + col_local;
    const int ph = blockIdx.z, qoff = ph < P ? 1 : 0;
    const int Ks = K / SK, kbase = ksplit * Ks;
    const uint16_t* wph = w + (size_t)ph * cout * K;

    int s0[MT], len[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int r = i < M ? i : M - 1;
        const int sq = seq_of(cu, n_seq, r);
        s0[i] = cu[sq];
        len[i] = cu[sq + 1] - cu[sq];
    }

    float acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) acc[i] = 0.f;

    if (j < cout) {
        const uint16_t* Wj = wph + (size_t)j * K + kbase + lane * VEC;
        const int niter = Ks / (WAVE * VEC);
        const int rem = Ks - niter * (WAVE * VEC);
        const bool has_tail = rem > 0 && lane < rem / VEC;
        halo_u32x4 wbuf[8];
#pragma unroll
        for (int t = 0; t < 8; ++t)
            if (t < niter) wbuf[t] = *(const halo_u32x4*)(Wj + t * WAVE * VEC);
        halo_u32x4 wtail;
        if (has_tail) wtail = *(const halo_u32x4*)(Wj + niter * WAVE * VEC);

        const halo_u32x4 zero = { 0u, 0u, 0u, 0u };
        for (int t = 0; t < niter; ++t) {
            const c1_bf162* wp = (const c1_bf162*)&wbuf[t];
            const int kk = kbase + lane * VEC + t * WAVE * VEC;
            const int tap = kk / cin, ci = kk % cin;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = i < M ? i : M - 1;
                const int qi = (r - s0[i]) + qoff - tap;
                const bool live = qi >= 0 && qi < len[i];
                const int irow = s0[i] + qi;
                const int irc = irow < 0 ? 0 : (irow >= rx ? rx - 1 : irow);
                const halo_u32x4 av =
                    live ? *(const halo_u32x4*)(x + (size_t)irc * x_ld + ci) : zero;
                const c1_bf162* ap = (const c1_bf162*)&av;
#pragma unroll
                for (int p = 0; p < VEC / 2; ++p)
                    acc[i] = __builtin_amdgcn_fdot2_f32_bf16(ap[p], wp[p], acc[i], false);
            }
        }
        if (has_tail) {
            const c1_bf162* wp = (const c1_bf162*)&wtail;
            const int kk = kbase + lane * VEC + niter * WAVE * VEC;
            const int tap = kk / cin, ci = kk % cin;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = i < M ? i : M - 1;
                const int qi = (r - s0[i]) + qoff - tap;
                const bool live = qi >= 0 && qi < len[i];
                const int irow = s0[i] + qi;
                const int irc = irow < 0 ? 0 : (irow >= rx ? rx - 1 : irow);
                const halo_u32x4 av =
                    live ? *(const halo_u32x4*)(x + (size_t)irc * x_ld + ci) : zero;
                const c1_bf162* ap = (const c1_bf162*)&av;
#pragma unroll
                for (int p = 0; p < VEC / 2; ++p)
                    acc[i] = __builtin_amdgcn_fdot2_f32_bf16(ap[p], wp[p], acc[i], false);
            }
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            float v = acc[i];
            for (int off = WAVE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0) red[(col_local * SK + ksplit) * MT + i] = v;
        }
    }
    __syncthreads();
    if (ksplit == 0 && j < cout && lane < M) {
        float v = 0.f;
        for (int s = 0; s < SK; ++s) v += red[(col_local * SK + s) * MT + lane];
        v += bf2f(bias[j]);
        const int sq = seq_of(cu, n_seq, lane);
        const int q = lane - cu[sq] + qoff;
        const int orow = cu[sq] * s + q * s + ph - P;
        y[(size_t)orow * y_ld + j] = f32_to_bf16(v);
    }
}

inline int conv_t1d_small_mt(int M) {
    if (M <= 1) return 1;
    if (M <= 2) return 2;
    if (M <= 4) return 4;
    if (M <= 8) return 8;
    return 16;
}
inline bool conv_t1d_small_legal(int K, int WV, int SK) {
    return SK >= 1 && WV >= 1 && WV * SK * 32 <= 1024 && K % SK == 0 && ((K / SK) % 8) == 0 &&
           (K / SK) / (32 * 8) <= 8;
}

}  // namespace

// ============================================================== the plugin ABI entry point
extern "C" int halo_rad_conv_t1d(const RadArgs* a, RadStream s) {
    const RadTensor *x = halo_opt(a, 0), *w = halo_opt(a, 1), *b = halo_opt(a, 2),
                    *cu = halo_opt(a, 3);
    const RadTensor *y = halo_opt(a, 4);
    if (!x || !w || !b || !cu || !y) return HALO_NO(RAD_E_INVAL);
    if (!c1_is_bf16(x) || !c1_is_bf16(w) || !c1_is_bf16(b) || !c1_is_bf16(y) || !c1_is_i32(cu))
        return HALO_NO(RAD_E_DTYPE);
    const long long cin = halo_geti_or(a, "cin", 0), cout = halo_geti_or(a, "cout", 0);
    const long long st = halo_geti_or(a, "stride", 0);
    /* Kernel 2 stride (two taps) or stride (one); without `pad`, the symmetric (k - stride) / 2,
     * which needs k - stride even. */
    const long long kw = halo_geti_or(a, "k", 2 * st), pad = halo_geti_or(a, "pad", -1);
    const long long P = pad >= 0 ? pad : (kw - st) / 2;
    if (cin <= 0 || cout <= 0 || st <= 0 || (kw != st && kw != 2 * st) ||
        (pad < 0 && ((kw - st) & 1)) || P >= st || (cin % 8))
        return HALO_NO(RAD_E_SHAPE);
    const int taps = (int)(kw / st);
    long long rx = 0, ry = 0;
    const long long xl = halo_rows_pitch(x, cin, &rx), yl = halo_rows_pitch(y, cout, &ry);
    if (xl < 0 || yl < 0 || (xl % 8) || ((uintptr_t)x->data & 15) || ((uintptr_t)w->data & 15))
        return HALO_NO(RAD_E_STRIDE);
    const long long n_seq = halo_numel(cu) - 1;
    if (n_seq <= 0 || rx <= 0 || ry < rx * st || halo_numel(w) < cin * cout * kw ||
        halo_numel(b) < cout)
        return HALO_NO(RAD_E_SHAPE);
    /* Small-row path: the decoder's few input rows against a large weight, streamed
     * coalesced once (the m16 pattern); the tiled kernel below stages the weight through
     * 128 rows of x. */
    const long long Kll = (long long)taps * cin;
    if (Kll <= (1ll << 30) && rx >= 1 && rx <= 16) {
        const int K = (int)Kll;
        const int wv = 1, mt = conv_t1d_small_mt((int)rx);
        int sk = 0;
        static const int ladder[] = { 1, 2, 4, 5, 8, 10, 16, 20, 32 };
        for (int cand : ladder)
            if (conv_t1d_small_legal(K, wv, cand)) { sk = cand; if ((long long)cout * cand >= 1280) break; }
        if (sk) {
            const dim3 grid((unsigned)((cout + wv - 1) / wv), 1, (unsigned)st),
                       block((unsigned)(wv * sk * 32));
            const size_t smem = (size_t)wv * sk * mt * sizeof(float);
#define C1TS(MT)                                                                                  \
            hipLaunchKernelGGL((conv_t1d_small_kernel<MT>), grid, block, smem, (hipStream_t)s,     \
                               (const uint16_t*)x->data, (const uint16_t*)w->data,                \
                               (const uint16_t*)b->data, (const int*)cu->data, (uint16_t*)y->data,\
                               (int)n_seq, (int)rx, (int)cin, (int)cout, (int)st, (int)P, taps,   \
                               (int)xl, (int)yl, (int)rx, K, wv, sk)
            switch (mt) {
                case 1: C1TS(1); break;
                case 2: C1TS(2); break;
                case 4: C1TS(4); break;
                case 8: C1TS(8); break;
                default: C1TS(16); break;
            }
#undef C1TS
            return c1_st_of(hipGetLastError());
        }
    }
    if (cout >= 128) {
        constexpr int TP = 4, TC = 2, BP = 128, BC = 128;
        hipLaunchKernelGGL((conv_t1d_kernel<TP, TC>),
                           dim3((unsigned)((rx + BP - 1) / BP), (unsigned)((cout + BC - 1) / BC),
                                (unsigned)st),
                           dim3(256), 0, (hipStream_t)s, (const uint16_t*)x->data,
                           (const uint16_t*)w->data, (const uint16_t*)b->data, (const int*)cu->data,
                           (uint16_t*)y->data, (int)n_seq, 0, (int)rx, (int)cin, (int)cout, (int)st,
                           (int)P, taps, (int)xl, (int)yl);
    } else {
        constexpr int TP = 4, TC = 1, BP = 128, BC = 64;
        hipLaunchKernelGGL((conv_t1d_kernel<TP, TC>),
                           dim3((unsigned)((rx + BP - 1) / BP), (unsigned)((cout + BC - 1) / BC),
                                (unsigned)st),
                           dim3(256), 0, (hipStream_t)s, (const uint16_t*)x->data,
                           (const uint16_t*)w->data, (const uint16_t*)b->data, (const int*)cu->data,
                           (uint16_t*)y->data, (int)n_seq, 0, (int)rx, (int)cin, (int)cout, (int)st,
                           (int)P, taps, (int)xl, (int)yl);
    }
    return c1_st_of(hipGetLastError());
}
