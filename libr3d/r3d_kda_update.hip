/* r3d_kda_update.hip -- kda_update: GLM's per-channel KDA recurrence over a paged f32 state cache,
 * including the speculative candidate snapshots.
 *
 * The convolution's outputs come straight in (q, k, v, a, b); the gates are decoded here and the
 * delta-rule step runs per token against one [128, 128] f32 state per head:
 *
 *   decay = exp(lower_bound / (1 + e^-(A_log[head] * (a + dt_bias))))     beta = sigmoid(b)
 *   q, k  l2-normalised (q's root scaled by `scale`, both by `l2_eps`)
 *   S *= decay        u = beta * (v - S k)        S += u k^T        o = S q
 *
 * `mode` "prefill" runs the same recurrence over a sequence's tokens and commits only its final
 * state; "verify" writes a state slot per candidate token, since which candidate survives is not
 * known while this runs. `state_form` "anchor" is the rollback pair: the state is read at the
 * anchor's slot, every candidate's factors are recorded into a scratch slot, and a second launch
 * replays the accepted prefix of that record into the committed slot. The replay is a separate
 * dispatch because every old factor read must finish before the next window overwrites them.
 *
 *   q, k, a, o  [tokens, heads, 128] bf16, accepted as [tokens, heads*128] too
 *   b           [tokens, heads] bf16 (width 1, the same two forms)
 *   A_log       [heads] f32,  dt_bias [heads*128] f32
 *   state       [slots, heads, 128, 128] f32 in/out, at the cache's own strides (a mamba page is
 *               padded to the attention page size, so a shape-derived stride reads every slot but
 *               the first at the wrong offset)
 *   state_idx   [seqs, width] i32, width >= 2 for `anchor` (column width/2 is the scratch slot)
 *   cu          [seqs+1] i32, absolute token offsets
 *   num_accepted, has_init  [seqs] i32, optional
 *
 * The launcher checks what libr4d's checks: 13 operands, heads 1..65535 with head_k = head_v =
 * 128, the five bf16 inputs in one of the two headed forms at the token count, the state a rank-4
 * f32 cache of at least one slot whose strides cover it, cu rank-1 i32 of 2..65536 entries,
 * state_idx rank-2 i32 with a row per sequence, A_log and dt_bias rank-1 f32 of at least heads and
 * heads*128, the two optional i32 vectors one entry per sequence, mode in {prefill, verify} and
 * state_form in {per_candidate, anchor} (anything else RAD_E_INVAL), anchor with a two-column
 * state_idx, lower_bound finite and negative, l2_eps finite and positive, scale finite.
 *
 * RDNA4 -> RDNA3 (from libr4d/r4d_kda.hip, geometry unchanged). gfx1102 has the same wave32 and
 * the same 64 KiB LDS as gfx1201, and this kernel tiles nothing in LDS: one wave owns four value
 * rows and keeps each row's 128 state elements in four registers a lane, so a workgroup stages
 * nothing and the 60 KiB bound the launcher states is vacuous by construction (a state tile moved
 * into LDS is 64 KiB, one head's whole state, and would not fit a workgroup's budget). The
 * reductions are libr3d's wave_sum, the same XOR butterflies in the same order as r4d_wave_sum, so
 * the l2 sums, the predictions and the outputs are bit-identical; a bf16 operand widens through
 * bf16_to_f32, the same shift into the high half as libr4d's bf16w_lo. Nothing here is fp8: the
 * state, the recorded factors and the decay are f32, and the recurrence is bandwidth-bound, so the
 * plain vectorised f32 ALU is the correct tool -- no WMMA (gfx11's is f16/i8 only) and no fp8
 * matrix instruction, which gfx11 lacks. libr4d's typed R4D_EMIT dispatch becomes R3D_KLAUNCH plus
 * hipGetLastError, returning RAD_E_DEVICE for a launch the driver refused, as R4D_EMIT does.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include <climits>
#include <cmath>
#include <cstring>

namespace {

/* Four waves a workgroup, as in libr4d: the rows kernel grids (128 / (waves * R), heads,
 * sequences) and the replay kernel, one value row a wave, (128 / waves, heads, sequences). */
constexpr uint32_t kThreads = 128;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr int kRows = 4;      /* value rows a wave owns: 32 % 4 == 0 */
constexpr int kDim = 128;     /* head_k and head_v, the row's compiled-in law */

/* The workgroup's dynamic stage. A wave keeps its state in registers, so there is none; the bound
 * is stated because RDNA3 gives a workgroup 64 KiB of LDS and a future tile of the state (one
 * head is 128*128*4 = 64 KiB) has to be checked against it here, not at the launch. */
constexpr size_t kLdsBytes = 0;
static_assert(kLdsBytes <= 60 * 1024, "a workgroup's stage must fit RDNA3's 64 KiB LDS");

struct KdaMeta {
    int row[5], head[5];
    long os0, os1, ss0, ss1, ss2;
    long is0, is1, cs0, as0, hs0;
    int tokens, slots, width, verify, anchor;
    float lower, eps, scale;
};
struct KdaJob : KdaMeta {
    const uint16_t* input[5];
    const float *alog, *bias;
    float* state;
    uint16_t* out;
    const int *idx, *cu, *accepted, *init;
};

/* A bf16 as ldg_u16 leaves it, widened: the code is the register's low half, so widening is a
 * shift (libr4d's bf16w_lo). */
__device__ __forceinline__ float bf_lo(uint32_t w) { return bf16_to_f32((uint16_t)w); }

__device__ long factor_offset(long base, int logical, long pitch) {
    return base + (logical / 128) * pitch + (logical % 128);
}
/* Separate dispatch: all old factor reads finish before the next window overwrites them. */
__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void kda_replay_kernel(
    float* state, const int* idx, const int* cu, const int* accepted, const int* init, KdaMeta m) {
    const int lane = threadIdx.x & (kWave - 1), e = blockIdx.x * kWaves + (threadIdx.x / kWave),
              head = blockIdx.y, seq = blockIdx.z;
    /* A sequence without tokens this step leaves its slots as they are. */
    if (cu[(long)seq * m.cs0] == cu[(long)(seq + 1) * m.cs0]) return;
    const int anchor = idx[(long)seq * m.is0],
              scratch = idx[(long)seq * m.is0 + (m.width / 2) * m.is1];
    if (anchor < 0 || anchor >= m.slots || scratch < 0 || scratch >= m.slots || anchor == scratch)
        return;
    const long fb = (long)scratch * m.ss0 + (long)head * m.ss1;
    const long sb = (long)anchor * m.ss0 + (long)head * m.ss1 + (long)e * m.ss2;
    if (init && !init[(long)seq * m.hs0]) {
#pragma unroll
        for (int d = 0; d < 4; ++d) state[sb + lane + d * 32] = 0;
        return;
    }
    const float count = state[fb];
    const int na = accepted ? accepted[(long)seq * m.as0] : 1;
    if (!(count >= 0 && count <= 32) || count != floorf(count) || na < 1 || na > 32 ||
        (count > 0 && na > count))
        return;
    float s[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) s[d] = state[sb + lane + d * 32];
    for (int t = 0; t < (count > 0 ? na : 0); ++t) {
        const int f = 1 + t * 385;
        float k[4], prediction = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const int c = lane + d * 32;
            k[d] = state[factor_offset(fb, f + c, m.ss2)];
            s[d] *= state[factor_offset(fb, f + 256 + c, m.ss2)];
            prediction += s[d] * k[d];
        }
        const float delta =
            state[factor_offset(fb, f + 384, m.ss2)] *
            (state[factor_offset(fb, f + 128 + e, m.ss2)] - wave_sum(prediction));
#pragma unroll
        for (int d = 0; d < 4; ++d) s[d] += delta * k[d];
    }
#pragma unroll
    for (int d = 0; d < 4; ++d) state[sb + lane + d * 32] = s[d];
}
/* R value rows a wave: the loads, norms and decay exponentials, which do not depend on the state,
 * are formed once for R rows, and the rows' reductions interleave. The next token's operands are
 * loaded while this one is reduced. */
template <int R>
__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void kda_rows_kernel(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, const uint16_t* g,
    const uint16_t* beta, const float* alog, const float* bias, float* state_ptr, uint16_t* out,
    const int* idx, const int* cu, const int* accepted_ptr, const int* init, KdaMeta meta) {
    static_assert(32 % R == 0, "the 128 value rows split evenly over the waves");
    KdaJob j{};
    static_cast<KdaMeta&>(j) = meta;
    j.input[0] = q;
    j.input[1] = k;
    j.input[2] = v;
    j.input[3] = g;
    j.input[4] = beta;
    j.alog = alog;
    j.bias = bias;
    j.state = state_ptr;
    j.out = out;
    j.idx = idx;
    j.cu = cu;
    j.accepted = accepted_ptr;
    j.init = init;
    const int lane = threadIdx.x & (kWave - 1),
              e0 = (blockIdx.x * kWaves + (threadIdx.x / kWave)) * R;
    const int head = blockIdx.y, seq = blockIdx.z;
    const int begin = j.cu[(long)seq * j.cs0], end = j.cu[(long)(seq + 1) * j.cs0];
    const int accepted = j.accepted ? j.accepted[(long)seq * j.as0] : 1;
    if (begin < 0 || end <= begin || end > j.tokens || accepted < 1 ||
        accepted > (j.anchor ? 32 : j.width) ||
        (j.verify && end - begin > (j.anchor ? 32 : j.width)))
        return;
    const int read = j.idx[(long)seq * j.is0 + (j.verify && !j.anchor ? accepted - 1 : 0) * j.is1];
    if (read >= j.slots) return;
    uint16_t* const out_row = j.out + (long)head * j.os1 + e0;
    if (read < 0) {
        if (lane < R)
            for (int t = begin; t < end; ++t) out_row[(long)t * j.os0 + lane] = 0;
        return;
    }
    const bool initialized = !j.init || j.init[(long)seq * j.hs0] != 0;
    const int scratch = j.anchor ? j.idx[(long)seq * j.is0 + (j.width / 2) * j.is1] : 0;
    if (j.anchor && (scratch < 0 || scratch >= j.slots || scratch == read)) return;
    const long fb = (long)scratch * j.ss0 + (long)head * j.ss1;
    const long initial = (long)read * j.ss0 + (long)head * j.ss1 + (long)e0 * j.ss2;
    float state[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int d = 0; d < 4; ++d)
            state[r][d] = initialized ? j.state[initial + r * j.ss2 + lane + 32 * d] : 0;
    const float gain = expf(j.alog[head]);
    float shift[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) shift[d] = j.bias[head * 128 + lane + 32 * d];
    const uint16_t* const qb = j.input[0] + (long)head * j.head[0] + lane;
    const uint16_t* const kb = j.input[1] + (long)head * j.head[1] + lane;
    const uint16_t* const vb = j.input[2] + (long)head * j.head[2] + e0;
    const uint16_t* const gb = j.input[3] + (long)head * j.head[3] + lane;
    const uint16_t* const bb = j.input[4] + (long)head * j.head[4];
    struct Token {
        uint32_t q[4], k[4], g[4], beta, value[R];
    };
    auto fetch = [&](int t, Token& o) {
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            o.q[d] = ldg_u16(qb + (long)t * j.row[0] + 32 * d);
            o.k[d] = ldg_u16(kb + (long)t * j.row[1] + 32 * d);
            o.g[d] = ldg_u16(gb + (long)t * j.row[3] + 32 * d);
        }
        o.beta = ldg_u16(bb + (long)t * j.row[4]);
#pragma unroll
        for (int r = 0; r < R; ++r) o.value[r] = ldg_u16(vb + (long)t * j.row[2] + r);
    };
    Token cur;
    if (begin < end) fetch(begin, cur);
    for (int t = begin; t < end; ++t) {
        Token next;
        /* The last token loads itself again rather than a row past `end`. */
        fetch(t + 1 < end ? t + 1 : t, next);
        float qv[4], kv[4], decay[4], sq = 0, sk = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            qv[d] = bf_lo(cur.q[d]);
            kv[d] = bf_lo(cur.k[d]);
            sq += qv[d] * qv[d];
            sk += kv[d] * kv[d];
            const float raw = bf_lo(cur.g[d]);
            decay[d] = expf(j.lower / (1.0f + expf(-gain * (raw + shift[d]))));
        }
        const float qi = j.scale * rsqrtf(wave_sum(sq) + j.eps),
                    ki = rsqrtf(wave_sum(sk) + j.eps);
        const float b = 1.0f / (1.0f + expf(-bf_lo(cur.beta)));
        float predicted[R];
#pragma unroll
        for (int r = 0; r < R; ++r) predicted[r] = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            qv[d] *= qi;
            kv[d] *= ki;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                state[r][d] *= decay[d];
                predicted[r] += state[r][d] * kv[d];
            }
        }
        float value[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            predicted[r] = wave_sum(predicted[r]);
            value[r] = bf_lo(cur.value[r]);
        }
        if (j.anchor && j.verify) {
            const int f = 1 + (t - begin) * 385;
            if (e0 == 0) {
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const int c = lane + 32 * d;
                    j.state[factor_offset(fb, f + c, j.ss2)] = kv[d];
                    j.state[factor_offset(fb, f + 256 + c, j.ss2)] = decay[d];
                }
                if (!lane) j.state[factor_offset(fb, f + 384, j.ss2)] = b;
            }
            if (!lane) {
#pragma unroll
                for (int r = 0; r < R; ++r)
                    j.state[factor_offset(fb, f + 128 + e0 + r, j.ss2)] = value[r];
            }
        }
        float result[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float delta = b * (value[r] - predicted[r]);
            result[r] = 0;
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                state[r][d] += delta * kv[d];
                result[r] += state[r][d] * qv[d];
            }
        }
        float mine = 0;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            result[r] = wave_sum(result[r]);
            if (lane == r) mine = result[r];
        }
        if (lane < R) out_row[(long)t * j.os0 + lane] = f32_to_bf16(mine);
        if ((!j.anchor && j.verify) || (!j.verify && t + 1 == end)) {
            const int write = j.idx[(long)seq * j.is0 + (j.verify ? t - begin : 0) * j.is1];
            if (write >= 0 && write < j.slots) {
                const long target = (long)write * j.ss0 + (long)head * j.ss1 + (long)e0 * j.ss2;
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int d = 0; d < 4; ++d)
                        j.state[target + r * j.ss2 + lane + 32 * d] = state[r][d];
            }
        }
        cur = next;
    }
    if (j.anchor && e0 == 0 && !lane) j.state[fb] = j.verify ? (float)(end - begin) : 0.0f;
}

/* A bf16 operand shaped [tokens, heads, dim] or the fused [tokens, heads*dim], read at its own row
 * and head pitches. libr4d's `headed`, with an unbound descriptor (null data) refused the same way
 * a missing operand is: as a shape the shim does not accept. */
template <class Pitch>
bool headed(const RadTensor* t, long tokens, long heads, long dim, Pitch* row, Pitch* head) {
    if (!t || !t->data || t->dtype != RAD_BF16 || t->shape[0] != tokens ||
        t->stride[0] > INT_MAX || (t->rank == 3 && t->stride[1] > INT_MAX))
        return false;
    if (t->rank == 3 && t->shape[1] == heads && t->shape[2] == dim && t->stride[2] == 1 &&
        t->stride[1] >= dim && t->stride[0] >= heads * t->stride[1]) {
        *row = t->stride[0];
        *head = t->stride[1];
        return true;
    }
    if (t->rank == 2 && t->shape[1] == heads * dim && t->stride[1] == 1 &&
        t->stride[0] >= heads * dim) {
        *row = t->stride[0];
        *head = dim;
        return true;
    }
    return false;
}
bool vector(const RadTensor* t, uint32_t dt, long n) {
    return t && t->data && t->dtype == dt && t->rank == 1 && t->shape[0] >= n && t->stride[0] == 1;
}
int body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 13) return R3D_NO(RAD_E_INVAL);
    const long H = r3d_geti_or(a, "heads", 0);
    if (H < 1 || H > 65535 || r3d_geti_or(a, "head_k", 0) != kDim ||
        r3d_geti_or(a, "head_v", 0) != kDim)
        return R3D_NO(RAD_E_SHAPE);
    KdaJob j{};
    const RadTensor *state = r3d_opt(a, 7), *idx = r3d_opt(a, 8), *cu = r3d_opt(a, 9);
    const long tokens = r3d_dim(r3d_opt(a, 0), 0);
    /* The three null guards below stand where libr4d tests `!state.data`, `!cu.data` and
     * `!idx.data`: an unbound operand is refused as the shape it is not, the same code. */
    if (tokens < 1 || tokens > INT_MAX || !state || state->rank != 4 || state->dtype != RAD_F32 ||
        !state->data || state->shape[0] < 1 || state->shape[0] > INT_MAX ||
        state->shape[1] != H || state->shape[2] != kDim || state->shape[3] != kDim ||
        state->stride[3] != 1 || state->stride[2] < kDim ||
        state->stride[1] < kDim * state->stride[2] || state->stride[0] < H * state->stride[1] ||
        !cu || cu->rank != 1 || cu->dtype != RAD_I32 || !cu->data || cu->shape[0] < 2 ||
        cu->shape[0] > 65536 || cu->stride[0] < 1 || !idx || idx->rank != 2 ||
        idx->dtype != RAD_I32 ||
        !idx->data || idx->shape[0] < cu->shape[0] - 1 || idx->shape[1] < 1 ||
        idx->shape[1] > INT_MAX || idx->stride[1] < 1 ||
        idx->stride[0] < idx->shape[1] * idx->stride[1])
        return R3D_NO(RAD_E_SHAPE);
    for (int i = 0; i < 5; ++i) {
        if (!headed(r3d_opt(a, i), tokens, H, i == 4 ? 1 : kDim, &j.row[i], &j.head[i]))
            return R3D_NO(RAD_E_SHAPE);
        j.input[i] = (const uint16_t*)r3d_p(r3d_opt(a, i));
    }
    const RadTensor* o = r3d_opt(a, 12);
    if (!headed(o, tokens, H, kDim, &j.os0, &j.os1) || !vector(r3d_opt(a, 5), RAD_F32, H) ||
        !vector(r3d_opt(a, 6), RAD_F32, H * kDim))
        return R3D_NO(RAD_E_SHAPE);
    const RadTensor *accepted = r3d_opt(a, 10), *init = r3d_opt(a, 11);
    if ((accepted && !vector(accepted, RAD_I32, cu->shape[0] - 1)) ||
        (init && !vector(init, RAD_I32, cu->shape[0] - 1)))
        return R3D_NO(RAD_E_SHAPE);
    const char* mode = r3d_gets(a, "mode");
    if (!mode || (std::strcmp(mode, "verify") && std::strcmp(mode, "prefill")))
        return R3D_NO(RAD_E_INVAL);
    j.verify = !std::strcmp(mode, "verify");
    const char* form = r3d_gets(a, "state_form");
    j.anchor = form && !std::strcmp(form, "anchor");
    if (form && !j.anchor && std::strcmp(form, "per_candidate")) return R3D_NO(RAD_E_INVAL);
    if (j.anchor && idx->shape[1] < 2) return R3D_NO(RAD_E_SHAPE);
    j.lower = (float)r3d_getf(a, "lower_bound", 0);
    j.eps = (float)r3d_getf(a, "l2_eps", 0);
    j.scale = (float)r3d_getf(a, "scale", 1);
    if (!std::isfinite(j.lower) || j.lower >= 0 || !std::isfinite(j.eps) || j.eps <= 0 ||
        !std::isfinite(j.scale))
        return R3D_NO(RAD_E_SHAPE);
    j.tokens = (int)tokens;
    j.slots = (int)state->shape[0];
    j.width = (int)idx->shape[1];
    j.alog = (const float*)r3d_p(r3d_opt(a, 5));
    j.bias = (const float*)r3d_p(r3d_opt(a, 6));
    j.state = (float*)r3d_wp(state);
    j.out = (uint16_t*)r3d_wp(o);
    j.ss0 = state->stride[0];
    j.ss1 = state->stride[1];
    j.ss2 = state->stride[2];
    j.idx = (const int*)idx->data;
    j.is0 = idx->stride[0];
    j.is1 = idx->stride[1];
    j.cu = (const int*)cu->data;
    j.cs0 = cu->stride[0];
    if (accepted) {
        j.accepted = (const int*)accepted->data;
        j.as0 = accepted->stride[0];
    }
    if (init) {
        j.init = (const int*)init->data;
        j.hs0 = init->stride[0];
    }
    const unsigned seqs = (unsigned)(cu->shape[0] - 1);
    const hipStream_t st = (hipStream_t)stream;
    if (j.anchor && j.verify) {
        /* One value row a wave, a workgroup's four waves covering four rows: 128 / 4 = 32
         * workgroups, as libr4d grids the replay. Clear the error flag before the chevron and read
         * it after: hipGetLastError reports the first error since it was last cleared, so an
         * uncleared read blames this launch for an earlier one. */
        (void)hipGetLastError();
        R3D_KLAUNCH(kda_replay_kernel, dim3(kDim / kWaves, (unsigned)H, seqs), dim3(kThreads), st,
                    j.state, j.idx, j.cu, j.accepted, j.init, static_cast<KdaMeta>(j));
        const hipError_t re = hipGetLastError();
        if (re != hipSuccess) {
            if (r3d_trace_on())
                std::fprintf(stderr, "[r3d] kda_update: %s on the replay: grid (%u, %u, %u)\n",
                             hipGetErrorName(re), (unsigned)(kDim / kWaves), (unsigned)H, seqs);
            return R3D_NO(RAD_E_DEVICE);
        }
    }
    /* The 128 value rows split over the workgroup's waves, R rows a wave: 128 / (4 waves * 4 rows)
     * = 8 workgroups a (head, sequence), as libr4d grids it. */
    (void)hipGetLastError();
    R3D_KLAUNCH(R3D_KNAME((kda_rows_kernel<kRows>)),
                dim3(kDim / (kWaves * kRows), (unsigned)H, seqs), dim3(kThreads), st, j.input[0],
                j.input[1], j.input[2], j.input[3], j.input[4], j.alog, j.bias, j.state, j.out,
                j.idx, j.cu, j.accepted, j.init, static_cast<KdaMeta>(j));
    const hipError_t ke = hipGetLastError();
    if (ke != hipSuccess) {
        if (r3d_trace_on())
            std::fprintf(stderr, "[r3d] kda_update: %s on the rows launch: grid (%u, %u, %u) %d\n",
                         hipGetErrorName(ke), kDim / (kWaves * kRows), (unsigned)H, seqs, kRows);
        return R3D_NO(RAD_E_DEVICE);
    }
    return RAD_OK;
}
}  // namespace

extern "C" {

int r3d_rad_kda_update(const RadArgs* a, RadStream s) { return body(a, s); }

}  // extern "C"
