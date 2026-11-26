/* r3d_router_group_topk.hip -- router_sigmoid_group_topk: group-limited sigmoid routing.
 *
 * p[e] = sigmoid(logit[e]) and rank[e] = p[e] + bias[e]. The experts fall into `n_group` equal
 * groups; a group ranks by the sum of its two highest ranks, the `topk_group` best groups are
 * kept, and the top_k are taken from the kept groups' experts only. The weights are the kept
 * experts' unbias'd p, times `route_scale` (divided by their own sum first, if `norm`): the
 * correction bias selects, it never weights.
 *
 *   logits [M][n_expert] f32, bias [n_expert] f32  ->  ids [M][top_k] i32, weights [M][top_k] f32
 *
 * One wave per row: a lane holds experts lane, lane+32, ... as 64-bit keys -- the rank's
 * order-preserving bits (-0 folded to +0, a NaN keyed 0 so it is never selected) above ~expert, so
 * a tie goes to the lower expert -- and lane g holds group g's sum. Round j takes the largest key
 * below round j-1's winner, which is a serial scan with a strict `>` done as a wave reduce: equal
 * ranks to the lower expert, equal group sums to the lower group, a group with no finite pair
 * never. A row with fewer selectable experts than top_k names expert 0 in the slots left over,
 * weighted by expert 0's p, as libr4d's wave leaves it.
 *
 * The shim takes the row count from `logits` and every pitch from the operands, and refuses:
 * fewer than 1 or more than 512 experts, top_k outside 1..32, fewer than 1 or more than 32 groups,
 * a group count that does not divide the experts into at least two each, topk_group outside
 * 1..n_group, a top_k wider than the kept groups can supply, a non-finite or non-positive
 * route_scale, a logits plane that is not [M >= 1, pitch >= n_expert] with rows of stride 1, a
 * bias that is not [n_expert] with a stride of at least 1, ids and weights that are not [M, pitch
 * >= top_k] with rows of stride 1, and any dtype other than f32 logits, f32 bias, i32 ids and f32
 * weights. Null operands are RAD_E_INVAL and every geometry rule above RAD_E_SHAPE, as in
 * libr4d's launcher; libr3d's convention (r3d_args.h: an issue bug, not a shape) names the null
 * case, and libr4d spells it inside its shape clause.
 *
 * gfx1102, from libr4d's router_group_wave_kernel (r4d_glm.hip). Two adaptations, no change to the
 * arithmetic:
 *   * One wave per block, one block per row, where libr4d packs four rows into a 128-thread block.
 *     A wave reduction in a loop body miscompiles for a non-first wave of a block here -- wave 0
 *     is right, waves 1+ drop out (r3d_moe_router_topk_bf16.hip) -- and every loop in this kernel
 *     is one: the group ranking runs a wave merge per group, the group selection and the top_k
 *     each run one per round. One wave a block retires the case, and the __syncthreads that stages
 *     the scores is then a single-wave barrier.
 *   * The wave reductions are libr3d's DPP16 and v_permlanex16 lane exchanges (r3d_common.h),
 *     which is what libr4d's butterflies lower to and what `__shfl_xor` costs on RDNA3 (a
 *     ds_bpermute and a dscnt wait). Same lanes, same order, so the merge is bit-identical to
 *     libr4d's: the top-2 merge over 16, 8, 4, 2, 1, the group max and the expert max are the r4d
 *     kernel's, key for key, so a tied row, a row with a -0 rank and a row with a NaN rank land on
 *     the same bytes in both backends. The scores are staged in LDS and the kept sum read back lane
 *     by lane (the libr3d router's idiom) instead of a gather shuffle per key.
 *
 * No matrix instruction: this is a selection over at most 512 f32, bandwidth-bound and
 * register-resident. The only LDS is a small static stage -- `n_expert` scores plus 32 kept sums,
 * 2.2 KiB at the widest row, far under RDNA3's 64 KiB -- so there is no dynamic LDS to size
 * against the cap in the launcher.
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#define R3D_BARRIER() do { __syncthreads(); asm volatile("" ::: "memory"); } while (0)

namespace {

/* (order-preserving rank, ~expert) as one 64-bit key: the rank's monotone bits (-0 folded to +0)
 * above the complemented expert, so a tie goes to the lower expert. A NaN rank keys 0, which no
 * round selects. libr4d's router_rank_key, unchanged. */
__device__ __forceinline__ unsigned long long r3d_router_rank_key(float rank, uint32_t e) {
    uint32_t u = __float_as_uint(rank);
    if (u == 0x80000000u) u = 0;
    const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return rank == rank ? ((unsigned long long)k << 32) | (uint32_t)~e : 0ull;
}

/* The wave's largest key: the xor butterfly over 16, 8, 4, 2, 1, the halves exchanged as a 64-bit
 * shuffle does. Keys are unique (the low half is the expert), so the winner is unique and the
 * butterfly's order cannot matter. */
__device__ __forceinline__ unsigned long long r3d_router_wave_max(unsigned long long b) {
    unsigned long long o;
    o = r3d_xor_lane_u64<16>(b);
    b = b > o ? b : o;
    o = r3d_xor_lane_u64<8>(b);
    b = b > o ? b : o;
    o = r3d_xor_lane_u64<4>(b);
    b = b > o ? b : o;
    o = r3d_xor_lane_u64<2>(b);
    b = b > o ? b : o;
    o = r3d_xor_lane_u64<1>(b);
    b = b > o ? b : o;
    return b;
}

/* One butterfly round of the two best: the higher head, and the higher of the two tails and the
 * lower head. A NaN rank never reaches here -- a lane's own scan drops it on both `>` tests. */
template <int N>
__device__ __forceinline__ void r3d_router_top2_merge(float& t0, float& t1) {
    const float o0 = r3d_xor_lane<N>(t0), o1 = r3d_xor_lane<N>(t1);
    const float hi = fmaxf(t0, o0), lo = fmaxf(fminf(t0, o0), fmaxf(t1, o1));
    t0 = hi;
    t1 = lo;
}

/* The wave's two best, in the order libr4d's `for (mask = 16; mask; mask >>= 1)` runs them. */
__device__ __forceinline__ void r3d_router_top2_wave(float& t0, float& t1) {
    r3d_router_top2_merge<16>(t0, t1);
    r3d_router_top2_merge<8>(t0, t1);
    r3d_router_top2_merge<4>(t0, t1);
    r3d_router_top2_merge<2>(t0, t1);
    r3d_router_top2_merge<1>(t0, t1);
}

/* One row, one wave, one block. OWN is a template bound because the keys are registers: a
 * lane-derived trip count would index them divergently. A tail expert reads address 0 of its row
 * and keys absent; it is never stored. */
template <int OWN>
__global__ __launch_bounds__(kWave) void r3d_router_group_topk_kernel(
    const float* __restrict__ logits, const float* __restrict__ bias, int32_t* __restrict__ ids,
    float* __restrict__ weights, long lp, long bp, long ip, long wp, uint32_t m, uint32_t ne,
    uint32_t k, uint32_t ng, uint32_t kg, int norm, float factor) {
    __shared__ float s_score[kWave * OWN];   // the row's p, so a weight is read back, not recomputed
    __shared__ float s_sel[kWave];           // the kept p, lane j = round j
    const uint32_t lane = threadIdx.x, r = blockIdx.x;
    if (r >= m) return;
    const uint32_t gs = ne / ng;             // experts a group; the shim divided ne by ng exactly

    float p[OWN], rank[OWN];
#pragma unroll
    for (uint32_t i = 0; i < OWN; ++i) {
        const uint32_t e = lane + i * kWave, s = e < ne ? e : 0;
        p[i] = 1.0f / (1.0f + expf(-logits[(size_t)r * lp + s]));   // the reference's sigmoid
        rank[i] = e < ne ? p[i] + bias[(size_t)s * bp] : -INFINITY;
        s_score[lane * OWN + i] = p[i];
    }
    R3D_BARRIER();

    // The groups, largest sum first: lane g takes group g's two-best sum after the wave merge.
    float group = -INFINITY;
    for (uint32_t g = 0; g < ng; ++g) {
        float t0 = -INFINITY, t1 = -INFINITY;
#pragma unroll
        for (uint32_t i = 0; i < OWN; ++i) {
            const uint32_t e = lane + i * kWave;
            if (e < ne && e / gs == g) {
                const float v = rank[i];
                if (v > t0) {
                    t1 = t0;
                    t0 = v;
                } else if (v > t1)
                    t1 = v;
            }
        }
        r3d_router_top2_wave(t0, t1);
        if (lane == g) group = t0 + t1;
    }
    const unsigned long long gkey =
        lane < ng && group > -INFINITY ? r3d_router_rank_key(group, lane) : 0ull;

    // The kg best groups, by key: bit g of `kept` says group g's experts are selectable.
    unsigned kept = 0;
    unsigned long long below = ~0ull;
    for (uint32_t j = 0; j < kg; ++j) {
        const unsigned long long best = r3d_router_wave_max(gkey < below ? gkey : 0ull);
        if (best) kept |= 1u << (int)~(uint32_t)best;
        below = best;
    }

    // Then the top_k of the kept groups' experts, the same rounds on the experts' own keys.
    unsigned long long key[OWN];
#pragma unroll
    for (uint32_t i = 0; i < OWN; ++i) {
        const uint32_t e = lane + i * kWave;
        key[i] = e < ne && ((kept >> (e / gs)) & 1u) ? r3d_router_rank_key(rank[i], e) : 0ull;
    }
    unsigned long long mine = 0;
    below = ~0ull;
    for (uint32_t j = 0; j < k; ++j) {
        unsigned long long best = 0;
#pragma unroll
        for (uint32_t i = 0; i < OWN; ++i) {
            const unsigned long long c = key[i] < below ? key[i] : 0ull;
            best = c > best ? c : best;
        }
        best = r3d_router_wave_max(best);
        if (lane == j) mine = best;
        below = best;
    }

    // The weights are the unbias'd scores, summed in round order -- lane order, since lane j parked
    // round j's winner -- and the sum is over the kept k, not over the wave.
    const uint32_t e = mine ? (uint32_t)~(uint32_t)mine : 0u;
    const float chosen = s_score[(e & (kWave - 1)) * OWN + (e >> 5)];
    s_sel[lane] = chosen;
    R3D_BARRIER();
    float total = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kWave; ++j)
        if (j < k) total += s_sel[j];
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    if (lane < k) {
        ids[(size_t)r * ip + lane] = (int32_t)e;
        weights[(size_t)r * wp + lane] = chosen * mult;
    }
    static_assert(sizeof(s_score) + sizeof(s_sel) <= 60 * 1024,
                  "the row's stage is 2.2 KiB at 512 experts, well under RDNA3's 64 KiB");
}

}  // namespace

extern "C" {

int r3d_rad_router_sigmoid_group_topk(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 4) return R3D_NO(RAD_E_INVAL);
    const long long ne = r3d_geti_or(a, "n_expert", 0), k = r3d_geti_or(a, "top_k", 0),
                    ng = r3d_geti_or(a, "n_group", 0), kg = r3d_geti_or(a, "topk_group", 0);
    const float factor = (float)r3d_getf(a, "route_scale", 0.0);
    const RadTensor* lg = r3d_opt(a, 0);
    const RadTensor* bs = r3d_opt(a, 1);
    const RadTensor* id = r3d_opt(a, 2);
    const RadTensor* w = r3d_opt(a, 3);
    if (!r3d_p(lg) || !r3d_p(bs) || !r3d_wp(id) || !r3d_wp(w)) return R3D_NO(RAD_E_INVAL);
    if (lg->dtype != RAD_F32 || bs->dtype != RAD_F32 || id->dtype != RAD_I32 ||
        w->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);

    const long long m = r3d_dim(lg, 0);   // the row count of the call, not the band's bound
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || ng < 1 || ng > 32 || ne % ng || ne / ng < 2 ||
        kg < 1 || kg > ng || k > kg * (ne / ng) || !std::isfinite(factor) || factor <= 0 ||
        lg->rank != 2 || m < 1 || m > 65535 || r3d_dim(lg, 1) != ne || bs->rank != 1 ||
        r3d_dim(bs, 0) != ne || id->rank != 2 || w->rank != 2 || r3d_dim(id, 0) != m ||
        r3d_dim(w, 0) != m || r3d_dim(id, 1) != k || r3d_dim(w, 1) != k ||
        r3d_stride(lg, 1) != 1 || r3d_stride(id, 1) != 1 || r3d_stride(w, 1) != 1 ||
        r3d_stride(lg, 0) < ne || r3d_stride(id, 0) < k || r3d_stride(w, 0) < k ||
        r3d_stride(bs, 0) < 1)
        return R3D_NO(RAD_E_SHAPE);

    const int norm = (int)(r3d_geti_or(a, "norm", 0) != 0);
    const long lp = r3d_stride(lg, 0), bp = r3d_stride(bs, 0), ip = r3d_stride(id, 0),
               wp = r3d_stride(w, 0);
    dim3 grid((unsigned)m), blk(kWave);   // one wave a row, one block a row (gfx1102)
    const float* lgd = (const float*)r3d_p(lg);
    const float* bpd = (const float*)r3d_p(bs);
    int32_t* ipd = (int32_t*)r3d_wp(id);
    float* wpd = (float*)r3d_wp(w);
    // 9 keys a lane to 288 experts (DeepSeek-V3's 256, GLM-5.3's 288), 16 to 512; libr4d's split.
    if (ne <= (long long)9 * kWave)
        R3D_KLAUNCH(r3d_router_group_topk_kernel<9>, grid, blk, (hipStream_t)s, lgd, bpd, ipd,
                    wpd, lp, bp, ip, wp, (uint32_t)m, (uint32_t)ne, (uint32_t)k, (uint32_t)ng,
                    (uint32_t)kg, norm, factor);
    else
        R3D_KLAUNCH(r3d_router_group_topk_kernel<16>, grid, blk, (hipStream_t)s, lgd, bpd, ipd,
                    wpd, lp, bp, ip, wp, (uint32_t)m, (uint32_t)ne, (uint32_t)k, (uint32_t)ng,
                    (uint32_t)kg, norm, factor);
    return (int)hipGetLastError();
}

}  // extern "C"
