# Four ranks on a PCIe fabric: what pays and what doesn't

Every TP4 lever tried on 4x R9700 (PCIe Gen5 P2P, no NVLink), and what decided it. Use it before
re-attacking the same ground on another card or fabric. All numbers are paired within-session
windows (bar 0.5% plus a sign test) at `prefill-span: 2`, `tp-wire: wht6`, `placement: all_vram`,
iq4r MoE, unless stated.

Shorthand: `pp` prefill; `c8` eight concurrent streams; `M` rows in a collective; `ti8` the int8
peer wire (`tp-wire: wht6`'s coding); `hc_write` the fused all-reduce + residual-write row.

## What shipped

- **`prefill-span: 2`**: overlaps the two half-chunk all-reduces with compute through the second
  lane's rows. +5-11% pp, but only with `checkpoint-interval: 2048` and `all_vram` (placement flips
  its sign).
- **The four-rank link-hold release** (`ar_*_4rank` rows release the link gate themselves, as the
  two-rank rows do). Without it every yielding DMA (stager copies, expert prefetch) idles behind
  each collective: removing it costs 3.7% at c8. Any new collective row must honour the gate.
- **Four-rank selftest coverage**: the wide-collective and fused-write cases at world 4.

## What was tried and lost

| lever | result | why it is closed |
|---|---|---|
| widen the MoE fast band to four classes | no change | the cap only bites 3-bit formats; inert for 4-bit experts |
| fused all-reduce + `hc_write` on the exact-wire band | never selected | `wht6` takes the lossy band, and the fused row is exact-only. Never timed in its own territory |
| per-class grid split for the MoE GEMM | not constructible in one launch | a run's class is not a property of `blockIdx.y`. A two-launch split was never tried |
| fused `ti8` two-shot + `hc_write` | c8 -0.43%, c16 sign-split | the saved launches sit at the ~4 us launch floor; the epilogue serialises behind the all-reduce spin |
| more all-reduce blocks (16 -> 32 -> 64) | GB/s falls (M=2048: 34.5 -> 31.4; M=128: 18.4 -> 7.4) | handshake-bound, not launch-starved |
| `tp-wire: exact` everywhere | -7.6 to -8.3% at span off | exact wins only at decode steps of 25 tokens or fewer, which it already serves |
| rank-load balancer | +1.3% at 64K only | changes output bits; not worth it |
| step-graph capture | none for the trunk | trunk submission is device backpressure, so the host is hidden. Evidence is a synthetic ring, not an engine capture |
| `ti8` MoE gather row | premise false | on the lossy wire the gather's all-reduce already rides `ti8` |
| faster `ti8` codec | none possible | encode is 4% of the row (36 of ~900 us); the peer push runs at ~60 GB/s, PCIe Gen5 x16 line rate. The "22% tax" against exact is the link |

Only a lever that ran correctly and still lost to a measured mechanism is closed for good: the
block count, the `ti8` fusion and the codec. The fused exact-wire rows, the two-launch class split
and draft-pass graph capture were never built or never timed where they would pay.

## Open levers, ranked

1. **Bigger steps for decode-heavy traffic**: `checkpoint-interval: 4096` with an 8192-token budget
   measured +2-3% c8 and -2-4% pp. Two flags, no code. A prefill chunk is
   `min(budget, interval x span)`; the engine prints the effective chunk at startup.
2. **Fused gather/write rows for exact-wire configs**, where they still have territory.
3. **The span floor at TP4 with tiered experts**: eight checkpoint intervals, priced on Flash-Next at
   TP2 only.
4. **A one-shot `ti8` band for 128-512 KiB messages**, which now pay the two-shot's 1.5N and two
   syncs.
5. **Draft-pass graph capture**, if low-concurrency latency matters: the host gap is real at c1/c2
   and hidden at c8.
6. **The dense pipelined prefill at four ranks** (`qwen35_fp8`, `qwen35_w8a8`; GUIDE
   §4.5). It runs at two ranks only (`g.world == 2` in the plugins' declare); enabling it needs a
   timing against the serial step at 18K-64K prompts.

## Instruments

- `debug-graph: true` lists which row each op resolves to. It lists declared bands, so confirm
  against the running step.
- `r4d_selftest --perfops` times small ops (every elementwise op costs ~4-4.6 us regardless of
  payload); `--perfhc` and the GEMM/MoE modes time the rest.
- `rad-kbench` for kernel-against-reference replay; BetterBench's fixed profile decides promotion.
