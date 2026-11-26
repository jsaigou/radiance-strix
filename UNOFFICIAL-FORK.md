# About this unofficial fork

**This is an unofficial fork of Radiance. The original is at <https://codeberg.org/StillDeadcode/radiance>**
(author: Deadcode / StillDeadcode; project site <https://radiance.stilldeadco.de/>). Use the original for releases,
documentation, support and bug reports. This fork is not affiliated with or endorsed by the original author.

All credit for Radiance belongs to Deadcode and the radiance contributors listed in [`NOTICE`](NOTICE).

## Branch `radiance-2x-port`: the kernel work on upstream 2.x

This branch is the original at commit [`f1113ee9`](https://codeberg.org/StillDeadcode/radiance) (`v2.2.0` plus 102 commits) with this fork's kernel, format,
architecture-plugin and tooling work forward-ported onto it. It is published as **one snapshot commit** on top of this fork's 1.17-based history; the
original's own per-commit history, and its authors, are in the original repository. All credit for the 2.x engine belongs to Deadcode and the radiance
contributors (see [`NOTICE`](NOTICE)). The Haru work on this fork's `main` is **not** included.

**One file is left out.** `data/flint-zero-bias.safetensors` (an all-zero tensor that the Flint conversion recipes pass to `rad-convert --add`) is not
published here, because this repository's push guard never publishes `*.safetensors` files. Take it from the original at the commit above.

**Identifiers that moved** (upstream 2.x had taken the same numbers; its meanings are kept): routed-expert formats IQ4_NL, Q5_0, Q2_0, IQ3_XXS, IQ2_S,
`w3nl64a8h` and IQ4_XS are 13 to 19 (were 10 to 16); the kernel template value of the 3-bit codebook is `W4 = 12` (was 5); its interleave mode bit is 7 (was 5).

**Behaviour differences from the 1.17-based fork.** 2.x takes its settings from a YAML file (`radiance --config FILE`), and the packed KV caches (`fp8_v4`,
`k6_v4`) no longer exist: use `i8`.

**What was tested** (Strix Halo, gfx1151, ROCm 10.1.0; one container, Qwen3.8-Flash-Next `flash-size.rad`, `RADIANCE_KERNELS=libhalo`, one GPU):

- Against the 1.17-based fork engine on the same container with `bf16` KV: mean KL 0.000000 and top-1 agreement 1.0000 over 1,575,352 positions.
- 1,405 English and Japanese benchmark prompts (old engine with `k6_v4` KV, this branch with `i8`): 93.8% against 93.9% (English, 835), 85.3% against 84.9%
  (Japanese, 570); no task differs significantly.
- Speed against the old engine: single-stream decode about 2-3% **slower**; three-stream decode and time to first token level.
- At 131072 tokens with 3 sequences and speculation depth 2: tool calls, prefix-cache reuse across turns, and three concurrent 126k-token prompts.
- CPU test suite: the failing tests are the ones that also fail on the unmodified base commit with this toolchain (`museglimmer_test` does not compile there
  either). The two GPU tests that fail also fail identically on the base.

**Not tested on this branch:** every other container (including the Qwen3.6 and Qwen3.8-27B ones the 1.17-based branches were checked on), the lossless
GGUF import end to end, multiple GPUs, other GPUs, long soaks.

## Why this fork exists

We run Radiance on AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S, gfx1151) and found some bugs while evaluating it.
We fixed two and wrote up the others, then went on to add the kernels, formats and tooling needed to serve Unsloth-Dynamic-style, mixed-precision
models on that hardware. We did not want to create an account on Codeberg to file the bugs, so instead of keeping the work private we publish it here. Nothing here is meant to compete with or replace the original; if the original author
wants any of it, they are welcome to take it (it is Apache-2.0), and this fork can then be dropped.

## What is different from the original

| Branch | Contents |
|---|---|
| `main` | An exact mirror of the original's `main`, including all release tags. Never committed to. |
| `unofficial-fixes` (default) | The original at commit `451b136` (2026-11-05, which is `v1.17.0` plus 58 later commits) **plus** the changes below. |
| `kernel-full` | The same history the default branch now carries (kept as a named reference for the kernel work). |

Changes on `unofficial-fixes`, relative to the original:

1. **Bug fix:** `libhalo/halo_attn_decode_h256_gqa6.hip` — MTP speculation at depth 3 aborted on Strix Halo with
   "scratch buffer absent or too small". The declare-time scratch sizing now also sweeps `q_len`. Details and verification:
   [`fork-notes/01-libhalo-mtp-depth3-scratch-undersized.md`](fork-notes/01-libhalo-mtp-depth3-scratch-undersized.md).
2. **Bug fix:** `libhalo/halo_attn_prefill_h256_gqa6.hip` — a mixed step (decoding rows plus two or more prefill chunks) silently overwrote the
   decoding sequence's attention output with stale scratch, giving degenerate answers in roughly 10-25% of concurrent requests. The split-KV merge
   now skips rows outside its launch's own sequence range. Details and verification:
   [`fork-notes/05-libhalo-prefill-merge-corrupts-decode-rows.md`](fork-notes/05-libhalo-prefill-merge-corrupts-decode-rows.md).
3. **Kernels and formats for mixed-precision models** (about 75 commits): a 3-bit routed-expert codebook (`w3nl64a8h`); unrotated expert GEMMs for IQ3_XXS,
   IQ2_S, IQ4_NL, IQ4_XS, Q5_0 and Q2_0; Q8_0 and Q6_K dense kernels; IQ4_XS and IQ4_NL embedding gathers; a faster MoE decode kernel (1.1x to 1.4x on the
   expert kernel); per-layer mixed-format support in the `qwen4exp_fp8` plugin; and a lossless GGUF import plugin for Qwen3.6. Details and measurements:
   [`docs/FORK-KERNELS.md`](docs/FORK-KERNELS.md).
4. **Tooling** to build `.rad` containers from bf16 weights with an Unsloth Dynamic GGUF as the format map (`tools/ud_map`, `tools/ud_repack`,
   `tools/q6k_bench`) and the recipes for the containers we built: [`docs/RAD-FILES.md`](docs/RAD-FILES.md).
5. **Documentation for Strix Halo**: [`docs/STRIX-HALO.md`](docs/STRIX-HALO.md) (platform, unified-memory behaviour, measured results).
6. This notice (`README.md` banner, this file, `docs/STRIX-HALO.md`, `docs/FORK-KERNELS.md`, `docs/RAD-FILES.md` and the `fork-notes/` folder).

The complete list of changed and added files is `git diff --stat 451b136..HEAD`. Changed source files are Apache-2.0 like the originals; the commit
history is the record of who changed what and when.

Bugs we found but did **not** fix (descriptions only): [`fork-notes/`](fork-notes/).

## What was tested, and what was not

- Hardware: AMD Ryzen AI MAX+ 395 (Radeon 8060S, gfx1151), 123 GB unified memory, Linux 7.2.11, `amdgpu.cwsr_enable=1`.
- Toolchain: ROCm 10.1.0 (HIP 7.16, clang 23). The older ROCm 7.1 packages did not compile `libr3d`.
- Models: the bug fixes were checked on `qwen3.6-35b-a3b-fp8.rad` (Qwen3.6-35B-A3B, FP8, MTP) at speculation depths 2-8, with FP8 and bf16 KV caches.
  The kernel and format work was exercised on Qwen3.6-35B-A3B (W4/W5 experts), Qwen3.8-27B (FP8) and Qwen3.8-Flash-Next (three containers, 84-87 GiB);
  see [`docs/STRIX-HALO.md`](docs/STRIX-HALO.md). Run with `RADIANCE_KERNELS=libhalo` (see fork-notes/02).
- Not tested: other GPUs, multi-GPU, the `v1.17.0` tag itself, long soaks, image/video/audio models, an absolute FP8/bf16 quality baseline for Flash-Next.

No warranty, as in the Apache-2.0 license. For anything beyond the changes above, the original project is the reference.

## Building (Strix Halo / gfx1151)

```
export ROCM_PATH=/path/to/rocm-10.1 PATH=$ROCM_PATH/bin:$PATH LD_LIBRARY_PATH=$ROCM_PATH/lib
cmake -S . -B build -G Ninja -DRAD_GPU_TARGETS=gfx1151 -DCMAKE_PREFIX_PATH=$ROCM_PATH && cmake --build build -j8
RADIANCE_HOME=$PWD/build/radiance_home RADIANCE_KERNELS=libhalo build/bin/radiance --model <file.rad> ...
```

The engine refuses to start unless `amdgpu.cwsr_enable=1`. The original's own README has the full instructions.

## Licensing

Apache-2.0, same as the original. [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) are unchanged. As in the original, two parts carry their
own terms: `core/server/systemone.cpp` includes text under the LFM Open License v1.0 (`core/server/LICENSE.lfm`, which limits commercial
use) and `data/vectors/` carries the Qwen community license. Model weights keep the license of the model they were converted from.
Apache-2.0 section 4(b) asks for a prominent notice of changed files: this file, the README banner and `git diff --stat 451b136..HEAD` are that notice.

## Contact

Issues with this fork's changes: the Issues tab of this repository. Anything about Radiance itself: the original project on Codeberg.
