---
license: apache-2.0
base_model: higurashi-ai/Haru-4B
pipeline_tag: audio-to-audio
language:
- en
- ja
tags:
- radiance
- rad
- speech
- full-duplex
- speech-to-speech
- haru
- amd
- strix-halo
- gfx1151
library_name: radiance
---
# Haru-4B, bf16, Radiance `.rad`

[Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B), a low-latency full-duplex spoken dialogue model (it listens and speaks at the same time, one 133 ms frame at a time), converted without any change of precision into a container for the **Radiance** inference engine on an AMD Strix Halo machine.

> **Unofficial.** This is a `.rad` container for the **Radiance** inference engine, built and tested by a third party. It is **not** a PyTorch, transformers, MLX or vLLM checkpoint and will not load in them (for those, use the
> [original Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B)). It needs the engine fork described below. The original Radiance project is at <https://codeberg.org/StillDeadcode/radiance>.

## Quick start (read this first)

**This file does not run in llama.cpp, Ollama, vLLM, transformers or MLX.** It needs the Radiance engine, in our unofficial fork, built for your GPU. It is a **speech** model: it is driven by a streaming audio session, not by an OpenAI chat request.

**You need**
- An AMD Strix Halo machine (Ryzen AI MAX+ 395, Radeon 8060S, gfx1151). It is the only hardware we have tested.
- Linux with ROCm 10.1 (the TheRock build) and the kernel parameter `amdgpu.cwsr_enable=1` (the engine refuses to start without it; changing it needs a reboot).
- `git`, `cmake`, `ninja` and a C++ compiler.

**1. Build the engine (once, roughly 10 to 20 minutes)**
```
git clone --branch haru/main https://github.com/jsaigou/radiance-strix.git && cd radiance-strix
export ROCM_PATH=/path/to/rocm-10.1 PATH=$ROCM_PATH/bin:$PATH LD_LIBRARY_PATH=$ROCM_PATH/lib
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1151 -DCMAKE_PREFIX_PATH=$ROCM_PATH
cmake --build build -j8
```

**2. Download the container**
```
hf download jsaigou/Haru-4B-bf16-radiance haru-bf16.rad --local-dir .
```

**3. Run it**
```
RADIANCE_HOME=$PWD/build/radiance_home RADIANCE_KERNELS=libhalo \
  build/bin/radiance --model haru-bf16.rad --host 127.0.0.1 --port 8102 --tp 1 \
  --max-model-len 14500 --max-num-seqs 2 --num-speculative-tokens 0 \
  --duplex-max-lanes 2 --duplex-max-ctx 14500
```

The engine then serves full-duplex audio sessions over a WebSocket (24 kHz audio in and out, 3,200 samples per frame, 7.5 frames a second). The session protocol is described in the fork:
[`arch/haru_common/README.md`](https://github.com/jsaigou/radiance-strix/blob/haru/main/arch/haru_common/README.md) and `core/gen/duplex_driver.*`. As in the original model, **Haru ships without voice clips**: a session is opened with a character card and a short reference clip of a voice you have the right to use.

Our deployment also passes `--duplex-temperature 0.55 --duplex-eop-bias 1 --duplex-dry-multiplier 1.0`; those are sampler settings we preferred in our own listening, not requirements.

If it will not start: check `cat /sys/module/amdgpu/parameters/cwsr_enable` prints `1`, that the build used `RAD_GPU_TARGETS=gfx1151`, and that you built branch `haru/main` of the fork, not upstream's 2.x. We built and tested this container at commit `3eafc24` of that branch ("Release HS-1b") and it is the engine we run; later commits on the branch are untested with this file.

## What is in this repository

| File | Size | SHA-256 |
|---|---:|---|
| `haru-bf16.rad` | 10549928384 bytes | `1fcecab82259ba0f6fa2c229f35f56700d8fd3e6fcceaba102dae0879cc79123` |

## How it was made

Source: [higurashi-ai/Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B) at revision `8d6e83bcd3701a76fc27a764965665da1df7da78`, converted on 2026-11-20 with the fork's `rad-convert` and the one-rule recipe `data/recipes/haru-bf16.recipe` (`* cast dtype=bf16`).
All 1022 tensors of the checkpoint are already bfloat16, so the conversion is an exact cast: nothing is quantised, pruned or fine-tuned. The audio projectors, the latent scale and bias, the norms and the MeanFlow head are declared float32 by the model's blocks; the loader widens their bfloat16 values to float32 exactly, so they stay float32 inside the container.

## Measured by us (one AMD Ryzen AI MAX+ 395, gfx1151, one lane unless stated)

- **Fidelity against the float32 reference** (the model's own Python implementation run in float32): the first 120 English frames match at cosine 0.9999 (log-mel distance 0.04); Japanese is a mean cosine of 0.988. Both degrade at the frames of our test recording that are sensitive to small numeric differences, in English at the same frames where the bfloat16 reference itself does. Fed its own previous output, as in real use, even the reference diverges chaotically from a fixed recording, so these are fidelity figures on a fixed clip, not a quality score.
- **Latency:** one decoding step is 68 to 78 ms at the median and 77 to 81 ms at the 95th percentile (the budget is the 133 ms frame); opening a session with a 70-frame reference voice takes about 0.8 s. Two lanes meet the frame budget but not a 100 ms 95th-percentile target. The machine's baseline drifted by about 7 ms during our measurements, so read these as a range.
- **Memory:** the container is 9.8 GiB; with the engine's state for two lanes of 14,500 positions our deployment budgets 12.75 GiB.
- **Conformance:** the Forge duplex conformance suite (10 checks) passes against this container.

We have not measured speech quality beyond listening to English and Japanese output, and we make no claim about the other languages the original model supports.

## Limitations

As the original authors state: Haru-4B is a pretrained base model, released for research. It holds a natural conversation, but it is not a reliable source of facts and does not follow instructions the way an assistant does. It can imitate the voice of a reference clip; only use voices you have the right to use.

## License and credits

Apache License 2.0, as the base model [higurashi-ai/Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B). The `LICENSE` file in this repository is that license. Model credit: higurashi-ai (code: <https://github.com/higurashi-ai/Haru>); per its authors it uses a Qwen3.5-4B text backbone and the VibeVoice acoustic tokenizer as codec.

Container building and testing: jsaigou. Engine: Radiance by Deadcode (Apache-2.0); this container needs the unofficial fork linked above.
No warranty; use at your own risk.
