<div align="center">

# AuK · T8star-Aix Native ComfyUI Nodes

Run AuK speech generation and editing directly inside ComfyUI

[中文说明](README_CN.md) · [Model repository](https://huggingface.co/t8star/Auk-Comfy) · [Standalone local package](https://pan.quark.cn/s/264edb7e36bd)

</div>

This is a standalone ComfyUI V3 custom-node package. It loads AuK, AuK-Flash, and Qwen2.5-Omni-3B directly in the ComfyUI process. It does not require AuK Local, a server at `127.0.0.1:7860`, or a service token.

This repository publishes only the **standalone ComfyUI node package**. There is one separate **AuK Local one-click package**. Each can be installed and run independently; they only share model sources and documentation links.

## Nodes

- **AuK Model Loader** defaults to the higher-quality AuK Base and can switch to the speed-oriented AuK-Flash. ComfyUI manages staged loading and offloading of the VAE, Qwen encoder, and DiT.
- **AuK Generate / Edit** exposes 17 Chinese task entries that cover all 16 upstream low-level tasks. Target-speaker extraction is a content-based entry for speaker separation. The node returns standard ComfyUI `AUDIO`, the final instruction, and run metadata JSON.

Tasks include instruction TTS, zero-shot voice cloning, speech and lyric editing, pitch/speed/volume/emotion/timbre editing, de-accenting, nonverbal editing, whisper conversion, enhancement, quality repair, speaker separation, vocal extraction, and target-speaker extraction.

## Install

### ComfyUI Manager

Search for **AuK · T8star-Aix** in ComfyUI Manager, install it, and restart ComfyUI. Confirm that Manager offers version 2.0.6 or later; if Registry processing still shows an older release, use the Git installation until the new version becomes active.

### Git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Auk-T8
cd Comfyui-Auk-T8
python -m pip install -r requirements.txt
```

Install dependencies with the same Python interpreter that runs ComfyUI. The requirements do not install or replace PyTorch or TorchAudio.

## Models

Download the model files from [t8star/Auk-Comfy](https://huggingface.co/t8star/Auk-Comfy) and keep this layout:

```text
ComfyUI/models/auk/
├── AuK-Flash/
│   ├── auk_flash.safetensors
│   ├── vae.safetensors
│   └── config.yaml
├── AuK/
│   ├── auk_base.safetensors
│   ├── vae.safetensors
│   └── config.yaml
└── Qwen2.5-Omni-3B/
    ├── config.json
    ├── model-00001-of-00003.safetensors
    ├── model-00002-of-00003.safetensors
    ├── model-00003-of-00003.safetensors
    └── the remaining repository files
```

You can also run the downloader with ComfyUI's Python from the node directory. Both model variants require Qwen:

```bash
python download_models.py --variant flash
python download_models.py --variant base
python download_models.py --variant all
```

The downloader is pinned to the tested Hugging Face snapshot recorded in `MODEL_MANIFEST.json` and verifies SHA-256 by default; use `--skip-sha256` only when you intentionally want a faster size-only check. The loader also searches every path registered as `auk` in `extra_model_paths.yaml`; the AuK checkpoint and Qwen folder may be stored in different registered roots.

Flash plus Qwen requires about 18.7 GB. Both AuK variants plus Qwen require about 25.5 GB.

## Run

1. Drag the JSON for the required task from `example_workflows` into ComfyUI. The folder contains an executable workflow for every one of the 17 entries.
2. **AuK Model Loader** defaults to AuK Base. Prefer Base for emotion, accent, timbre, nonverbal, whisper, and repair tasks; Flash is intended for fast previews.
3. Select a task in **AuK Generate / Edit**. The node displays its official input requirement, field meanings, example, and warning. Connect ComfyUI `Load Audio` for tasks that require source or reference audio.
4. Queue the workflow. AuK-Flash always uses NFE=4 and CFG=0; Base uses the advanced sampling controls.

Instruction TTS and voice cloning default to **automatic TTS duration**. This estimates the spoken length from the target text and prevents a short sentence from continuing into AuK's internal no-reference marker when a much longer duration is requested. Select **manual duration** when exact timing is required. The Seed widget uses ComfyUI's standard **randomize after generation** mode by default; switch its control mode to fixed to reproduce a result. The metadata output records the actual seed, requested duration, resolved duration, and duration mode.

Pitch, volume, timbre, de-accent, whisper, enhancement, quality repair, and separation match the audio actually received by the node. Speed uses `input duration / multiplier`. Emotion uses the official factors: 1.22× for sad, 1.16× for fearful, and 1.06× for the other supported emotions. Speech and lyric edits estimate the result from the text added or removed. Nonverbal edits add or remove the official event duration. These automatic rules ignore a stale target-duration widget, so a 48-second source trimmed to four seconds is validated as the actual four-second node input.

Before inference, the node uses official Silero VAD to measure the unpadded speech interval and adds 0.1 seconds only around the model input. Whisper conversion uses the official -44.47 LUFS target; lyric and music-separation outputs use the official -14 LUFS downward limiter and 0.95 peak ceiling. Lyric editing requires clean isolated a cappella solo vocals. Emotion, de-accenting, and whisper conversion require ordinary spoken speech. Singing is not a valid test source for those tasks, and already-standard speech is not a valid de-accenting test. Model loading and generation report native ComfyUI progress through configuration, Qwen, VAE, AuK, text/reference encoding, sampling, and decoding stages. Speed supports `0.5`, `0.75`, `1.25`, `1.5`, or `2.0`; pitch uses `+1/+2/+3` or `-1/-2/-3` semitones, and volume uses `+5/+10/+15` or `-5/-10/-15` dB.

Source/reference audio and generated output are each limited to 30 seconds independently. They are not added together, so a 30-second input may produce a 30-second output. CPU mode is available for compatibility testing but is very slow; NVIDIA CUDA with bf16 is recommended.

> Version 2.0.6 fixes validation against stale pre-trim durations and applies independent 30-second limits to input and output. It completes the official duration and preprocessing rules, adds quality repair, and defaults to AuK Base. The package includes drag-and-drop workflows for all 17 task entries.

> Version 2.0.6 uses official unpadded Silero speech duration and LUFS handling, validates nonverbal/quality/speaker-order requests before inference, and displays the matching official guide inside the node.

## Standalone local package

The [AuK Local one-click package](https://pan.quark.cn/s/264edb7e36bd) remains a separate light-themed web workstation with its own Python runtime, model management, task history, and launch scripts. It is no longer a runtime prerequisite for these ComfyUI nodes.

## Compatibility

- ComfyUI `>=0.23.0` with the V3 custom-node API and staged model-management interfaces.
- Python `>=3.10`.
- Verified with PyTorch/TorchAudio 2.7.x + CUDA 12.8 and a 24 GB NVIDIA GPU.
- Model construction uses substantial host memory; 48 GB or more system RAM is recommended.
- Output is 24 kHz float audio.

## Links

- [Bilibili](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [Standalone local package](https://pan.quark.cn/s/264edb7e36bd)
- [AuK-Comfy models](https://huggingface.co/t8star/Auk-Comfy)
- [Hugging Face profile](https://huggingface.co/t8star)
- [Original AuK project](https://github.com/Tencent-Hunyuan/AuK)

## License

The node code is released under the [MIT License](LICENSE). Model files retain the licenses included by their upstream repositories.
