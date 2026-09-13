<div align="center">

# AuK Local · T8star-Aix for ComfyUI

ComfyUI V3 bridge nodes for AuK speech generation and editing

[中文说明](README_CN.md) · [Model weights](https://huggingface.co/t8star/Auk-Comfy) · [One-click package](https://pan.quark.cn/s/264edb7e36bd)

</div>

This repository contains the ComfyUI side of the AuK Local integration package. The nodes call the isolated AuK Local service at `http://127.0.0.1:7860`, so AuK, Qwen, and their Python dependencies stay outside the ComfyUI process.

## Features

- Two ComfyUI V3 nodes: **AuK Local Connection** and **AuK Local Generate / Edit**.
- 16 tasks covering instruction TTS, zero-shot voice cloning, speech and lyric editing, pitch/speed/volume/emotion/timbre editing, de-accenting, nonverbal and whisper conversion, enhancement, and source separation.
- Standard ComfyUI `AUDIO` output plus the final instruction and run metadata JSON.
- AuK-Flash and AuK Base selection, deterministic seed, CPU offload, cancellation, retry recovery, and a 30-second input-plus-output guard.
- Three ready-to-load workflows in [`example_workflows`](example_workflows).

## Install

### ComfyUI Manager

Search for **AuK Local · T8star-Aix** in ComfyUI Manager and install it, then restart ComfyUI.

### Git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Auk-T8
```

This bridge has no extra pip dependencies. ComfyUI supplies PyTorch and Torchaudio.

## Run

1. Download the single [AuK Local + ComfyUI integration package](https://pan.quark.cn/s/264edb7e36bd).
2. Start `启动AuK服务.cmd` in the package and keep its window open.
3. In ComfyUI, load one of the workflows in `example_workflows`.
4. In **AuK Local Connection**, leave `service_url` as `http://127.0.0.1:7860`. If the node was installed by Manager, set the advanced `token_file` field to the package's absolute `data/session-token` path. The package installer configures this path automatically.
5. Choose the task and run the workflow. Use your own audio in `Load Audio` for cloning or editing examples.

The service listens on loopback only. The workflow stores the token file path, never the token itself.

## Models

The exact model mirror used by this release is hosted at [t8star/Auk-Comfy](https://huggingface.co/t8star/Auk-Comfy):

- `AuK-Flash` — fast four-step generation.
- `AuK` — base model for higher-quality generation and editing.
- `Qwen2.5-Omni-3B` — prompt understanding used by the local service.

The Hugging Face model card links back to this ComfyUI repository and records the upstream repositories, revisions, file sizes, and SHA-256 hashes.

## Example workflows

- `AuK-01-描述生成语音.json` — instruction TTS without reference audio.
- `AuK-02-参考声音克隆.json` — zero-shot voice cloning.
- `AuK-03-语音文字编辑.json` — edit spoken content while keeping the source voice.

## Compatibility

- ComfyUI `>=0.3.48` with the V3 custom-node API.
- Windows 10/11 x64 for the published integration package.
- Verified with Python 3.10, PyTorch/Torchaudio 2.7.1 + CUDA 12.8, and an NVIDIA RTX 5090 Laptop GPU with 24 GB VRAM.
- Audio output: 24 kHz float WAV.

## Links

- [Bilibili](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI integration package](https://pan.quark.cn/s/264edb7e36bd)
- [Hugging Face models](https://huggingface.co/t8star/Auk-Comfy)
- [Hugging Face profile](https://huggingface.co/t8star)
- [Original AuK project](https://github.com/Tencent-Hunyuan/AuK)

## License

The node code is released under the [MIT License](LICENSE). Model files in the Hugging Face repository retain the license files supplied by their original authors.
