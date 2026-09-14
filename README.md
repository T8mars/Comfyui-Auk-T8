<div align="center">

# AuK · T8star-Aix Native ComfyUI Nodes

Run AuK speech generation and editing directly inside ComfyUI

[中文说明](README_CN.md) · [Model repository](https://huggingface.co/t8star/Auk-Comfy) · [Standalone local package](https://pan.quark.cn/s/264edb7e36bd)

</div>

This is a standalone ComfyUI V3 custom-node package. It loads AuK, AuK-Flash, and Qwen2.5-Omni-3B directly in the ComfyUI process. It does not require AuK Local, a server at `127.0.0.1:7860`, or a service token.

This repository publishes only the **standalone ComfyUI node package**. There is one separate **AuK Local one-click package**. Each can be installed and run independently; they only share model sources and documentation links.

## Nodes

- **AuK Model Loader** selects AuK-Flash or AuK Base. ComfyUI manages staged loading and offloading of the VAE, Qwen encoder, and DiT.
- **AuK Generate / Edit** exposes 16 local task templates and returns standard ComfyUI `AUDIO`, the final instruction, and run metadata JSON.

Tasks include instruction TTS, zero-shot voice cloning, speech and lyric editing, pitch/speed/volume/emotion/timbre editing, de-accenting, nonverbal editing, whisper conversion, enhancement, speaker separation, vocal extraction, and target-speaker extraction.

## Install

### ComfyUI Manager

Search for **AuK · T8star-Aix** in ComfyUI Manager, install it, and restart ComfyUI. Confirm that Manager offers version 2.0.1 or later; if Registry processing still shows an older release, use the Git installation until the new version becomes active.

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

1. Load a workflow from `example_workflows`.
2. Select AuK-Flash or AuK Base in **AuK Model Loader**.
3. Select a task and enter its content in **AuK Generate / Edit**. Connect ComfyUI `Load Audio` for tasks that require source or reference audio.
4. Queue the workflow. AuK-Flash always uses NFE=4 and CFG=0; Base uses the advanced sampling controls.

Source/reference audio and the generated target share a 30-second sequence limit. CPU mode is available for compatibility testing but is very slow; NVIDIA CUDA with bf16 is recommended.

> Version 2.0.1 is the current native release. It replaces the old HTTP bridge and fixes model-path discovery, CPU and multi-GPU model lifecycle, checkpoint validation, and workflow compatibility. Replace workflows containing `AuKLocalConnection` or `AuKLocalGenerateEdit`, and update from 2.0.0.

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
