---
license: other
language:
- zh
- en
tags:
- audio
- text-to-speech
- voice-cloning
- speech-editing
- comfyui
---

# AuK-Comfy model mirror / 模型镜像

This repository contains the models used by the independent [AuK · T8star-Aix native ComfyUI nodes](https://github.com/T8mars/Comfyui-Auk-T8) and by the separate [AuK Local one-click package](https://pan.quark.cn/s/264edb7e36bd).

本仓库保存独立的 [AuK · T8star-Aix ComfyUI 原生节点](https://github.com/T8mars/Comfyui-Auk-T8)与单独的 [AuK Local 一键整合包](https://pan.quark.cn/s/264edb7e36bd)所需模型。两者可以分别安装和运行。

## Contents / 内容

| Folder | Purpose | Upstream source | Pinned revision |
| --- | --- | --- | --- |
| `AuK-Flash` | Four-step speech generation / 四步快速语音生成 | [tencent/AuK-Flash](https://huggingface.co/tencent/AuK-Flash) | `575b92f0895f75180bf2cbd35f2e176c5732b8ed` |
| `AuK` | Base speech generation and editing / 基础语音生成与编辑 | [tencent/AuK](https://huggingface.co/tencent/AuK) | `790742b71a4430120daf2b2099192abae449eb9f` |
| `Qwen2.5-Omni-3B` | Multimodal instruction encoder / 多模态指令编码器 | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | `f75b40e3da2003cdd6e1829b1f420ca70797c34e` |

## Native ComfyUI layout / 原生节点目录

Copy the three folders into `ComfyUI/models/auk`. The native node loads them directly inside ComfyUI and does not require port 7860 or a token.

将三个模型目录复制到 `ComfyUI/models/auk`。原生节点在 ComfyUI 中直接加载，不需要 7860 服务或令牌。

## Links / 社媒与资源

- [GitHub nodes / GitHub 节点](https://github.com/T8mars/Comfyui-Auk-T8)
- [Bilibili / B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [Online AI apps / 在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [Standalone local package / 独立本地整合包](https://pan.quark.cn/s/264edb7e36bd)
- [Hugging Face profile / Hugging Face 主页](https://huggingface.co/t8star)
