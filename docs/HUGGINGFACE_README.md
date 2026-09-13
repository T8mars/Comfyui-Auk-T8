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

This repository contains the model files used by the **AuK Local · T8star-Aix** ComfyUI integration. The custom nodes and bilingual setup guide are at [T8mars/Comfyui-Auk-T8](https://github.com/T8mars/Comfyui-Auk-T8). The single Windows integration package is available [here](https://pan.quark.cn/s/264edb7e36bd).

本仓库保存 **AuK Local · T8star-Aix** ComfyUI 整合包所使用的模型文件。节点代码和中英文教程位于 [T8mars/Comfyui-Auk-T8](https://github.com/T8mars/Comfyui-Auk-T8)，唯一的 Windows 一键整合包在[这里下载](https://pan.quark.cn/s/264edb7e36bd)。

## Contents / 内容

| Folder | Purpose | Upstream source | Pinned revision |
| --- | --- | --- | --- |
| `AuK-Flash` | Four-step speech generation / 四步快速语音生成 | [tencent/AuK-Flash](https://huggingface.co/tencent/AuK-Flash) | `575b92f0895f75180bf2cbd35f2e176c5732b8ed` |
| `AuK` | Base speech generation and editing / 基础语音生成与编辑 | [tencent/AuK](https://huggingface.co/tencent/AuK) | `790742b71a4430120daf2b2099192abae449eb9f` |
| `Qwen2.5-Omni-3B` | Prompt understanding / 指令理解 | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | `f75b40e3da2003cdd6e1829b1f420ca70797c34e` |

`MODEL_MANIFEST.json` records the expected size and SHA-256 of every file required by the integration package. Each model folder includes the upstream license supplied with that model.

`MODEL_MANIFEST.json` 记录整合包所需文件的大小和 SHA-256。每个模型目录均保留上游随模型提供的许可文件。

## Use with ComfyUI / 配合 ComfyUI 使用

Install the node from ComfyUI Manager by searching for **AuK Local · T8star-Aix**, or clone [the GitHub repository](https://github.com/T8mars/Comfyui-Auk-T8). The published node talks to the isolated service included in the integration package; it does not load these models inside the ComfyUI Python process.

在 ComfyUI Manager 中搜索 **AuK Local · T8star-Aix** 安装，或克隆 [GitHub 节点仓库](https://github.com/T8mars/Comfyui-Auk-T8)。该节点连接整合包内的独立服务，不会在 ComfyUI 的 Python 进程里直接加载模型。

## Links / 社媒与资源

- [Bilibili / B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [Online AI apps / 在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI integration package / ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)
- [GitHub nodes / GitHub 节点](https://github.com/T8mars/Comfyui-Auk-T8)
- [This model repository / 本模型仓库](https://huggingface.co/t8star/Auk-Comfy)
- [Hugging Face profile / Hugging Face 主页](https://huggingface.co/t8star)
