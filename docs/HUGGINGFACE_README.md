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

# AuK models and complete Windows package / 模型与 Windows 完整整合包

By B站 T8star-Aix

## Complete Windows package / Windows 完整版下载

**[Download AuK-Local.rar / 下载完整整合包（海外直链）](https://huggingface.co/t8star/Auk-Comfy/resolve/main/AuK-Local.rar?download=true)**

- Size / 大小: **20,372,471,916 bytes — 20.37 GB (18.97 GiB)**.
- Includes / 包含: AuK Local, `AuK-Local.exe`, bundled Python, AuK Base, AuK-Flash, and Qwen2.5-Omni-3B models.
- Fully extract the RAR and double-click `AuK-Local.exe` inside the `AuK-Local` folder. Keep the console open; the browser opens when the service is ready.
- 完整解压 RAR，双击 `AuK-Local` 文件夹内的 `AuK-Local.exe`；保持控制台开启，服务就绪后自动打开网页。
- Resume interrupted downloads with a download manager / 下载中断可使用支持断点续传的下载工具继续。
- GitHub [AuK Local Releases](https://github.com/T8mars/AuK-Local/releases) provide **code-only updates**, without models or Python / GitHub Release 是**程序更新包**，不含模型和 Python。

SHA-256: `a99966c3eb7336ca9d027c0ea491edaddcfa3582b2eb436444a36cfc2f2e9061`

The RAR is the standalone Windows app. For native ComfyUI nodes, install the GitHub node repository and use the model folders below / RAR 为独立 Windows 工作台；使用原生 ComfyUI 节点时，安装下方 GitHub 节点仓库并使用三个模型目录。

This repository contains the models used by the independent [AuK · T8star-Aix native ComfyUI nodes](https://github.com/T8mars/Comfyui-Auk-T8) and by the separate [AuK Local source and releases](https://github.com/T8mars/AuK-Local). The complete local package is available through the Hugging Face download link above, with [Quark Drive](https://pan.quark.cn/s/264edb7e36bd) as an additional download channel.

本仓库保存独立的 [AuK · T8star-Aix ComfyUI 原生节点](https://github.com/T8mars/Comfyui-Auk-T8)与单独的 [AuK Local 源码及 Release](https://github.com/T8mars/AuK-Local)所需模型；完整版整合包可通过上方 Hugging Face 直链下载，也保留[夸克网盘](https://pan.quark.cn/s/264edb7e36bd)入口。两者可以分别安装和运行。

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
- [AuK Local source and releases / AuK Local 源码与更新](https://github.com/T8mars/AuK-Local)
- [Bilibili / B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [Online AI apps / 在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [Standalone local package / 独立本地整合包](https://pan.quark.cn/s/264edb7e36bd)
- [Complete Windows package / Windows 完整版海外下载](https://huggingface.co/t8star/Auk-Comfy/resolve/main/AuK-Local.rar?download=true)
- [Hugging Face profile / Hugging Face 主页](https://huggingface.co/t8star)
