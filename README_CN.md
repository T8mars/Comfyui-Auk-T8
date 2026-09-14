<div align="center">

# AuK · T8star-Aix ComfyUI 原生节点

在 ComfyUI 进程内直接运行 AuK 语音生成与编辑

[English](README.md) · [模型仓库](https://huggingface.co/t8star/Auk-Comfy) · [独立本地整合包](https://pan.quark.cn/s/264edb7e36bd)

</div>

这是独立的 ComfyUI V3 节点。节点直接加载 AuK、AuK-Flash 和 Qwen2.5-Omni-3B，不依赖 AuK Local、不连接 `127.0.0.1:7860`，也不需要服务令牌。

本仓库只发布**独立 ComfyUI 节点**；另外只有一个 **AuK Local 本地整合包**。二者可以分别安装和运行，只共用模型来源和文档链接。

## 节点

- **AuK 模型加载器**：选择 AuK-Flash 或 AuK Base，由 ComfyUI 管理 VAE、Qwen 与 DiT 的分阶段加载和显存释放。
- **AuK 生成 / 编辑**：提供 16 类本地任务模板，输出标准 ComfyUI `AUDIO`、最终指令和运行参数 JSON。

支持描述生成语音、参考声音克隆、语音及歌词编辑、音高/速度/音量/情绪/音色编辑、去口音、非语言声音编辑、耳语转换、语音增强、说话人分离、音乐人声提取和指定说话人提取。

## 安装

### ComfyUI Manager

在 ComfyUI Manager 搜索 **AuK · T8star-Aix**，安装后重启 ComfyUI。

### Git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Auk-T8
cd Comfyui-Auk-T8
pip install -r requirements.txt
```

必须使用运行 ComfyUI 的同一个 Python 安装依赖。`requirements.txt` 不会安装或更换 PyTorch、TorchAudio。

## 模型

从 [t8star/Auk-Comfy](https://huggingface.co/t8star/Auk-Comfy) 下载模型，并保持下面的目录结构：

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
    └── 其余仓库文件
```

也可以使用 ComfyUI 的 Python 在节点目录下载。Flash 与 Base 都需要 Qwen：

```bash
python download_models.py --variant flash
python download_models.py --variant base
python download_models.py --variant all
```

Flash + Qwen 约需 18.7 GB，两种 AuK 模型与 Qwen 全部下载约需 25.5 GB。

## 使用

1. 加载 `example_workflows` 中的工作流。
2. 在 **AuK 模型加载器**选择 AuK-Flash 或 AuK Base。
3. 在 **AuK 生成 / 编辑**选择任务并填写内容；需要参考音频的任务请连接 ComfyUI `Load Audio`。
4. 运行工作流。AuK-Flash 自动固定为 NFE=4、CFG=0；Base 使用高级参数。

输入/参考音频与生成目标共用 30 秒序列上限。CPU 模式可用于兼容测试，但速度很慢，推荐 NVIDIA CUDA 与 bf16。

> 2.0.0 从旧版 HTTP 桥接架构改为原生节点。旧工作流中的 `AuKLocalConnection` 和 `AuKLocalGenerateEdit` 已移除，请改用随 2.0.0 提供的新工作流。

## 独立本地整合包

[AuK Local 一键整合包](https://pan.quark.cn/s/264edb7e36bd)继续提供独立的浅色网页工作台，自带 Python、模型管理、任务记录和启动脚本。它不再是本 ComfyUI 节点的运行前置条件。

## 兼容环境

- ComfyUI `>=0.23.0`，需要 V3 自定义节点 API 和分阶段模型管理接口。
- Python `>=3.10`。
- 已验证 PyTorch/TorchAudio 2.7.x + CUDA 12.8 和 24 GB NVIDIA 显卡。
- 首次构建模型会占用较多内存，建议 48 GB 以上系统内存。
- 输出为 24 kHz float 音频。

## 社媒与资源

- [B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [独立本地整合包](https://pan.quark.cn/s/264edb7e36bd)
- [AuK-Comfy 模型仓库](https://huggingface.co/t8star/Auk-Comfy)
- [Hugging Face 主页](https://huggingface.co/t8star)
- [AuK 官方项目](https://github.com/Tencent-Hunyuan/AuK)

## 许可

节点代码采用 [MIT License](LICENSE)。模型文件保留各上游仓库随模型提供的许可文件。
