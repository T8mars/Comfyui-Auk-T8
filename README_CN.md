<div align="center">

# AuK · T8star-Aix ComfyUI 原生节点

在 ComfyUI 进程内直接运行 AuK 语音生成与编辑

[English](README.md) · [模型仓库](https://huggingface.co/t8star/Auk-Comfy) · [独立本地整合包](https://pan.quark.cn/s/264edb7e36bd)

</div>

这是独立的 ComfyUI V3 节点。节点直接加载 AuK、AuK-Flash 和 Qwen2.5-Omni-3B，不依赖 AuK Local、不连接 `127.0.0.1:7860`，也不需要服务令牌。

本仓库只发布**独立 ComfyUI 节点**；另外只有一个 **AuK Local 本地整合包**。二者可以分别安装和运行，只共用模型来源和文档链接。

## 节点

- **AuK 模型加载器**：默认选择高质量的 AuK Base，也可切换到速度优先的 AuK-Flash；ComfyUI 管理 VAE、Qwen 与 DiT 的分阶段加载和显存释放。
- **AuK 生成 / 编辑**：提供覆盖上游全部 16 个底层任务的 17 个中文任务入口；“指定说话人提取”是说话人分离的内容定位入口。节点输出标准 ComfyUI `AUDIO`、最终指令和运行参数 JSON。

支持描述生成语音、参考声音克隆、语音及歌词编辑、音高/速度/音量/情绪/音色编辑、去口音、非语言声音编辑、耳语转换、语音增强、音质修复、说话人分离、音乐人声提取和指定说话人提取。

## 安装

### ComfyUI Manager

在 ComfyUI Manager 搜索 **AuK · T8star-Aix**，安装后重启 ComfyUI。请确认 Manager 提供的是 2.0.6 或更高版本；如果 Registry 尚在处理、仍显示旧版本，请先使用 Git 安装。

### Git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Auk-T8
cd Comfyui-Auk-T8
python -m pip install -r requirements.txt
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

下载器固定到 `MODEL_MANIFEST.json` 记录的已测试 Hugging Face 快照，并默认校验 SHA-256；只有明确需要更快的仅尺寸检查时才使用 `--skip-sha256`。加载器也会搜索 `extra_model_paths.yaml` 中注册为 `auk` 的全部路径；AuK 权重与 Qwen 目录可以放在不同的已注册根目录中。

Flash + Qwen 约需 18.7 GB，两种 AuK 模型与 Qwen 全部下载约需 25.5 GB。

## 使用

1. 把 `example_workflows` 中对应任务的 JSON 直接拖进 ComfyUI；目录内提供全部 17 个入口的可执行工作流。
2. **AuK 模型加载器**默认使用 AuK Base。情绪、口音、音色、非语言、耳语及修复任务请优先使用 Base；Flash 适合快速预览。
3. 在 **AuK 生成 / 编辑**选择任务；节点会动态显示该任务的官方输入要求、填写字段、示例和注意事项。需要参考音频的任务请连接 ComfyUI `Load Audio`。
4. 运行工作流。AuK-Flash 自动固定为 NFE=4、CFG=0；Base 使用高级参数。

描述生成与声音克隆默认使用 **TTS 自动估时**：按目标文本估算说话时长，避免短句设置了过长目标时长后继续读出 AuK 的无参考音频内部标记。需要精确时长时可切到“手动指定”。Seed 使用 ComfyUI 标准的“生成后随机化”模式，适合连续抽卡；需要复现结果时把 Seed 控制模式改为 fixed。运行参数 JSON 会保存实际 Seed、界面请求时长、最终采用时长和时长模式。

音高、音量、音色、去口音、耳语、增强、音质修复和分离任务与实际输入音频等长；速度编辑按“输入时长 ÷ 速度倍率”计算；情绪编辑使用官方系数（悲伤 ×1.22、恐惧 ×1.16、其他支持情绪 ×1.06）；文字与歌词编辑按本次文字增删比例估算；添加或删除非语言声音时会按官方事件时长增减。这些自动任务会忽略界面中残留的“目标时长”数值，因此把 48 秒素材剪成 4 秒后会以节点收到的 4 秒音频为准。

节点在推理前使用官方 Silero VAD 检测未加缓冲的真实语音区间，并只为模型输入补 0.1 秒边缘；耳语按官方 -44.47 LUFS 标定，歌词和音乐分离输出按 -14 LUFS/0.95 峰值保护，防止后半段爆音。歌词编辑的输入必须是无伴奏、隔离干净的独唱（a cappella）。情绪、去口音和耳语需要普通说话素材；把歌声用于这些任务，或把标准普通话用于去口音，通常不会出现可听变化。模型加载及生成通过 ComfyUI 原生进度条显示配置、Qwen、VAE、AuK、文本/参考音频编码、采样和解码阶段。速度倍率只支持 `0.5`、`0.75`、`1.25`、`1.5`、`2.0`；音高使用 `+1/+2/+3` 或 `-1/-2/-3` 半音，音量使用 `+5/+10/+15` 或 `-5/-10/-15` 分贝。

输入/参考音频与生成目标分别最多 30 秒，二者不相加计算；30 秒输入可以生成 30 秒输出。CPU 模式可用于兼容测试，但速度很慢，推荐 NVIDIA CUDA 与 bf16。

> 2.0.6 修复裁剪后仍按旧时长校验的问题，并把输入与输出改为各自独立的 30 秒上限；补全官方时长策略和任务预处理，新增音质修复，并默认使用 AuK Base。发布包包含全部 17 个任务入口的可拖入工作流。

> 2.0.6 使用官方 Silero 未加缓冲语音时长和 LUFS 规则；严格规范非语言声、音质修复与说话人顺序输入；选择任务时直接显示对应的官方用法。

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
