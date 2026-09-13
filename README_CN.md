<div align="center">

# AuK Local · T8star-Aix ComfyUI 节点

AuK 语音生成与编辑的 ComfyUI V3 桥接节点

[English](README.md) · [模型仓库](https://huggingface.co/t8star/Auk-Comfy) · [一键整合包](https://pan.quark.cn/s/264edb7e36bd)

</div>

这是 AuK Local 单一整合包的 ComfyUI 节点仓库。节点通过 `http://127.0.0.1:7860` 调用独立的 AuK Local 服务，AuK、Qwen 和专用 Python 依赖不会加载到 ComfyUI 进程中，避免依赖冲突和重复占用显存。

## 功能

- 两个 ComfyUI V3 节点：**AuK Local 连接**、**AuK Local 生成 / 编辑**。
- 覆盖 16 类任务：描述生成语音、参考声音克隆、语音文字编辑、歌词编辑、音高/速度/音量/情绪/音色编辑、去口音、非语言声音编辑、耳语转换、语音增强、说话人分离、音乐人声提取和指定说话人提取。
- 输出标准 ComfyUI `AUDIO`、最终指令和完整运行参数 JSON。
- 支持 AuK-Flash / AuK Base、固定 Seed、CPU Offload、取消、断线恢复和“输入时长 + 输出时长不超过 30 秒”校验。
- `example_workflows` 内附三份可直接加载的工作流。

## 安装

### ComfyUI Manager

在 ComfyUI Manager 中搜索 **AuK Local · T8star-Aix**，安装后重启 ComfyUI。

### Git 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Auk-T8
```

桥接节点没有额外 pip 依赖，PyTorch 与 Torchaudio 由 ComfyUI 提供。

## 使用

1. 下载唯一的 [AuK Local + ComfyUI 一键整合包](https://pan.quark.cn/s/264edb7e36bd)。
2. 在整合包根目录双击 `启动AuK服务.cmd`，保持服务窗口运行。
3. 在 ComfyUI 中加载 `example_workflows` 里的任一工作流。
4. `AuK Local 连接` 的 `service_url` 默认保持 `http://127.0.0.1:7860`。如果节点来自 Manager，请在高级输入 `token_file` 中填写整合包内 `data/session-token` 的绝对路径；整合包自己的安装脚本会自动配置该路径。
5. 选择任务并运行。声音克隆或编辑示例中的 `Load Audio` 文件名只是占位，请换成自己的音频。

服务只监听本机回环地址。工作流只保存令牌文件路径，不保存令牌内容。

## 模型

本版本使用的完整模型镜像位于 [t8star/Auk-Comfy](https://huggingface.co/t8star/Auk-Comfy)：

- `AuK-Flash`：四步快速生成，整合包默认模型。
- `AuK`：高质量语音生成和编辑基础模型。
- `Qwen2.5-Omni-3B`：本地服务使用的指令理解模型。

Hugging Face 模型卡会反向链接本 GitHub 仓库，并记录上游仓库、固定 revision、文件大小和 SHA-256。

## 示例工作流

- `AuK-01-描述生成语音.json`：无需参考音频，按声音描述生成语音。
- `AuK-02-参考声音克隆.json`：零样本声音克隆。
- `AuK-03-语音文字编辑.json`：保留原声音色并修改说话内容。

## 兼容环境

- ComfyUI `>=0.3.48`，使用 V3 自定义节点 API。
- 发布的一键整合包支持 Windows 10/11 x64。
- 已验证 Python 3.10、PyTorch/Torchaudio 2.7.1 + CUDA 12.8、NVIDIA RTX 5090 Laptop 24 GB。
- 输出音频为 24 kHz float WAV。

## 社媒与资源

- [B站](https://space.bilibili.com/385085361)
- [YouTube](https://www.youtube.com/@T8star-Aix/)
- [API](https://api.seedance.nz/sign-up?aff=5f4w)
- [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121)
- [ComfyUI 一键整合包](https://pan.quark.cn/s/264edb7e36bd)
- [AuK-Comfy 模型仓库](https://huggingface.co/t8star/Auk-Comfy)
- [Hugging Face 主页](https://huggingface.co/t8star)
- [AuK 官方项目](https://github.com/Tencent-Hunyuan/AuK)

## 许可

节点代码采用 [MIT License](LICENSE)。Hugging Face 仓库中的模型文件保留原作者随模型提供的许可文件。

