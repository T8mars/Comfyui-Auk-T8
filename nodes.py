from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import folder_paths
import torch
import torchaudio
from comfy import model_management
from comfy.utils import ProgressBar
from comfy_api.v0_0_2 import ComfyExtension, io
from omegaconf import OmegaConf

from .duration import (
    AUTO_DURATION_MODE,
    MANUAL_DURATION_MODE,
    TTS_TASK_KEYS,
    estimate_tts_seconds,
)
from .runtime import (
    MAX_SEQUENCE_SECONDS,
    QWEN_AUDIO_SAMPLE_RATE,
    AuKEngine,
    latent_frames_to_seconds,
    source_aligned_seconds,
    source_latent_frames,
    validate_sequence_duration,
)
from .task_templates import (
    TASK_BY_LABEL,
    TASKS,
    build_instruction,
    parse_speed_multiplier,
)

logger = logging.getLogger("ComfyUI-AuK-T8")
AUK_ENGINE = io.Custom("AUK_ENGINE")
MODEL_ROOT = Path(folder_paths.models_dir) / "auk"
folder_paths.add_model_folder_path("auk", str(MODEL_ROOT), is_default=True)

MODEL_VARIANTS = {
    "AuK-Flash": ("AuK-Flash", "auk_flash.safetensors"),
    "AuK Base": ("AuK", "auk_base.safetensors"),
}


def load_manifest() -> dict[str, Any]:
    return json.loads(Path(__file__).with_name("MODEL_MANIFEST.json").read_text(encoding="utf-8"))


def model_search_roots() -> list[Path]:
    """Return ComfyUI's default and extra configured AuK model roots."""
    candidates = [MODEL_ROOT]
    try:
        candidates.extend(Path(path) for path in folder_paths.get_folder_paths("auk"))
    except KeyError:
        pass

    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.expanduser().resolve()).casefold()
        if normalized not in seen:
            roots.append(candidate.expanduser().resolve())
            seen.add(normalized)
    return roots


def resolve_model_directory(directory_name: str, manifest: dict[str, Any]) -> Path:
    problems: list[str] = []
    for root in model_search_roots():
        directory = root / directory_name
        directory_problems: list[str] = []
        for filename, details in manifest[directory_name]["files"].items():
            path = directory / filename
            if not path.is_file():
                directory_problems.append(f"缺少 {path}")
            elif path.stat().st_size != int(details["size"]):
                directory_problems.append(f"大小不符 {path}")
        if not directory_problems:
            return directory
        problems.extend(directory_problems)

    preview = "；".join(problems[:5])
    if len(problems) > 5:
        preview += f"；另有 {len(problems) - 5} 个文件"
    searched = "、".join(str(root) for root in model_search_roots())
    raise FileNotFoundError(f"AuK 模型目录 {directory_name} 不完整：{preview}。已搜索：{searched}")


def resolve_model_files(model_variant: str) -> tuple[Path, Path, Path]:
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"未知模型：{model_variant}")
    model_directory, checkpoint_name = MODEL_VARIANTS[model_variant]
    qwen_directory = "Qwen2.5-Omni-3B"
    manifest = load_manifest()["models"]
    try:
        model_path = resolve_model_directory(model_directory, manifest)
        qwen_path = resolve_model_directory(qwen_directory, manifest)
    except FileNotFoundError as exc:
        download_variant = "flash" if model_directory == "AuK-Flash" else "base"
        raise FileNotFoundError(
            f"{exc}。请把 Hugging Face t8star/Auk-Comfy 中的目录放到 ComfyUI/models/auk，"
            f"或在节点目录运行 python download_models.py --variant {download_variant}。"
        ) from exc
    checkpoint = model_path / checkpoint_name
    config = model_path / "config.yaml"
    return checkpoint, config, qwen_path


def resolve_device(setting: str) -> torch.device:
    if setting == "auto":
        device = model_management.get_torch_device()
    else:
        device = torch.device(setting)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"AuK 当前只支持 CUDA 或 CPU，ComfyUI 当前设备是 {device}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("选择了 CUDA，但当前 PyTorch 无法使用 CUDA")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA 设备不存在：cuda:{index}")
        return torch.device("cuda", index)
    return torch.device("cpu")


def resolve_dtype(setting: str, device: torch.device) -> str:
    supports_bf16 = False
    if device.type == "cuda":
        with torch.cuda.device(device):
            supports_bf16 = torch.cuda.is_bf16_supported()
    if setting == "auto":
        if device.type == "cpu":
            return "fp32"
        return "bf16" if supports_bf16 else "fp16"
    if setting not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"不支持的数据类型：{setting}")
    if device.type == "cpu" and setting != "fp32":
        raise ValueError("CPU 推理必须使用 fp32")
    if device.type == "cuda" and setting == "bf16" and not supports_bf16:
        raise ValueError("当前 CUDA 设备不支持 bf16，请选择 fp16")
    return setting


def normalize_audio(audio: dict[str, Any] | None) -> tuple[torch.Tensor, int] | None:
    if audio is None:
        return None
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("input_audio 必须是 ComfyUI AUDIO")
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        shape = tuple(waveform.shape) if torch.is_tensor(waveform) else type(waveform).__name__
        raise ValueError(f"input_audio waveform 必须是 [B, C, T]，当前为 {shape}")
    if waveform.shape[0] != 1:
        raise ValueError(f"AuK 每次只接受一段音频，当前 batch={waveform.shape[0]}")
    if waveform.shape[1] < 1 or waveform.shape[2] < 1:
        raise ValueError("输入音频为空")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"输入采样率无效：{sample_rate!r}")
    waveform = waveform[0].detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(waveform).all():
        raise ValueError("输入音频包含 NaN 或 Inf")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.contiguous(), sample_rate


def resolve_generation_seconds(
    engine: AuKEngine,
    duration_strategy: str,
    audio: tuple[torch.Tensor, int] | None,
    requested_seconds: float,
    primary: str,
    task_key: str | None = None,
    duration_mode: str = MANUAL_DURATION_MODE,
) -> float:
    """Resolve the output duration while preserving source length for fixed-length edits."""
    if duration_strategy == "source":
        if audio is None:
            raise ValueError("等长任务需要输入音频")
        return source_aligned_seconds(engine, audio)
    if duration_strategy == "speed":
        if audio is None:
            raise ValueError("速度编辑需要输入音频")
        target_frames = math.ceil(source_latent_frames(engine, audio) / parse_speed_multiplier(primary))
        return latent_frames_to_seconds(engine, target_frames)
    if task_key in TTS_TASK_KEYS and duration_mode == AUTO_DURATION_MODE:
        return estimate_tts_seconds(primary, max_seconds=MAX_SEQUENCE_SECONDS)
    return float(requested_seconds)


class AuKModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        devices = ["auto"] + [f"cuda:{index}" for index in range(torch.cuda.device_count())] + ["cpu"]
        return io.Schema(
            node_id="AuKModelLoader",
            display_name="AuK 模型加载器",
            category="AuK · T8star-Aix",
            description="直接在 ComfyUI 内加载 AuK，不需要启动 7860 服务。模型由 ComfyUI 分阶段管理显存。",
            inputs=[
                io.Combo.Input("model_variant", display_name="模型", options=list(MODEL_VARIANTS), default="AuK-Flash"),
                io.Combo.Input("device", display_name="设备", options=devices, default="auto", advanced=True),
                io.Combo.Input(
                    "dtype",
                    display_name="精度",
                    options=["auto", "bf16", "fp16", "fp32"],
                    default="auto",
                    advanced=True,
                ),
            ],
            outputs=[AUK_ENGINE.Output("engine", display_name="AuK 模型")],
        )

    @classmethod
    def execute(cls, model_variant: str, device: str = "auto", dtype: str = "auto") -> io.NodeOutput:
        checkpoint, config, qwen = resolve_model_files(model_variant)
        model_config = OmegaConf.load(config)
        expected_name = MODEL_VARIANTS[model_variant][0]
        if str(model_config.model.get("name", "")) != expected_name:
            raise ValueError(f"配置文件模型类型错误：{config}")
        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(dtype, resolved_device)
        logger.info("Loading %s on %s (%s)", model_variant, resolved_device, resolved_dtype)
        progress = ProgressBar(4)
        load_steps = {"config": 1, "qwen": 2, "vae": 3, "auk": 4}

        def load_progress(phase: str) -> None:
            model_management.throw_exception_if_processing_interrupted()
            progress.update_absolute(load_steps[phase])

        try:
            engine = AuKEngine(checkpoint, config, qwen, resolved_device, resolved_dtype, load_progress)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"缺少 AuK 运行依赖 {exc.name!r}；请用 ComfyUI 的 Python 执行 pip install -r requirements.txt"
            ) from exc
        engine.model_variant = model_variant
        manifest = load_manifest()["models"]
        engine.model_revision = manifest[expected_name]["revision"]
        engine.qwen_revision = manifest["Qwen2.5-Omni-3B"]["revision"]
        return io.NodeOutput(engine)


class AuKGenerateEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AuKGenerateEdit",
            display_name="AuK 生成 / 编辑",
            category="AuK · T8star-Aix",
            description="在 ComfyUI 进程内执行 AuK 的 16 类任务；等长与变速编辑会自动计算输出时长。",
            inputs=[
                AUK_ENGINE.Input("engine", display_name="AuK 模型"),
                io.Combo.Input("task", display_name="任务", options=[task.label for task in TASKS], default=TASKS[0].label),
                io.String.Input(
                    "primary",
                    display_name="主要内容（变速填倍率；音高/音量填带符号数值）",
                    multiline=True,
                    default="你好，欢迎使用 AuK。",
                ),
                io.String.Input(
                    "secondary",
                    display_name="声音描述 / 附加要求",
                    multiline=True,
                    default="",
                ),
                io.Float.Input(
                    "generation_seconds",
                    display_name="目标时长（秒；编辑任务可自动）",
                    default=3.0,
                    min=0.2,
                    max=MAX_SEQUENCE_SECONDS,
                    step=0.1,
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0x7FFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.randomize,
                ),
                io.Audio.Input("input_audio", display_name="输入 / 参考音频", optional=True),
                io.Int.Input("nfe_steps", display_name="NFE 步数", default=32, min=4, max=64, advanced=True),
                io.Float.Input(
                    "cfg_strength",
                    display_name="CFG 强度",
                    default=2.0,
                    min=0.0,
                    max=5.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Float.Input(
                    "sway_sampling_coef",
                    display_name="Sway 系数",
                    default=-1.0,
                    min=-1.0,
                    max=1.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Combo.Input(
                    "duration_mode",
                    display_name="TTS 时长模式",
                    options=[AUTO_DURATION_MODE, MANUAL_DURATION_MODE],
                    default=AUTO_DURATION_MODE,
                    tooltip="自动估算可减少短文本因目标时长过长而在结尾读出内部提示词；非 TTS 任务仍按任务规则处理。",
                ),
            ],
            outputs=[
                io.Audio.Output("generated_audio", display_name="生成音频"),
                io.String.Output("instruction", display_name="最终指令"),
                io.String.Output("metadata", display_name="运行参数 JSON"),
            ],
        )

    @classmethod
    def execute(
        cls,
        engine: AuKEngine,
        task: str,
        primary: str,
        secondary: str,
        generation_seconds: float,
        seed: int,
        input_audio: dict[str, Any] | None = None,
        nfe_steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        duration_mode: str = AUTO_DURATION_MODE,
    ) -> io.NodeOutput:
        if task not in TASK_BY_LABEL:
            raise ValueError(f"未知任务：{task}")
        template = TASK_BY_LABEL[task]
        audio = normalize_audio(None if not template.needs_audio else input_audio)
        if template.needs_audio and audio is None:
            raise ValueError(f"“{task}”需要连接输入或参考音频")
        instruction = build_instruction(template.key, primary, secondary)
        requested_seconds = float(generation_seconds)
        validate_sequence_duration(engine, None, requested_seconds)
        target_seconds = resolve_generation_seconds(
            engine,
            template.duration_strategy,
            audio,
            requested_seconds,
            primary,
            template.key,
            duration_mode,
        )
        validate_sequence_duration(engine, audio, target_seconds)
        if engine.is_flash:
            nfe_steps, cfg_strength, sway_sampling_coef = 4, 0.0, -1.0

        model_instruction = instruction if audio is not None else instruction + "|<no_prompt_audio>|"
        content: list[dict[str, Any]] = [{"type": "text", "text": model_instruction}]
        if audio is not None:
            waveform, sample_rate = audio
            qwen_waveform = waveform
            if sample_rate != QWEN_AUDIO_SAMPLE_RATE:
                qwen_waveform = torchaudio.functional.resample(waveform, sample_rate, QWEN_AUDIO_SAMPLE_RATE)
            content.append({"type": "audio", "audio": qwen_waveform.squeeze(0).contiguous().numpy()})
        messages = [{"role": "user", "content": content}]

        progress = ProgressBar(100)
        progress.update_absolute(1)
        current_phase = "preparing"
        sampling_updates = 0
        phase_progress = {
            "encoding_reference": 12,
            "encoding_instruction": 28,
            "sampling": 42,
            "decoding": 92,
        }

        def phase_callback(phase: str) -> None:
            nonlocal current_phase
            current_phase = phase
            progress.update_absolute(phase_progress.get(phase, progress.current))

        def interrupt_callback() -> None:
            nonlocal sampling_updates
            model_management.throw_exception_if_processing_interrupted()
            if current_phase == "sampling":
                sampling_updates += 1
                sampling_total = max(1, int(nfe_steps))
                progress.update_absolute(min(90, 42 + math.ceil(48 * sampling_updates / sampling_total)))

        started = time.perf_counter()
        waveform, sample_rate = engine.generate(
            messages,
            audio,
            target_seconds,
            int(nfe_steps),
            float(cfg_strength),
            float(sway_sampling_coef),
            int(seed),
            interrupt_callback,
            phase_callback,
        )
        progress.update_absolute(100)
        effective_duration_strategy = (
            "auto_text" if template.key in TTS_TASK_KEYS and duration_mode == AUTO_DURATION_MODE else template.duration_strategy
        )
        metadata = {
            "model": engine.model_variant,
            "device": str(engine.device),
            "dtype": engine.dtype,
            "task": template.key,
            "seed": int(seed),
            "generation_seconds": target_seconds,
            "requested_generation_seconds": requested_seconds,
            "duration_strategy": effective_duration_strategy,
            "duration_mode": duration_mode,
            "sample_rate": int(sample_rate),
            "actual_output_seconds": waveform.shape[-1] / int(sample_rate),
            "nfe_steps": int(nfe_steps),
            "cfg_strength": float(cfg_strength),
            "sway_sampling_coef": float(sway_sampling_coef),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "runtime": "native_comfyui",
            "model_revision": engine.model_revision,
            "qwen_revision": engine.qwen_revision,
        }
        return io.NodeOutput(
            {"waveform": waveform.unsqueeze(0).cpu(), "sample_rate": int(sample_rate)},
            instruction,
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )


class AuKExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AuKModelLoader, AuKGenerateEdit]


async def comfy_entrypoint() -> AuKExtension:
    return AuKExtension()
