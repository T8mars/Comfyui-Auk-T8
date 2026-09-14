from __future__ import annotations

import math
import threading
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torchaudio
from comfy import model_management
from comfy.model_patcher import CoreModelPatcher


MAX_SEQUENCE_SECONDS = 30.0
QWEN_AUDIO_SAMPLE_RATE = 16_000


class _ManagedComponent(torch.nn.Module):
    def __init__(self, component: torch.nn.Module):
        super().__init__()
        self.component = component
        self.device = torch.device("cpu")


class AuKEngine:
    """AuK components managed by ComfyUI and kept on CPU between stages."""

    def __init__(self, checkpoint: Path, config: Path, qwen: Path, device: torch.device, dtype: str):
        from .auk_core.infer.infer_auk import AukInfer

        self.device = device
        self.dtype = dtype
        self.lock = threading.Lock()
        self.inference = AukInfer(
            config_path=str(config),
            ckpt_path=str(checkpoint),
            device=str(device),
            dtype=dtype,
            qwen_path=str(qwen),
            defer_to_cpu=True,
        )
        cpu = torch.device("cpu")
        self.vae_component = _ManagedComponent(self.inference.vae_model)
        self.qwen_component = _ManagedComponent(self.inference.model.text_encoder)
        self.transformer_component = _ManagedComponent(self.inference.model.transformer)
        self.vae_patcher = CoreModelPatcher(self.vae_component, load_device=device, offload_device=cpu)
        self.qwen_patcher = CoreModelPatcher(self.qwen_component, load_device=device, offload_device=cpu)
        self.transformer_patcher = CoreModelPatcher(self.transformer_component, load_device=device, offload_device=cpu)

    @property
    def is_flash(self) -> bool:
        return bool(self.inference.is_flash)

    @property
    def target_sample_rate(self) -> int:
        return int(self.inference.target_sample_rate)

    @property
    def downsample_rate(self) -> int:
        return int(self.inference.downsample_rate)

    def get_models(self) -> list[CoreModelPatcher]:
        return [self.vae_patcher, self.qwen_patcher, self.transformer_patcher]

    def _load(self, patcher: CoreModelPatcher) -> None:
        if patcher.load_device.type == "cpu":
            return
        model_management.load_models_gpu([patcher], force_full_load=True)

    def _autocast(self):
        if self.device.type == "cuda" and self.inference.dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=self.inference.dtype)
        return nullcontext()

    @staticmethod
    def _unload(patcher: CoreModelPatcher) -> None:
        if patcher.load_device.type != "cpu":
            model_management.unload_model_and_clones(patcher, all_devices=True)
        # torch.nn.utils.weight_norm stores its computed ``weight`` as a plain
        # tensor, so Module.to(cpu) does not move it with weight_g/weight_v.
        # Refresh that cache after ComfyUI offloads the registered parameters.
        model = getattr(patcher, "model", None)
        if model is None:
            return
        with torch.no_grad():
            for module in model.modules():
                for hook in module._forward_pre_hooks.values():
                    name = getattr(hook, "name", None)
                    compute_weight = getattr(hook, "compute_weight", None)
                    if not name or not callable(compute_weight):
                        continue
                    cached = getattr(module, name, None)
                    weight_v = getattr(module, f"{name}_v", None)
                    if torch.is_tensor(cached) and torch.is_tensor(weight_v) and cached.device != weight_v.device:
                        setattr(module, name, compute_weight(module))

    def _encode_reference(
        self,
        audio: tuple[torch.Tensor, int] | None,
        phase_callback: Callable[[str], None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio is None:
            empty = torch.zeros(1, 0, self.inference.latent_dim, dtype=torch.float32)
            lengths = torch.zeros(1, dtype=torch.long)
            return empty, lengths

        waveform, sample_rate = audio
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.target_sample_rate)
        waveform = waveform.contiguous()
        phase_callback("encoding_reference")
        try:
            self._load(self.vae_patcher)
            device_waveform = waveform.to(self.device).unsqueeze(0)
            ref_latent_len = device_waveform.shape[-1] // self.downsample_rate
            ref_lens = torch.tensor([ref_latent_len], dtype=torch.long, device=self.device)
            audio_lens = ref_lens * self.downsample_rate
            ref_latents, encoded_lens = self.inference.vae_model.encoding_and_normalization(
                device_waveform,
                sample_lengths=audio_lens,
            )
            ref_lens = torch.minimum(ref_lens, encoded_lens.to(ref_lens.device))
            return ref_latents.cpu(), ref_lens.cpu()
        finally:
            self._unload(self.vae_patcher)

    def _encode_text(
        self,
        messages: list[dict[str, Any]],
        phase_callback: Callable[[str], None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase_callback("encoding_instruction")
        cond_inputs = self.inference.model.build_cond_inputs([messages], self.inference.model.text_processor)
        fusion = self.inference.model
        try:
            self._load(self.qwen_patcher)
            fusion.layer_weights.data = fusion.layer_weights.data.to(self.device)
            fusion.layer_scale.data = fusion.layer_scale.data.to(self.device)
            with self._autocast():
                text_embeds, context_mask = fusion.encode_text(cond_inputs, self.device)
            return text_embeds.cpu(), context_mask.cpu()
        finally:
            fusion.layer_weights.data = fusion.layer_weights.data.cpu()
            fusion.layer_scale.data = fusion.layer_scale.data.cpu()
            self._unload(self.qwen_patcher)

    def _sample_latents(
        self,
        ref_latents: torch.Tensor,
        ref_lens: torch.Tensor,
        text_embeds: torch.Tensor,
        context_mask: torch.Tensor,
        target_seconds: float,
        nfe_steps: int,
        cfg_strength: float,
        sway_sampling_coef: float,
        seed: int,
        interrupt_callback: Callable[[], None],
        phase_callback: Callable[[str], None],
    ) -> torch.Tensor:
        phase_callback("sampling")
        try:
            self._load(self.transformer_patcher)
            ref_latents = ref_latents.to(self.device)
            ref_lens = ref_lens.to(self.device)
            text_embeds = text_embeds.to(self.device)
            context_mask = context_mask.to(self.device)
            target_frames = max(1, math.ceil(target_seconds * self.target_sample_rate / self.downsample_rate))
            total_lens = ref_lens + target_frames
            t_grid = None
            if self.is_flash:
                nfe_steps = 4
                cfg_strength = 0.0
                sway_sampling_coef = -1.0
                t_grid = [0.0, 0.07612049579620361, 0.2928932309150696, 0.6173166036605835, 1.0]
            with self._autocast():
                generated, _ = self.inference.model.sample_from_embeddings(
                    ref_latents,
                    text_embeds,
                    context_mask,
                    total_lens,
                    lens=ref_lens,
                    steps=nfe_steps,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=None if self.is_flash else sway_sampling_coef,
                    t_grid=t_grid,
                    seed=seed,
                    interrupt_callback=interrupt_callback,
                )
            start = int(ref_lens[0].item())
            end = int(total_lens[0].item())
            latent = generated[0, start:end, :].unsqueeze(0)
            if latent.shape[1] == 0 or not torch.isfinite(latent).all():
                raise RuntimeError("AuK 生成的潜变量为空或包含 NaN/Inf")
            return latent.cpu()
        finally:
            self.inference.model.transformer.clear_cache()
            self._unload(self.transformer_patcher)

    def _decode(self, latent: torch.Tensor, phase_callback: Callable[[str], None]) -> torch.Tensor:
        phase_callback("decoding")
        try:
            self._load(self.vae_patcher)
            latent = self.inference.vae_model.denormalize(latent.to(self.device))
            waveform = self.inference.vae_model.inference_from_latents(latent.permute(0, 2, 1)).cpu()
        finally:
            self._unload(self.vae_patcher)
        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        waveform = waveform.to(torch.float32)
        if waveform.ndim != 2 or waveform.shape[-1] == 0 or not torch.isfinite(waveform).all():
            raise RuntimeError("AuK 输出音频为空或包含 NaN/Inf")
        return waveform

    def generate(
        self,
        messages: list[dict[str, Any]],
        audio: tuple[torch.Tensor, int] | None,
        target_seconds: float,
        nfe_steps: int,
        cfg_strength: float,
        sway_sampling_coef: float,
        seed: int,
        interrupt_callback: Callable[[], None],
        phase_callback: Callable[[str], None],
    ) -> tuple[torch.Tensor, int]:
        with self.lock:
            cuda_devices = []
            if self.device.type == "cuda":
                cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=cuda_devices):
                torch.random.default_generator.manual_seed(seed)
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(seed)
                return self._generate_seeded(
                    messages,
                    audio,
                    target_seconds,
                    nfe_steps,
                    cfg_strength,
                    sway_sampling_coef,
                    seed,
                    interrupt_callback,
                    phase_callback,
                )

    def _generate_seeded(
        self,
        messages: list[dict[str, Any]],
        audio: tuple[torch.Tensor, int] | None,
        target_seconds: float,
        nfe_steps: int,
        cfg_strength: float,
        sway_sampling_coef: float,
        seed: int,
        interrupt_callback: Callable[[], None],
        phase_callback: Callable[[str], None],
    ) -> tuple[torch.Tensor, int]:
        interrupt_callback()
        ref_latents, ref_lens = self._encode_reference(audio, phase_callback)
        try:
            interrupt_callback()
            text_embeds, context_mask = self._encode_text(messages, phase_callback)
            interrupt_callback()
            latent = self._sample_latents(
                ref_latents,
                ref_lens,
                text_embeds,
                context_mask,
                target_seconds,
                nfe_steps,
                cfg_strength,
                sway_sampling_coef,
                seed,
                interrupt_callback,
                phase_callback,
            )
            interrupt_callback()
            waveform = self._decode(latent, phase_callback)
            interrupt_callback()
            return waveform, self.target_sample_rate
        finally:
            for patcher in self.get_models():
                self._unload(patcher)


def validate_sequence_duration(engine: AuKEngine, audio: tuple[torch.Tensor, int] | None, target_seconds: float) -> None:
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("生成时长必须是大于 0 的有限数值")
    source_seconds = 0.0 if audio is None else audio[0].shape[-1] / audio[1]
    source_frames = math.floor(source_seconds * engine.target_sample_rate / engine.downsample_rate)
    if audio is not None and source_frames < 1:
        minimum_seconds = engine.downsample_rate / engine.target_sample_rate
        raise ValueError(f"输入音频过短，至少需要 {minimum_seconds:.3f}s")
    target_frames = max(1, math.ceil(target_seconds * engine.target_sample_rate / engine.downsample_rate))
    max_frames = int(MAX_SEQUENCE_SECONDS * engine.target_sample_rate / engine.downsample_rate)
    if source_frames + target_frames > max_frames:
        raise ValueError(
            f"输入 {source_seconds:.2f}s 与输出 {target_seconds:.2f}s 超过 AuK 的 {MAX_SEQUENCE_SECONDS:.0f}s 总时长限制"
        )
