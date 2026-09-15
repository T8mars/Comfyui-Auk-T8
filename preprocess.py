from __future__ import annotations

import math
from typing import Any


VAD_SKIP_TASKS = frozenset(
    {"enhance", "quality", "speech_separate", "target_speaker", "music_separate", "nonverbal", "lyric_edit"}
)
WHISPER_TARGET_RMS = 0.0064
WHISPER_TO_NORMAL_TARGET_RMS = 0.000707945784384138
PEAK_CEILING = 0.95


def _energy_trim(waveform, sample_rate: int):
    """Trim obvious leading/trailing silence when Silero VAD is unavailable.

    This is deliberately conservative: it only removes edge regions and never
    removes internal pauses, which are meaningful to AuK editing tasks.
    """
    import torch

    mono = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
    if mono.numel() < max(2, round(0.10 * sample_rate)):
        return waveform, None
    frame = max(1, round(0.03 * sample_rate))
    hop = max(1, round(0.01 * sample_rate))
    if mono.numel() < frame:
        return waveform, None
    frames = mono.unfold(0, frame, hop)
    rms = torch.sqrt(torch.mean(frames.to(torch.float64) ** 2, dim=1))
    high = float(torch.quantile(rms, 0.95))
    noise = float(torch.quantile(rms, 0.20))
    if not math.isfinite(high) or high <= 1e-7:
        return waveform, None
    threshold = max(1e-5, min(noise * 2.5, high * 0.15))
    active = torch.nonzero(rms >= threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        return waveform, None
    padding = round(0.10 * sample_rate)
    start = max(0, int(active[0]) * hop - padding)
    end = min(mono.numel(), int(active[-1]) * hop + frame + padding)
    if end <= start or (start == 0 and end == mono.numel()):
        return waveform, None
    return waveform[:, start:end].contiguous(), (start / sample_rate, end / sample_rate)


def _normalize_rms(waveform, target_rms: float):
    import torch

    mono = waveform.mean(dim=0).to(torch.float64)
    measured = float(torch.sqrt(torch.mean(mono**2)))
    if not math.isfinite(measured) or measured <= 1e-9:
        return waveform, measured
    output = waveform.to(torch.float64) * (target_rms / measured)
    peak = float(output.abs().max())
    if math.isfinite(peak) and peak > PEAK_CEILING:
        output *= PEAK_CEILING / peak
    return output.to(dtype=waveform.dtype), measured


def prepare_model_audio(waveform, sample_rate: int, task_key: str, primary: str) -> tuple[Any, dict[str, Any]]:
    """Apply the task-aware input preparation used by AuK's prompt enhancer."""
    original_frames = int(waveform.shape[-1])
    prepared = waveform
    trim_bounds = None
    whisper_to_normal = task_key == "whisper" and any(
        word in str(primary or "") for word in ("正常", "别耳语", "非耳语")
    )
    if task_key not in VAD_SKIP_TASKS and not whisper_to_normal:
        prepared, trim_bounds = _energy_trim(prepared, sample_rate)

    target_rms = None
    measured_rms = None
    if task_key == "whisper":
        target_rms = WHISPER_TO_NORMAL_TARGET_RMS if whisper_to_normal else WHISPER_TARGET_RMS
        prepared, measured_rms = _normalize_rms(prepared, target_rms)

    return prepared.contiguous(), {
        "method": "task_aware_energy_vad_rms",
        "source_frames_before": original_frames,
        "source_frames_after": int(prepared.shape[-1]),
        "vad_trim_bounds_sec": list(trim_bounds) if trim_bounds is not None else None,
        "whisper_target_rms": target_rms,
        "whisper_measured_rms_before": measured_rms,
    }


def limit_vocal_output(waveform, task_key: str):
    """Apply AuK's vocal-task peak ceiling without an optional LUFS package."""
    if task_key not in {"lyric_edit", "music_separate"}:
        return waveform, False
    peak = float(waveform.abs().max()) if waveform.numel() else 0.0
    if math.isfinite(peak) and peak > PEAK_CEILING:
        return waveform * (PEAK_CEILING / peak), True
    return waveform, False
