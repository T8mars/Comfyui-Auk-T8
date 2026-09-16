from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from typing import Any


VAD_SKIP_TASKS = frozenset(
    {"enhance", "quality", "speech_separate", "target_speaker", "music_separate", "nonverbal", "lyric_edit"}
)
VAD_TRIM_PAD_SEC = 0.1
VAD_NORM_RMS_THRESHOLD = 0.05
VAD_NORM_TARGET_PEAK = 0.99
WHISPER_TARGET_LUFS = -44.47
WHISPER_TARGET_RMS = 0.0064
WHISPER_TO_NORMAL_TARGET_RMS = 0.000707945784384138
VOCAL_OUTPUT_TARGET_LUFS = -14.0
PEAK_CEILING = 0.95

_VAD_MODEL = None
_VAD_LOCK = threading.Lock()


def _add_vendored_dependencies() -> None:
    vendor = Path(__file__).with_name("_vendor")
    if vendor.is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def _get_vad_model():
    global _VAD_MODEL
    with _VAD_LOCK:
        if _VAD_MODEL is None:
            _add_vendored_dependencies()
            from silero_vad import load_silero_vad

            _VAD_MODEL = load_silero_vad()
        return _VAD_MODEL


def _silero_speech_bounds(waveform, sample_rate: int):
    """Return official PE-style unpadded first/last speech bounds."""
    try:
        import torch
        import torchaudio

        _add_vendored_dependencies()
        from silero_vad import get_speech_timestamps

        mono = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
        signal = mono
        if sample_rate != 16_000:
            signal = torchaudio.functional.resample(signal, sample_rate, 16_000)
        signal = signal.contiguous()
        rms = float(torch.sqrt(torch.mean(signal.to(torch.float64) ** 2)))
        peak = float(signal.abs().max()) if signal.numel() else 0.0
        if 0 < rms < VAD_NORM_RMS_THRESHOLD and peak > 0:
            signal = signal * (VAD_NORM_TARGET_PEAK / peak)
        model = _get_vad_model()
        with _VAD_LOCK:
            timestamps = get_speech_timestamps(
                signal,
                model,
                sampling_rate=16_000,
                return_seconds=True,
                time_resolution=3,
            )
        if not timestamps:
            return None
        return float(timestamps[0]["start"]), float(timestamps[-1]["end"])
    except Exception:
        return None


def _energy_speech_bounds(waveform, sample_rate: int):
    """Conservative offline fallback when bundled Silero cannot load."""
    import torch

    mono = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
    if mono.numel() < max(2, round(0.10 * sample_rate)):
        return None
    frame = max(1, round(0.03 * sample_rate))
    hop = max(1, round(0.01 * sample_rate))
    if mono.numel() < frame:
        return None
    frames = mono.unfold(0, frame, hop)
    rms = torch.sqrt(torch.mean(frames.to(torch.float64) ** 2, dim=1))
    high = float(torch.quantile(rms, 0.95))
    noise = float(torch.quantile(rms, 0.20))
    if not math.isfinite(high) or high <= 1e-7:
        return None
    threshold = max(1e-5, min(noise * 2.5, high * 0.15))
    active = torch.nonzero(rms >= threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        return None
    start = int(active[0]) * hop / sample_rate
    end = min(mono.numel(), int(active[-1]) * hop + frame) / sample_rate
    return (start, end) if end > start else None


def _trim_with_padding(waveform, sample_rate: int, bounds):
    if bounds is None:
        return waveform, None
    total = int(waveform.shape[-1])
    padding = round(VAD_TRIM_PAD_SEC * sample_rate)
    start = max(0, round(bounds[0] * sample_rate) - padding)
    end = min(total, round(bounds[1] * sample_rate) + padding)
    if end <= start or (start == 0 and end == total):
        return waveform, (start / sample_rate, end / sample_rate)
    return waveform[:, start:end].contiguous(), (start / sample_rate, end / sample_rate)


def _normalize_level(waveform, sample_rate: int, *, target_rms: float, target_lufs: float | None = None):
    import torch

    mono = waveform.mean(dim=0).to(torch.float64)
    measured_rms = float(torch.sqrt(torch.mean(mono**2)))
    measured_lufs = None
    gain = None
    method = "rms"
    if target_lufs is not None:
        try:
            _add_vendored_dependencies()
            import pyloudnorm as pyln

            measured_lufs = float(pyln.Meter(sample_rate).integrated_loudness(mono.numpy()))
            if math.isfinite(measured_lufs):
                gain = 10.0 ** ((target_lufs - measured_lufs) / 20.0)
                method = "lufs"
        except Exception:
            gain = None
    if gain is None:
        if not math.isfinite(measured_rms) or measured_rms <= 1e-9:
            return waveform, measured_rms, measured_lufs, "unchanged"
        gain = target_rms / measured_rms
    output = waveform.to(torch.float64) * gain
    peak = float(output.abs().max()) if output.numel() else 0.0
    if math.isfinite(peak) and peak > PEAK_CEILING:
        output *= PEAK_CEILING / peak
    return output.to(dtype=waveform.dtype), measured_rms, measured_lufs, method


def prepare_model_audio(waveform, sample_rate: int, task_key: str, primary: str) -> tuple[Any, dict[str, Any]]:
    """Apply AuK Prompt Enhancer VAD, duration and whisper-level rules."""
    original_frames = int(waveform.shape[-1])
    original_seconds = original_frames / sample_rate
    prepared = waveform
    speech_bounds = None
    trim_bounds = None
    vad_engine = "skipped"
    whisper_to_normal = task_key == "whisper" and any(
        word in str(primary or "") for word in ("正常", "别耳语", "非耳语")
    )
    if task_key not in VAD_SKIP_TASKS and not whisper_to_normal:
        speech_bounds = _silero_speech_bounds(prepared, sample_rate)
        vad_engine = "silero" if speech_bounds is not None else "energy_fallback"
        if speech_bounds is None:
            speech_bounds = _energy_speech_bounds(prepared, sample_rate)
        prepared, trim_bounds = _trim_with_padding(prepared, sample_rate, speech_bounds)

    target_rms = None
    target_lufs = None
    measured_rms = None
    measured_lufs = None
    level_method = None
    if task_key == "whisper":
        target_rms = WHISPER_TO_NORMAL_TARGET_RMS if whisper_to_normal else WHISPER_TARGET_RMS
        target_lufs = None if whisper_to_normal else WHISPER_TARGET_LUFS
        prepared, measured_rms, measured_lufs, level_method = _normalize_level(
            prepared,
            sample_rate,
            target_rms=target_rms,
            target_lufs=target_lufs,
        )

    speech_seconds = speech_bounds[1] - speech_bounds[0] if speech_bounds is not None else original_seconds
    return prepared.contiguous(), {
        "method": "official_silero_vad_lufs" if vad_engine == "silero" else "official_rules_with_energy_fallback",
        "vad_engine": vad_engine,
        "source_frames_before": original_frames,
        "source_frames_after": int(prepared.shape[-1]),
        "speech_seconds_unpadded": speech_seconds,
        "vad_speech_bounds_sec": list(speech_bounds) if speech_bounds is not None else None,
        "vad_trim_bounds_sec": list(trim_bounds) if trim_bounds is not None else None,
        "whisper_level_method": level_method,
        "whisper_target_lufs": target_lufs,
        "whisper_measured_lufs_before": measured_lufs,
        "whisper_target_rms": target_rms,
        "whisper_measured_rms_before": measured_rms,
    }


def limit_vocal_output(waveform, task_key: str, sample_rate: int = 24_000):
    """Match the official Gradio -14 LUFS downward limiter and 0.95 peak cap."""
    if task_key not in {"lyric_edit", "music_separate"}:
        return waveform, False
    import torch

    output = waveform.to(torch.float64)
    applied = False
    try:
        _add_vendored_dependencies()
        import pyloudnorm as pyln

        mono = output.mean(dim=0) if output.ndim == 2 else output
        measured_lufs = float(pyln.Meter(sample_rate).integrated_loudness(mono.numpy()))
        if math.isfinite(measured_lufs) and measured_lufs > VOCAL_OUTPUT_TARGET_LUFS:
            output *= 10.0 ** ((VOCAL_OUTPUT_TARGET_LUFS - measured_lufs) / 20.0)
            applied = True
    except Exception:
        pass
    peak = float(output.abs().max()) if output.numel() else 0.0
    if math.isfinite(peak) and peak > PEAK_CEILING:
        output *= PEAK_CEILING / peak
        applied = True
    return output.to(dtype=waveform.dtype), applied


def protect_audio_output(waveform):
    """Reject invalid model output and prevent integer audio exporters clipping.

    Normal output, including quiet whispers, keeps its original gain. Only
    peaks outside the standard AUDIO range are scaled to a 0.99 ceiling.
    """
    import torch

    if not torch.is_tensor(waveform) or waveform.ndim != 2 or min(waveform.shape) < 1:
        raise ValueError("模型输出必须是非空 [C, T] 音频")
    if not torch.isfinite(waveform).all():
        raise ValueError("模型输出包含 NaN 或 Inf；请更换 Seed 后重试")
    peak = float(waveform.abs().max())
    gain = 0.99 / peak if peak > 1.0 else 1.0
    return waveform * gain if gain != 1.0 else waveform, {
        "applied": gain != 1.0,
        "peak_before": peak,
        "gain_db": 20 * math.log10(gain),
    }
