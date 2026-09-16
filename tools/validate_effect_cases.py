from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = ROOT.parent / "AuK-Local"


def load_plugin():
    package_name = "auk_comfy_effect_validation"
    spec = importlib.util.spec_from_file_location(
        package_name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load ComfyUI AuK package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return __import__(f"{package_name}.nodes", fromlist=["nodes"])


def main() -> int:
    source_path = LOCAL_ROOT / "assets" / "demo-input-audio" / "pitch" / "pitch-1-input.wav"
    samples, rate = sf.read(source_path, dtype="float32", always_2d=True)
    clean = samples.mean(axis=1, dtype=np.float32)
    rng = np.random.default_rng(20260916)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(np.square(clean), dtype=np.float64)))
    noise *= signal_rms / (10 ** (8.0 / 20.0)) / max(float(np.sqrt(np.mean(np.square(noise)))), 1e-8)
    ir = np.zeros(round(0.32 * rate), dtype=np.float32)
    ir[0] = 1.0
    for delay, gain in ((0.045, 0.42), (0.091, 0.28), (0.147, 0.18), (0.235, 0.10)):
        ir[min(len(ir) - 1, round(delay * rate))] = gain
    degraded = np.convolve(clean, ir, mode="full")[: len(clean)].astype(np.float32) + noise
    peak = float(np.max(np.abs(degraded)))
    if peak > 0.95:
        degraded *= 0.95 / peak

    nodes = load_plugin()
    model_root = LOCAL_ROOT / "models"
    engine = nodes.AuKEngine(
        model_root / "AuK" / "auk_base.safetensors",
        model_root / "AuK" / "config.yaml",
        model_root / "Qwen2.5-Omni-3B",
        torch.device("cuda:0"),
        "bf16",
    )
    engine.model_variant = "AuK Base"
    manifest = nodes.load_manifest()["models"]
    engine.model_revision = manifest["AuK"]["revision"]
    engine.qwen_revision = manifest["Qwen2.5-Omni-3B"]["revision"]
    started = time.time()
    result = nodes.AuKGenerateEdit.execute(
        engine=engine,
        task="语音增强",
        primary="去噪并去除房间混响",
        secondary="",
        generation_seconds=29.9,
        seed=42,
        input_audio={"waveform": torch.from_numpy(degraded).reshape(1, 1, -1), "sample_rate": int(rate)},
        nfe_steps=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        duration_mode=nodes.AUTO_DURATION_MODE,
    ).result
    output_audio, instruction, metadata_text = result
    out_dir = ROOT / "planning" / "comfy-effect-v2.0.6"
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path, degraded_path, output_path = (out_dir / "clean-reference.wav", out_dir / "degraded-input.wav", out_dir / "enhance-output.wav")
    sf.write(clean_path, clean, int(rate), subtype="FLOAT")
    sf.write(degraded_path, degraded, int(rate), subtype="FLOAT")
    output = output_audio["waveform"].squeeze(0).numpy().T
    sf.write(output_path, output, int(output_audio["sample_rate"]), subtype="FLOAT")
    report = {
        "schema": 1,
        "task": "enhance_controlled",
        "state": "succeeded",
        "instruction": instruction,
        "seed": 42,
        "wall_seconds": round(time.time() - started, 3),
        "clean_reference": clean_path.relative_to(ROOT).as_posix(),
        "degraded_input": degraded_path.relative_to(ROOT).as_posix(),
        "result_path": output_path.relative_to(ROOT).as_posix(),
        "result_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "metadata": json.loads(metadata_text),
    }
    (ROOT / "planning" / "comfy-effect-v2.0.6.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"state": "succeeded", "wall_seconds": report["wall_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
