from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_plugin():
    package_name = "auk_comfy_real_validation"
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load ComfyUI AuK package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return __import__(f"{package_name}.nodes", fromlist=["nodes"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one real AuK Base inference through the native ComfyUI node")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task", default="情绪编辑")
    parser.add_argument("--primary", default="恐惧")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    nodes = load_plugin()
    samples, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())
    engine = nodes.AuKEngine(
        args.model_root / "AuK" / "auk_base.safetensors",
        args.model_root / "AuK" / "config.yaml",
        args.model_root / "Qwen2.5-Omni-3B",
        torch.device("cuda:0"),
        "bf16",
    )
    engine.model_variant = "AuK Base"
    manifest = nodes.load_manifest()["models"]
    engine.model_revision = manifest["AuK"]["revision"]
    engine.qwen_revision = manifest["Qwen2.5-Omni-3B"]["revision"]
    result = nodes.AuKGenerateEdit.execute(
        engine=engine,
        task=args.task,
        primary=args.primary,
        secondary="",
        generation_seconds=29.9,
        seed=20260916,
        input_audio={"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},
        nfe_steps=32,
        cfg_strength=2.0,
        sway_sampling_coef=-1.0,
        duration_mode=nodes.AUTO_DURATION_MODE,
    ).result
    output_audio, instruction, metadata_text = result
    output = output_audio["waveform"].squeeze(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, output.T.numpy(), output_audio["sample_rate"], subtype="FLOAT")
    metadata = json.loads(metadata_text)
    metadata["instruction"] = instruction
    metadata["input_path"] = str(args.input.resolve())
    metadata["output_path"] = str(args.output.resolve())
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
