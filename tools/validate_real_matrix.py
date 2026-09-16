from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ASSETS = ROOT.parent / "AuK-Local" / "assets" / "demo-input-audio"

CASES = (
    ("instruct_tts", "描述生成语音", "一只小猫在叫啊", "自然、清晰、温暖", None, None),
    ("zero_shot_tts", "参考声音克隆", "Ladies and gentlemen, it is an honor to welcome you today.", "", "zero-shot-tts/ref.wav", None),
    ("content_edit", "语音文字编辑", "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.", "", "content-edit/content.wav", None),
    ("lyric_edit", "歌词编辑", "Replace 'rear view' with 'like you' in the lyrics", "", "vocal-edit/vocaledit-en-1-input.wav", None),
    ("pitch", "音高编辑", "+3", "", "pitch/pitch-1-input.wav", None),
    ("speed", "速度编辑", "1.5", "", "speed/speed-edit-1-input.wav", None),
    ("volume", "音量编辑", "+10", "", "energy/energy-edit-1-input.wav", None),
    ("emotion", "情绪编辑", "sad", "", "emotion-edit/en-1-input.wav", None),
    ("timbre", "音色编辑", "低沉而浑厚，语速平稳，吐字清晰", "", "vc/vc-1-input.wav", None),
    ("deaccent", "去口音", "去掉方言口音", "", "accent/accent-anhui-input.wav", None),
    ("nonverbal", "非语言声音编辑", "在“这个月”前增加叹气声", "", "nv/zh-a-input.wav", None),
    ("whisper", "耳语转换", "转换成耳语", "", "pitch/pitch-1-input.wav", None),
    ("enhance", "语音增强", "去噪并去除房间混响", "", "se/se-zh-1-input.wav", None),
    ("quality", "音质修复", "去掉电话感", "", "pitch/pitch-1-input.wav", "telephone"),
    ("speech_separate", "说话人分离", "第二个开始说话的人", "", "ss/zh-1-input.wav", None),
    ("music_separate", "音乐人声提取", "只保留歌声", "", "vocal-extraction/vocal-1-input.wav", None),
    ("target_speaker", "指定说话人提取", "警队规矩", "", "ss/zh-1-input.wav", None),
)


def load_plugin():
    package_name = "auk_comfy_matrix_validation"
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


def load_audio(relative: str | None, transform):
    if relative is None:
        return None, None
    path = LOCAL_ASSETS / relative
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())
    if isinstance(transform, float):
        waveform = waveform[:, : round(transform * sample_rate)]
    elif transform == "telephone":
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), 8_000)
        waveform = torchaudio.functional.lowpass_biquad(waveform, 8_000, 3_400)
        waveform = torchaudio.functional.highpass_biquad(waveform, 8_000, 300)
        waveform = torchaudio.functional.resample(waveform, 8_000, int(sample_rate))
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args(argv)
    selected = [case for case in CASES if not args.only or case[0] in set(args.only)]
    if len(selected) != (len(set(args.only)) if args.only else len(CASES)):
        raise ValueError(f"Unknown or duplicate --only values: {args.only}")
    nodes = load_plugin()
    model_root = ROOT.parent / "AuK-Local" / "models"
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
    output_root = ROOT / "planning" / "comfy-real-v2.0.6"
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "planning" / "comfy-real-v2.0.6.json"
    if args.append and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected_names = {case[0] for case in selected}
        report["cases"] = [case for case in report.get("cases", []) if case.get("task_key") not in selected_names]
    else:
        report = {"schema": 1, "started_at": time.time(), "fixed_seed": 20260916, "cases": []}
    for key, task, primary, secondary, relative, transform in selected:
        input_audio, source_path = load_audio(relative, transform)
        started = time.time()
        print(f"START {key}", flush=True)
        try:
            result = nodes.AuKGenerateEdit.execute(
                engine=engine,
                task=task,
                primary=primary,
                secondary=secondary,
                generation_seconds=29.9,
                seed=20260916,
                input_audio=input_audio,
                nfe_steps=32,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                duration_mode=nodes.AUTO_DURATION_MODE,
            ).result
            output_audio, instruction, metadata_text = result
            output = output_audio["waveform"].squeeze(0).numpy()
            output_path = output_root / f"{key}.wav"
            sf.write(output_path, output.T, output_audio["sample_rate"], subtype="FLOAT")
            metadata = json.loads(metadata_text)
            entry = {
                "task_key": key,
                "state": "succeeded",
                "instruction": instruction,
                "source_asset": relative,
                "source_transform": transform,
                "result_path": output_path.relative_to(ROOT).as_posix(),
                "result_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "metadata": metadata,
                "stale_duration_ignored": key in {"instruct_tts", "zero_shot_tts"}
                or float(metadata["generation_seconds"]) != 29.9,
                "wall_seconds": round(time.time() - started, 3),
            }
        except Exception as exc:
            entry = {
                "task_key": key,
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": round(time.time() - started, 3),
            }
        report["cases"].append(entry)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"DONE {key} {entry['state']}", flush=True)
        if entry["state"] != "succeeded":
            break
    report["finished_at"] = time.time()
    report["all_succeeded"] = len(report["cases"]) == len(CASES) and all(
        item["state"] == "succeeded" and item["stale_duration_ignored"] for item in report["cases"]
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["all_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
