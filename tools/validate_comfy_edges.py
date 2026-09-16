from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf

from validate_comfy_service import AUTO, LOCAL_ASSETS, ROOT, run_prompt, save_generated


async def main(args):
    base = args.url.rstrip("/")
    client = str(uuid.uuid4())
    output_root = ROOT / "planning" / "comfy-edges-v2.0.7"
    output_root.mkdir(parents=True, exist_ok=True)
    samples, rate = sf.read(LOCAL_ASSETS / "speed/speed-edit-1-input.wav", dtype="float32", always_2d=True)
    source = np.tile(samples, (math.ceil(48 * rate / len(samples)), 1))[:48 * rate]
    sf.write(args.input_dir / "audit207_edges48.wav", source, rate, subtype="FLOAT")
    sf.write(args.input_dir / "audit207_edges30.wav", source[:30 * rate], rate, subtype="FLOAT")
    sf.write(args.input_dir / "audit207_edges_over30.wav", source[:30 * rate + 1], rate, subtype="FLOAT")
    checks = [
        ("speed_cropped4", "速度编辑", "1.25", "audit207_edges48.wav", 0, 4, None),
        ("manual_tts", "描述生成语音", "你好，这是固定三秒的测试。", None, 0, 0, None),
        ("input30_output30", "语音增强", "去噪并去除房间混响", "audit207_edges30.wav", 0, 0, None),
        ("one_extra_sample_rejected", "语音增强", "去噪", "audit207_edges_over30.wav", 0, 0, "30.000042"),
        ("invalid_trim_rejected", None, "", "audit207_edges48.wav", 4, 2, "裁剪范围无效"),
        ("trim_recovers_after_error", None, "", "audit207_edges48.wav", 1, 3, None),
    ]
    report = {"version": "2.0.7", "started_at": time.time(), "cases": [], "all_succeeded": False}
    report_path = ROOT / "planning" / "comfy-edges-v2.0.7.json"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        async with session.ws_connect(base + "/ws", params={"clientId": client}) as ws:
            for key, task, primary, audio, start, end, expected_error in checks:
                print(f"START {key}", flush=True)
                folder = output_root / key
                folder.mkdir(exist_ok=True)
                prompt = {
                    "5": {"class_type": "SaveAudio", "inputs": {"audio": ["6", 0], "filename_prefix": f"audit207_edges/{key}"}},
                    "7": {"class_type": "PreviewAny", "inputs": {"source": ["6", 2]}},
                    "8": {"class_type": "PreviewAny", "inputs": {"source": ["6", 1]}},
                }
                if audio:
                    prompt["2"] = {"class_type": "LoadAudio", "inputs": {"audio": audio}}
                    prompt["6"] = {"class_type": "AuKAudioTrim", "inputs": {"audio": ["2", 0], "start_seconds": start, "end_seconds": end}}
                if task:
                    prompt["1"] = {"class_type": "AuKModelLoader", "inputs": {"model_variant": "AuK Base", "device": "auto", "dtype": "auto"}}
                    prompt["3"] = {"class_type": "AuKGenerateEdit", "inputs": {
                        "engine": ["1", 0], "task": task, "primary": primary, "secondary": "",
                        "generation_seconds": 3.0 if key == "manual_tts" else 12.4, "seed": 20260916,
                        "nfe_steps": 32, "cfg_strength": 2.0, "sway_sampling_coef": -1.0,
                        "duration_mode": "手动指定" if key == "manual_tts" else AUTO,
                    }}
                    if audio:
                        prompt["3"]["inputs"]["input_audio"] = ["6", 0]
                    for output, slot in (("5", 0), ("7", 2), ("8", 3)):
                        prompt[output]["inputs"]["audio" if output == "5" else "source"] = ["3", slot]
                try:
                    pid, history, events = await run_prompt(session, ws, base, client, prompt)
                    if expected_error:
                        raise AssertionError("Invalid input unexpectedly succeeded")
                    seconds, sha256 = await save_generated(session, base, history, folder)
                    target = float(history["outputs"]["8"]["text"][0])
                    metadata = json.loads(history["outputs"]["7"]["text"][0]) if task else None
                    assert abs(seconds - target) < .03
                    if key == "speed_cropped4":
                        assert metadata["original_input_seconds"] == 4 and seconds < 4
                    if key == "input30_output30":
                        assert metadata["original_input_seconds"] == 30 and seconds == 30
                    if key == "manual_tts":
                        assert seconds == 3
                    if key == "trim_recovers_after_error":
                        assert seconds == 2 and sf.info(args.input_dir / audio).duration == 48
                    entry = {"key": key, "state": "succeeded", "prompt_id": pid, "seconds": seconds,
                             "metadata": metadata, "sha256": sha256, "progress_events": len(events)}
                    (folder / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf8")
                except RuntimeError as exc:
                    if not expected_error or expected_error not in str(exc):
                        raise
                    entry = {"key": key, "state": "expected_error_verified", "error": str(exc)}
                report["cases"].append(entry)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
                print(f"DONE {key} {entry['state']}", flush=True)
    report["all_succeeded"] = True
    report["finished_at"] = time.time()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8195")
    parser.add_argument("--input-dir", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
