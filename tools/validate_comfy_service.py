"""Run real native nodes on a separately started ComfyUI service.

The published workflows remain frontend JSON. HTTP prompts here are a test
harness that also captures native progress and the node's metadata output.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import time
import uuid
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf

from validate_real_matrix import CASES, LOCAL_ASSETS, ROOT

AUTO = "自动适配（按任务规则）"


async def run_prompt(session, ws, base, client_id, prompt):
    async with session.post(base + "/prompt", json={"prompt": prompt, "client_id": client_id}) as response:
        receipt = await response.json()
        if response.status != 200:
            raise RuntimeError(json.dumps(receipt, ensure_ascii=False))
    prompt_id = receipt["prompt_id"]
    events = []
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        try:
            message = await ws.receive(timeout=1)
            if message.type == aiohttp.WSMsgType.TEXT:
                event = json.loads(message.data)
                if event.get("data", {}).get("prompt_id") == prompt_id and event.get("type") in {
                    "progress", "executing", "execution_error", "execution_success",
                }:
                    events.append(event)
        except asyncio.TimeoutError:
            pass
        async with session.get(base + "/history/" + prompt_id) as response:
            history = await response.json()
        if prompt_id in history:
            entry = history[prompt_id]
            if entry.get("status", {}).get("status_str") != "success":
                raise RuntimeError(json.dumps(entry.get("status"), ensure_ascii=False))
            return prompt_id, entry, events
    raise TimeoutError(f"ComfyUI prompt did not finish: {prompt_id}")


async def save_generated(session, base, entry, folder):
    audio = entry["outputs"]["5"]["audio"][0]
    async with session.get(base + "/view", params=audio) as response:
        response.raise_for_status()
        content = await response.read()
    samples, rate = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
    if not np.isfinite(samples).all() or not samples.size:
        raise ValueError("Output contains nonfinite samples or is empty")
    output = folder / "output.wav"
    sf.write(output, samples, rate, subtype="FLOAT")
    return samples.shape[0] / rate, hashlib.sha256(output.read_bytes()).hexdigest()


async def main_async(args):
    base = args.url.rstrip("/")
    client_id = str(uuid.uuid4())
    report_path = ROOT / "planning" / "comfy-service-v2.0.7.json"
    output_root = ROOT / "planning" / "comfy-service-v2.0.7"
    output_root.mkdir(parents=True, exist_ok=True)
    report = json.loads(report_path.read_text(encoding="utf8")) if args.append else {
        "version": "2.0.7", "started_at": time.time(), "cases": [],
    }
    report["all_succeeded"] = False
    selected = [case for case in CASES if not args.only or case[0] in args.only]
    if args.only and len(selected) != len(set(args.only)):
        raise ValueError("Unknown --only task")
    cases = list(selected)
    if not args.append:
        cases.insert(0, ("trim48_to4", None, "", "", "speed/speed-edit-1-input.wav", None))
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(base + "/ws", params={"clientId": client_id}) as ws:
            for key, task, primary, secondary, relative, transform in cases:
                started = time.time()
                print(f"START {key}", flush=True)
                folder = output_root / key
                folder.mkdir(parents=True, exist_ok=True)
                audio_name = f"audit207_{key}.wav"
                crop_end = 0.0
                if relative:
                    samples, rate = sf.read(LOCAL_ASSETS / relative, dtype="float32", always_2d=True)
                    if key == "trim48_to4":
                        samples = np.tile(samples, (math.ceil(48 * rate / samples.shape[0]), 1))[:48 * rate]
                        crop_end = 4.0
                    elif transform == "telephone":
                        import torch
                        import torchaudio
                        waveform = torch.from_numpy(samples.T.copy()).mean(0, keepdim=True)
                        waveform = torchaudio.functional.resample(waveform, rate, 8000)
                        waveform = torchaudio.functional.lowpass_biquad(waveform, 8000, 3400)
                        waveform = torchaudio.functional.highpass_biquad(waveform, 8000, 300)
                        samples = torchaudio.functional.resample(waveform, 8000, rate).T.numpy()
                    sf.write(args.input_dir / audio_name, samples, rate, subtype="FLOAT")
                    sf.write(folder / "input.wav", samples, rate, subtype="FLOAT")
                prompt = {
                    "1": {"class_type": "AuKModelLoader", "inputs": {"model_variant": "AuK Base", "device": "auto", "dtype": "auto"}},
                    "3": {"class_type": "AuKGenerateEdit", "inputs": {
                        "engine": ["1", 0], "task": task, "primary": primary, "secondary": secondary,
                        "generation_seconds": 12.4, "seed": 20260916, "nfe_steps": 32,
                        "cfg_strength": 2.0, "sway_sampling_coef": -1.0, "duration_mode": AUTO,
                    }},
                    "5": {"class_type": "SaveAudio", "inputs": {"audio": ["3", 0], "filename_prefix": f"audit207/{key}"}},
                    "7": {"class_type": "PreviewAny", "inputs": {"source": ["3", 2]}},
                    "8": {"class_type": "PreviewAny", "inputs": {"source": ["3", 3]}},
                }
                if relative:
                    prompt["2"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_name}}
                    prompt["6"] = {"class_type": "AuKAudioTrim", "inputs": {
                        "audio": ["2", 0], "start_seconds": 0.0, "end_seconds": crop_end,
                    }}
                    prompt["3"]["inputs"]["input_audio"] = ["6", 0]
                if key == "trim48_to4":
                    del prompt["1"], prompt["3"]
                    prompt["5"]["inputs"]["audio"] = ["6", 0]
                    prompt["7"]["inputs"]["source"] = ["6", 2]
                    prompt["8"]["inputs"]["source"] = ["6", 1]
                (folder / "prompt-test.json").write_text(json.dumps(prompt, ensure_ascii=False, indent=2), encoding="utf8")
                try:
                    prompt_id, history, events = await run_prompt(session, ws, base, client_id, prompt)
                    seconds, sha256 = await save_generated(session, base, history, folder)
                    metadata_text = history["outputs"]["7"]["text"][0]
                    target = float(history["outputs"]["8"]["text"][0])
                    metadata = None if task is None else json.loads(metadata_text)
                    if abs(seconds - target) > .03:
                        raise AssertionError(f"Saved audio {seconds}s does not match target {target}s")
                    if key == "trim48_to4" and seconds != 4:
                        raise AssertionError("48-second source was not trimmed to four seconds")
                    if metadata and metadata["requested_generation_seconds"] != 12.4:
                        raise AssertionError("Stale Float test was not applied")
                    entry = {"task_key": key, "state": "succeeded", "prompt_id": prompt_id,
                             "actual_seconds": seconds, "resolved_seconds": target, "output_sha256": sha256,
                             "metadata": metadata, "trim_info": metadata_text if task is None else None,
                             "progress_event_count": sum(event["type"] == "progress" for event in events),
                             "wall_seconds": round(time.time() - started, 3)}
                    (folder / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf8")
                    (folder / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf8")
                except Exception as exc:
                    entry = {"task_key": key, "state": "failed", "error": f"{type(exc).__name__}: {exc}"}
                report["cases"] = [case for case in report["cases"] if case["task_key"] != key]
                report["cases"].append(entry)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
                print(f"DONE {key} {entry['state']}", flush=True)
                if entry["state"] != "succeeded":
                    print(entry.get("error"), flush=True)
                    return 1
    report["all_succeeded"] = True
    report["finished_at"] = time.time()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8195")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--append", action="store_true")
    raise SystemExit(asyncio.run(main_async(parser.parse_args())))
