from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class FakeEngine:
    is_flash = True
    target_sample_rate = 24_000
    downsample_rate = 1_920
    model_variant = "AuK-Flash"
    model_revision = "test-model-revision"
    qwen_revision = "test-qwen-revision"
    device = torch.device("cuda:0")
    dtype = "bf16"

    def __init__(self):
        self.call = None

    def generate(self, *args):
        self.call = args
        phase_callback = args[-1]
        for phase in ("encoding_instruction", "sampling", "decoding"):
            phase_callback(phase)
        return torch.zeros(1, 24_000), 24_000


def test_v3_extension_registers_two_native_nodes(plugin):
    extension = asyncio.run(plugin.comfy_entrypoint())
    classes = asyncio.run(extension.get_node_list())
    assert [node.__name__ for node in classes] == ["AuKModelLoader", "AuKGenerateEdit"]


def test_runtime_has_no_service_client(plugin):
    root = Path(plugin.__file__).parent
    runtime_text = "\n".join((root / name).read_text(encoding="utf-8") for name in ("nodes.py", "runtime.py"))
    for forbidden in ("urllib", "service_url", "token_file", "AUK_LOCAL_CONNECTION", "127.0.0.1"):
        assert forbidden not in runtime_text


def test_all_task_templates_build_local_instruction(plugin):
    assert len(plugin.nodes.TASKS) == 16
    for task in plugin.nodes.TASKS:
        instruction = plugin.nodes.build_instruction(task.key, "主要内容", "附加要求")
        assert "主要内容" in instruction


def test_normalize_audio_downmixes_to_mono(plugin):
    value = {"waveform": torch.tensor([[[1.0, -1.0], [0.0, 0.0]]]), "sample_rate": 16_000}
    waveform, sample_rate = plugin.nodes.normalize_audio(value)
    assert sample_rate == 16_000
    assert waveform.shape == (1, 2)
    assert torch.equal(waveform, torch.tensor([[0.5, -0.5]]))


def test_audio_task_requires_audio(plugin):
    with pytest.raises(ValueError, match="需要连接"):
        plugin.nodes.AuKGenerateEdit.execute(
            FakeEngine(),
            "参考声音克隆",
            "测试",
            "",
            1.0,
            42,
        )


def test_flash_generation_is_native_and_uses_fixed_recipe(plugin):
    engine = FakeEngine()
    result = plugin.nodes.AuKGenerateEdit.execute(
        engine,
        "描述生成语音",
        "测试文本",
        "自然声音",
        1.0,
        42,
        nfe_steps=32,
        cfg_strength=2.0,
    )
    audio, instruction, metadata_text = result.result
    assert audio["waveform"].shape == (1, 1, 24_000)
    assert audio["sample_rate"] == 24_000
    assert "测试文本" in instruction
    messages = engine.call[0]
    assert messages[0]["content"][0]["text"].endswith("|<no_prompt_audio>|")
    metadata = json.loads(metadata_text)
    assert metadata["runtime"] == "native_comfyui"
    assert (metadata["nfe_steps"], metadata["cfg_strength"], metadata["sway_sampling_coef"]) == (4, 0.0, -1.0)


def test_audio_generation_passes_mono_audio_to_engine(plugin):
    engine = FakeEngine()
    source = {"waveform": torch.ones(1, 2, 8_000), "sample_rate": 16_000}
    plugin.nodes.AuKGenerateEdit.execute(
        engine,
        "参考声音克隆",
        "测试文本",
        "参考文字",
        1.0,
        42,
        input_audio=source,
    )
    messages, audio = engine.call[:2]
    assert audio[0].shape == (1, 8_000)
    assert messages[0]["content"][1]["audio"].shape == (8_000,)


def test_sequence_limit_counts_source_and_target(plugin):
    engine = FakeEngine()
    source = (torch.zeros(1, 24_000 * 10), 24_000)
    with pytest.raises(ValueError, match="30s"):
        plugin.nodes.validate_sequence_duration(engine, source, 20.1)


def test_native_engine_seed_covers_reference_encoding_and_restores_rng(plugin):
    engine = plugin.runtime.AuKEngine.__new__(plugin.runtime.AuKEngine)
    engine.device = torch.device("cpu")
    engine.lock = threading.Lock()
    engine.inference = SimpleNamespace(target_sample_rate=24_000)
    engine.get_models = lambda: []
    engine._encode_reference = lambda _audio, _callback: (torch.randn(1, 1, 1), torch.ones(1, dtype=torch.long))
    engine._encode_text = lambda _messages, _callback: (torch.zeros(1), torch.ones(1, dtype=torch.bool))
    engine._sample_latents = lambda ref, *_args: ref
    engine._decode = lambda latent, _callback: latent.flatten().unsqueeze(0)

    torch.manual_seed(9001)
    state_before = torch.random.get_rng_state().clone()
    first, _ = engine.generate([], None, 0.2, 4, 0.0, -1.0, 77, lambda: None, lambda _phase: None)
    assert torch.equal(torch.random.get_rng_state(), state_before)
    second, _ = engine.generate([], None, 0.2, 4, 0.0, -1.0, 77, lambda: None, lambda _phase: None)
    assert torch.equal(first, second)


def test_runtime_only_accepts_safetensors(plugin):
    root = Path(plugin.__file__).parent / "auk_core"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "torch.load(" not in source


def test_model_paths_are_confined_to_comfy_models(plugin, tmp_path, monkeypatch):
    manifest = {
        "models": {
            "AuK-Flash": {"files": {"auk_flash.safetensors": {"size": 1}, "vae.safetensors": {"size": 1}, "config.yaml": {"size": 1}}},
            "Qwen2.5-Omni-3B": {"files": {"config.json": {"size": 1}}},
        }
    }
    monkeypatch.setattr(plugin.nodes, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(plugin.nodes, "load_manifest", lambda: manifest)
    for relative in ("AuK-Flash/auk_flash.safetensors", "AuK-Flash/vae.safetensors", "AuK-Flash/config.yaml", "Qwen2.5-Omni-3B/config.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    checkpoint, config, qwen = plugin.nodes.resolve_model_files("AuK-Flash")
    assert checkpoint.is_relative_to(tmp_path)
    assert config.is_relative_to(tmp_path)
    assert qwen.is_relative_to(tmp_path)


def test_example_workflows_only_use_native_node_ids(plugin):
    root = Path(plugin.__file__).parent / "example_workflows"
    for path in root.glob("*.json"):
        workflow = json.loads(path.read_text(encoding="utf-8"))
        node_ids = {node["type"] for node in workflow["nodes"]}
        assert "AuKModelLoader" in node_ids
        assert "AuKGenerateEdit" in node_ids
        assert not {"AuKLocalConnection", "AuKLocalGenerateEdit"} & node_ids
