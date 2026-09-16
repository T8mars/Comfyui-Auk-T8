from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from comfy_api.v0_0_2 import io


class FakeEngine:
    is_flash = True
    target_sample_rate = 24_000
    downsample_rate = 480
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


def test_web_task_guide_covers_every_visible_task(plugin):
    root = Path(plugin.__file__).parent
    assert plugin.WEB_DIRECTORY == "./web"
    guide_data = json.loads((root / "web" / "task_guides.json").read_text(encoding="utf-8"))
    assert set(guide_data) == {task.label for task in plugin.nodes.TASKS}
    for task in plugin.nodes.TASKS:
        entry = guide_data[task.label]
        guide = plugin.nodes.TASK_GUIDES[task.key]
        assert entry["key"] == task.key
        assert entry["primary_label"] == task.primary_label
        assert entry["secondary_label"] == task.secondary_label
        assert entry["requirement"] == guide.requirement
        assert entry["example"] == guide.example
        assert entry["note"] == guide.note
    script = (root / "web" / "js" / "auk_task_guide.js").read_text(encoding="utf-8")
    assert "loadedGraphNode" in script
    assert "serialize: false" in script


def test_runtime_has_no_service_client(plugin):
    root = Path(plugin.__file__).parent
    runtime_text = "\n".join((root / name).read_text(encoding="utf-8") for name in ("nodes.py", "runtime.py"))
    for forbidden in ("urllib", "service_url", "token_file", "AUK_LOCAL_CONNECTION", "127.0.0.1"):
        assert forbidden not in runtime_text


def test_all_task_templates_build_local_instruction(plugin):
    assert len(plugin.nodes.TASKS) == 17
    discrete_inputs = {
        "pitch": "+2", "speed": "1.25", "volume": "-10", "emotion": "悲伤",
        "content_edit": "把“今天”改成“明天”",
        "lyric_edit": "把歌词“今天”改成“明天”",
        "whisper": "转换成耳语",
        "enhance": "去噪并去混响",
        "music_separate": "只保留歌声",
        "quality": "去掉电话感",
        "nonverbal": "在开头增加笑声",
        "speech_separate": "第一个开始说话的人",
    }
    for task in plugin.nodes.TASKS:
        primary = discrete_inputs.get(task.key, "主要内容")
        instruction = plugin.nodes.build_instruction(task.key, primary, "附加要求")
        assert instruction


def test_zero_shot_uses_official_instruction_without_reference_transcript(plugin):
    instruction = plugin.nodes.build_instruction("zero_shot_tts", "你好", "参考音频文字")
    assert instruction == 'Say the following with the same voice: "你好"'
    assert "参考音频文字" not in instruction


@pytest.mark.parametrize(
    ("task_key", "primary", "expected"),
    [
        ("instruct_tts", "欢迎回来", '请基于下面的描述: "不应发送",生成语音内容"欢迎回来".'),
        ("zero_shot_tts", "欢迎回来", 'Say the following with the same voice: "欢迎回来"'),
        ("content_edit", "把“今天”改成“明天”", "把‘今天’改成‘明天’"),
        ("lyric_edit", "把歌词“今天”改成“明天”", "把这段歌词中的“今天”改成“明天”。"),
        ("pitch", "+1", "将音调升高1个半音。"),
        ("speed", "1.5", "将语速调整为1.5倍。"),
        ("volume", "-10", "将音量降低10分贝。"),
        ("emotion", "悲伤", "将情感转变为悲伤。"),
        ("timbre", "低沉男声", "请将这段音频的音色修改为符合以下描述的声音：“低沉男声”。"),
        ("deaccent", "去掉方言口音", "请去掉这段语音里的方言口音，保持说话人音色一致。"),
        ("nonverbal", "在开头增加笑声", "在语音开头增加笑声。"),
        ("whisper", "转换成耳语", "用小声耳语的方式把这段话说出来。"),
        ("enhance", "去噪并去混响", "请对这段语音做纯净化处理，保留所有说话人的人声，并去除其中的噪声和混响，输出与输入等长的干净人声。"),
        ("quality", "去掉电话感", "This audio suffers from limited bandwidth. Please restore it to a wideband, clear-sounding speech."),
        ("speech_separate", "第一个开始说话的人", "Keep only the first speaker"),
        ("music_separate", "只保留歌声", "请只保留歌声，其余声音都去掉。"),
        ("target_speaker", "欢迎大家", "Keep only the speaker who says “欢迎大家”"),
    ],
)
def test_every_task_uses_exact_official_instruction_and_ignores_secondary(plugin, task_key, primary, expected):
    assert plugin.nodes.build_instruction(task_key, primary, "不应发送") == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("sad", "Say this in a sad tone"), ("fearful", "Say this in a afraid tone")],
)
def test_english_emotion_uses_official_demo_wording(plugin, value, expected):
    assert plugin.nodes.build_instruction("emotion", value) == expected


@pytest.mark.parametrize(
    ("task_key", "raw", "expected"),
    [
        ("content_edit", "夜色那么美改成白天那么美", "把‘夜色那么美’改成‘白天那么美’"),
        ("content_edit", "把“今天下午开会”改成“明天上午开会”", "把‘今天下午开会’改成‘明天上午开会’"),
        ("lyric_edit", "把歌词‘明天你好’改成‘未来你好’", "把这段歌词中的“明天你好”改成“未来你好”。"),
    ],
)
def test_replacement_requests_use_official_templates(plugin, task_key, raw, expected):
    assert plugin.nodes.build_instruction(task_key, raw, "整段原文不应发送") == expected


@pytest.mark.parametrize(
    ("task_key", "value", "expected"),
    [
        ("pitch", "+2", "将音调升高2个半音。"),
        ("pitch", "降低 3 个半音", "将音调降低3个半音。"),
        ("volume", "+10", "将音量升高10分贝。"),
        ("volume", "-5 dB", "将音量降低5分贝。"),
    ],
)
def test_signed_adjustments_use_official_direction_templates(plugin, task_key, value, expected):
    assert plugin.nodes.build_instruction(task_key, value) == expected


@pytest.mark.parametrize(("task_key", "value"), [("pitch", "0"), ("pitch", "4"), ("volume", "3")])
def test_signed_adjustments_reject_unsupported_values(plugin, task_key, value):
    with pytest.raises(ValueError):
        plugin.nodes.build_instruction(task_key, value)


@pytest.mark.parametrize("value", ["升高 -2 个半音", "降低 +2 个半音"])
def test_signed_adjustments_reject_conflicting_direction(plugin, value):
    with pytest.raises(ValueError, match="冲突"):
        plugin.nodes.build_instruction("pitch", value)


@pytest.mark.parametrize(("value", "expected"), [("1.5", "将语速调整为1.5倍。"), ("2x", "将语速调整为2.0倍。")])
def test_speed_uses_supported_official_multipliers(plugin, value, expected):
    assert plugin.nodes.build_instruction("speed", value) == expected


@pytest.mark.parametrize("value", ["1", "1.8", "快一点"])
def test_speed_rejects_ambiguous_or_unsupported_values(plugin, value):
    with pytest.raises(ValueError, match="速度倍率"):
        plugin.nodes.build_instruction("speed", value)


def test_nonverbal_requests_are_canonicalized_and_unknown_events_rejected(plugin):
    assert plugin.nodes.build_instruction("nonverbal", "在开头增加笑声") == "在语音开头增加笑声。"
    assert plugin.nodes.build_instruction("nonverbal", "删除全部呼吸") == "删除音频中所有的呼吸声。"
    assert plugin.nodes.build_instruction("nonverbal", "在“欢迎回来”后增加叹气") == "在“欢迎回来”后增加叹气声。"
    with pytest.raises(ValueError, match="未识别非语言声音"):
        plugin.nodes.build_instruction("nonverbal", "在开头增加火车声")
    with pytest.raises(ValueError, match="明确删除"):
        plugin.nodes.build_instruction("nonverbal", "增加笑声")


def test_quality_cleanup_is_only_added_when_explicit(plugin):
    assert plugin.nodes.build_instruction("quality", "去掉电话感") == (
        "This audio suffers from limited bandwidth. Please restore it to a wideband, clear-sounding speech."
    )
    assert plugin.nodes.build_instruction("quality", "去掉电话感并去噪") == (
        "请消除这段音频的电话带宽感，同时完成去噪，输出正常带宽的干净人声。"
    )


def test_replacement_trailing_quote_and_period_do_not_leak(plugin):
    instruction = plugin.nodes.build_instruction(
        "content_edit", "Replace 'old words' with 'new words'.",
    )
    assert instruction == "Replace 'old words' with 'new words'."


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


def test_tts_auto_duration_matches_verified_short_phrase(plugin):
    engine = FakeEngine()
    result = plugin.nodes.AuKGenerateEdit.execute(
        engine,
        "描述生成语音",
        "一只小猫在叫啊",
        "自然、清晰、温暖",
        3.0,
        42,
        duration_mode=plugin.duration.AUTO_DURATION_MODE,
    )
    metadata = json.loads(result.result[2])
    assert engine.call[2] == 1.7
    assert metadata["generation_seconds"] == 1.7
    assert metadata["requested_generation_seconds"] == 3.0
    assert metadata["duration_strategy"] == "auto_text"


def test_seed_widget_randomizes_after_generation_by_default(plugin):
    schema = plugin.nodes.AuKGenerateEdit.define_schema()
    seed = next(value for value in schema.inputs if value.id == "seed")
    assert seed.control_after_generate == io.ControlAfterGenerate.randomize


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


def test_equal_length_task_uses_source_duration_and_reports_it(plugin):
    engine = FakeEngine()
    source = {"waveform": torch.ones(1, 1, 24_000 * 5), "sample_rate": 24_000}
    result = plugin.nodes.AuKGenerateEdit.execute(
        engine,
        "音高编辑",
        "+1",
        "",
        3.0,
        42,
        input_audio=source,
    )
    assert engine.call[2] < 5.0
    assert pytest.approx(engine.call[2], abs=1e-9) == 5.0
    metadata = json.loads(result.result[2])
    assert pytest.approx(metadata["generation_seconds"], abs=1e-9) == 5.0
    assert metadata["requested_generation_seconds"] == 3.0
    assert metadata["duration_strategy"] == "source"


def test_emotion_task_uses_official_duration_multiplier(plugin):
    engine = FakeEngine()
    source = {"waveform": torch.ones(1, 1, 24_000 * 5), "sample_rate": 24_000}
    result = plugin.nodes.AuKGenerateEdit.execute(
        engine,
        "情绪编辑",
        "悲伤",
        "不会附加到指令",
        3.0,
        42,
        input_audio=source,
    )
    assert engine.call[0][0]["content"][0]["text"].startswith("将情感转变为悲伤。")
    assert pytest.approx(engine.call[2], abs=1e-9) == 6.1
    metadata = json.loads(result.result[2])
    assert pytest.approx(metadata["generation_seconds"], abs=1e-9) == 6.1
    assert metadata["duration_strategy"] == "emotion"


def test_manual_duration_task_keeps_requested_duration(plugin):
    engine = FakeEngine()
    source = {"waveform": torch.ones(1, 1, 24_000 * 5), "sample_rate": 24_000}
    plugin.nodes.AuKGenerateEdit.execute(
        engine,
        "参考声音克隆",
        "测试文本",
        "",
        2.5,
        42,
        input_audio=source,
        duration_mode=plugin.duration.MANUAL_DURATION_MODE,
    )
    assert engine.call[2] == 2.5


def test_speed_task_scales_source_duration(plugin):
    engine = FakeEngine()
    source = {"waveform": torch.ones(1, 1, 24_000 * 5), "sample_rate": 24_000}
    plugin.nodes.AuKGenerateEdit.execute(
        engine,
        "速度编辑",
        "2x",
        "",
        3.0,
        42,
        input_audio=source,
    )
    assert engine.call[2] < 2.5
    assert pytest.approx(engine.call[2], abs=1e-9) == 2.5


def test_source_duration_matches_resampled_latent_frame_count(plugin):
    engine = FakeEngine()
    source = (torch.zeros(1, 321), 16_000)
    assert plugin.runtime.source_latent_frames(engine, source) == 1
    seconds = plugin.runtime.source_aligned_seconds(engine, source)
    assert seconds < 0.02
    assert plugin.runtime.math.ceil(seconds * engine.target_sample_rate / engine.downsample_rate) == 1


def test_official_equal_length_tasks_are_marked_source_aligned(plugin):
    expected = {
        "pitch",
        "volume",
        "timbre",
        "deaccent",
        "whisper",
        "enhance",
        "quality",
        "speech_separate",
        "music_separate",
        "target_speaker",
    }
    actual = {task.key for task in plugin.nodes.TASKS if task.duration_strategy == "source"}
    assert actual == expected
    assert next(task for task in plugin.nodes.TASKS if task.key == "emotion").duration_strategy == "emotion"
    assert {
        task.key for task in plugin.nodes.TASKS if task.duration_strategy == "content"
    } == {"content_edit", "lyric_edit"}
    assert next(task for task in plugin.nodes.TASKS if task.key == "nonverbal").duration_strategy == "nonverbal"


def test_content_and_nonverbal_tasks_use_official_duration_rules(plugin):
    engine = FakeEngine()
    source = {"waveform": torch.ones(1, 1, 24_000 * 10), "sample_rate": 24_000}
    content = plugin.nodes.AuKGenerateEdit.execute(
        engine,
        "语音文字编辑",
        "把“今天”改成“明天上午”",
        "我们今天开会",
        48.0,
        42,
        input_audio=source,
    )
    assert pytest.approx(engine.call[2], abs=1e-9) == 13.34
    assert json.loads(content.result[2])["duration_strategy"] == "content"

    nonverbal = plugin.nodes.AuKGenerateEdit.execute(
        engine,
        "非语言声音编辑",
        "在语音开头增加笑声",
        "不应发送",
        48.0,
        42,
        input_audio=source,
    )
    assert pytest.approx(engine.call[2], abs=1e-9) == 10.76
    assert json.loads(nonverbal.result[2])["duration_strategy"] == "nonverbal"


def test_automatic_edit_ignores_stale_out_of_range_duration_widget(plugin):
    engine = FakeEngine()
    source = {"waveform": torch.ones(1, 1, 24_000 * 4), "sample_rate": 24_000}
    result = plugin.nodes.AuKGenerateEdit.execute(
        engine,
        "去口音",
        "去掉方言口音",
        "",
        48.0,
        42,
        input_audio=source,
    )
    assert engine.call[2] == pytest.approx(4.0)
    assert json.loads(result.result[2])["requested_generation_seconds"] == 48.0


def test_sequence_limit_checks_source_and_target_independently(plugin):
    engine = FakeEngine()
    source = (torch.zeros(1, 24_000 * 30), 24_000)
    plugin.nodes.validate_sequence_duration(engine, source, 30.0)
    too_long_source = (torch.zeros(1, 24_000 * 31), 24_000)
    with pytest.raises(ValueError, match="30s"):
        plugin.nodes.validate_sequence_duration(engine, too_long_source, 20.0)
    with pytest.raises(ValueError, match="30s"):
        plugin.nodes.validate_sequence_duration(engine, source, 30.1)


def test_sequence_limit_rejects_sub_frame_reference(plugin):
    engine = FakeEngine()
    source = (torch.zeros(1, engine.downsample_rate - 1), engine.target_sample_rate)
    with pytest.raises(ValueError, match="输入音频过短"):
        plugin.nodes.validate_sequence_duration(engine, source, 1.0)


def test_native_engine_seed_covers_reference_encoding_and_restores_rng(plugin):
    engine = plugin.runtime.AuKEngine.__new__(plugin.runtime.AuKEngine)
    engine.device = torch.device("cpu")
    engine.lock = threading.Lock()
    engine.inference = SimpleNamespace(target_sample_rate=24_000)
    engine.get_models = list
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


def test_fp32_cuda_setting_does_not_enter_unsupported_autocast(plugin):
    engine = plugin.runtime.AuKEngine.__new__(plugin.runtime.AuKEngine)
    engine.device = torch.device("cuda:0")
    engine.inference = SimpleNamespace(dtype=torch.float32)
    assert isinstance(engine._autocast(), nullcontext)


def test_unload_releases_selected_non_default_cuda_device(plugin, monkeypatch):
    calls = []
    monkeypatch.setattr(
        plugin.runtime.model_management,
        "unload_model_and_clones",
        lambda patcher, **kwargs: calls.append((patcher, kwargs)),
    )
    patcher = SimpleNamespace(load_device=torch.device("cuda:1"), model=None)
    plugin.runtime.AuKEngine._unload(patcher)
    assert calls == [(patcher, {"all_devices": True})]


def test_cpu_components_bypass_comfy_gpu_model_registry(plugin, monkeypatch):
    load_calls = []
    unload_calls = []
    monkeypatch.setattr(
        plugin.runtime.model_management,
        "load_models_gpu",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        plugin.runtime.model_management,
        "unload_model_and_clones",
        lambda *args, **kwargs: unload_calls.append((args, kwargs)),
    )
    engine = plugin.runtime.AuKEngine.__new__(plugin.runtime.AuKEngine)
    patcher = SimpleNamespace(load_device=torch.device("cpu"), model=torch.nn.Identity())
    engine._load(patcher)
    engine._unload(patcher)
    assert load_calls == []
    assert unload_calls == []


def test_reference_load_failure_still_requests_unload(plugin):
    engine = plugin.runtime.AuKEngine.__new__(plugin.runtime.AuKEngine)
    engine.device = torch.device("cpu")
    engine.vae_patcher = SimpleNamespace(load_device=torch.device("cpu"), model=torch.nn.Identity())
    engine.inference = SimpleNamespace(latent_dim=64, target_sample_rate=24_000, downsample_rate=480)
    unloaded = []
    engine._load = lambda _patcher: (_ for _ in ()).throw(RuntimeError("load failed"))
    engine._unload = lambda patcher: unloaded.append(patcher)

    with pytest.raises(RuntimeError, match="load failed"):
        engine._encode_reference((torch.zeros(1, 480), 24_000), lambda _phase: None)
    assert unloaded == [engine.vae_patcher]


def test_runtime_only_accepts_safetensors(plugin):
    root = Path(plugin.__file__).parent / "auk_core"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "torch.load(" not in source


def test_vae_checkpoint_mismatch_is_fatal(plugin, tmp_path, monkeypatch):
    import safetensors.torch

    vae_module = importlib.import_module(f"{plugin.__name__}.auk_core.model.vae")
    monkeypatch.setattr(safetensors.torch, "load_file", lambda *_args, **_kwargs: {"wrong": torch.ones(1)})
    checkpoint = tmp_path / "vae.safetensors"
    checkpoint.touch()
    with pytest.raises(RuntimeError, match="does not match"):
        vae_module.load_ckpt(torch.nn.Linear(1, 1), str(checkpoint))


def test_auk_checkpoint_mismatch_is_fatal(plugin, tmp_path, monkeypatch):
    import safetensors.torch

    infer_module = importlib.import_module(f"{plugin.__name__}.auk_core.infer.infer_auk")
    monkeypatch.setattr(safetensors.torch, "load_file", lambda *_args, **_kwargs: {"wrong": torch.ones(1)})
    checkpoint = tmp_path / "auk.safetensors"
    checkpoint.touch()
    inference = infer_module.AukInfer.__new__(infer_module.AukInfer)
    inference.device = "cpu"
    with pytest.raises(RuntimeError, match="does not match"):
        inference._load_ema_weights(torch.nn.Linear(1, 1), str(checkpoint))


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


def test_model_paths_support_split_extra_auk_roots(plugin, tmp_path, monkeypatch):
    default_root = tmp_path / "default"
    model_root = tmp_path / "model-extra"
    qwen_root = tmp_path / "qwen-extra"
    manifest = {
        "models": {
            "AuK-Flash": {
                "files": {
                    "auk_flash.safetensors": {"size": 1},
                    "vae.safetensors": {"size": 1},
                    "config.yaml": {"size": 1},
                }
            },
            "Qwen2.5-Omni-3B": {"files": {"config.json": {"size": 1}}},
        }
    }
    monkeypatch.setattr(plugin.nodes, "MODEL_ROOT", default_root)
    monkeypatch.setattr(plugin.nodes, "load_manifest", lambda: manifest)
    monkeypatch.setattr(
        plugin.nodes.folder_paths,
        "get_folder_paths",
        lambda _name: [str(default_root), str(model_root), str(qwen_root)],
    )
    for relative in ("AuK-Flash/auk_flash.safetensors", "AuK-Flash/vae.safetensors", "AuK-Flash/config.yaml"):
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    qwen_file = qwen_root / "Qwen2.5-Omni-3B/config.json"
    qwen_file.parent.mkdir(parents=True, exist_ok=True)
    qwen_file.write_bytes(b"x")

    checkpoint, config, qwen = plugin.nodes.resolve_model_files("AuK-Flash")
    assert checkpoint == model_root / "AuK-Flash/auk_flash.safetensors"
    assert config == model_root / "AuK-Flash/config.yaml"
    assert qwen == qwen_root / "Qwen2.5-Omni-3B"


def test_example_workflows_only_use_native_node_ids(plugin):
    root = Path(plugin.__file__).parent / "example_workflows"
    paths = sorted(root.glob("*.json"))
    assert len(paths) == len(plugin.nodes.TASKS) == 17
    workflow_tasks = set()
    for path in paths:
        workflow = json.loads(path.read_text(encoding="utf-8"))
        node_ids = {node["type"] for node in workflow["nodes"]}
        assert "AuKModelLoader" in node_ids
        assert "AuKGenerateEdit" in node_ids
        assert not {"AuKLocalConnection", "AuKLocalGenerateEdit"} & node_ids
        assert "SaveAudio" in node_ids
        assert "SaveAudioAdvanced" not in node_ids
        loader = next(node for node in workflow["nodes"] if node["type"] == "AuKModelLoader")
        generator = next(node for node in workflow["nodes"] if node["type"] == "AuKGenerateEdit")
        assert loader["widgets_values"][0] == "AuK Base"
        assert len(generator["widgets_values"]) == 10
        label, primary, secondary = generator["widgets_values"][:3]
        template = plugin.nodes.TASK_BY_LABEL[label]
        plugin.nodes.build_instruction(template.key, primary, secondary)
        workflow_tasks.add(label)
        if template.needs_audio:
            assert "LoadAudio" in node_ids
    assert workflow_tasks == {task.label for task in plugin.nodes.TASKS}


def test_downloader_pins_the_mirror_revision(plugin, tmp_path, monkeypatch):
    root = Path(plugin.__file__).parent
    spec = importlib.util.spec_from_file_location("auk_download_models_test", root / "download_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received: dict[str, object] = {}
    verified = []

    def fake_download(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr(module, "snapshot_download", fake_download)
    monkeypatch.setattr(module, "verify", lambda *args: verified.append(args))
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--model-root", str(tmp_path)])
    module.main()

    manifest = json.loads((root / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    assert received["repo_id"] == "t8star/Auk-Comfy"
    assert received["revision"] == manifest["repository_revision"]
    assert verified[0][2] is True
