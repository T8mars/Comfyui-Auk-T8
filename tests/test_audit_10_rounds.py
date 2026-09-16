from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_nodes import FakeEngine


def test_automatic_duration_mode_is_visible_and_default(plugin):
    schema = plugin.nodes.AuKGenerateEdit.define_schema()
    mode = next(value for value in schema.inputs if value.id == "duration_mode")
    assert mode.default == plugin.duration.AUTO_TASK_DURATION_MODE
    assert not mode.advanced
    assert "适配" in mode.display_name
    assert plugin.duration.MANUAL_DURATION_MODE in mode.options
    assert schema.outputs[-1].io_type == "FLOAT"


@pytest.mark.parametrize(("task", "primary", "expected"), [
    ("描述生成语音", "你好，欢迎使用。", 2.0),
    ("参考声音克隆", "你好，欢迎使用。", 2.0),
    ("语音文字编辑", "把“今天”改成“明天”", 4.0),
    ("歌词编辑", "把歌词“今天”改成“明天”", 4.0),
    ("音高编辑", "+1", 4.0), ("速度编辑", "1.25", 3.2), ("音量编辑", "+5", 4.0),
    ("情绪编辑", "悲伤", 4.88), ("音色编辑", "低沉磁性的男声", 4.0),
    ("去口音", "去掉方言口音", 4.0), ("非语言声音编辑", "在开头增加笑声", 4.76),
    ("耳语转换", "转换成耳语", 4.0), ("语音增强", "去噪并去混响", 4.0),
    ("音质修复", "补充高频并提升清晰度", 4.0), ("说话人分离", "第一个开始说话的人", 4.0),
    ("音乐人声提取", "只保留歌声", 4.0), ("指定说话人提取", "今天下午开会", 4.0),
])
def test_all_17_tasks_ignore_stale_float_and_use_task_duration(plugin, monkeypatch, task, primary, expected):
    monkeypatch.setattr(plugin.nodes, "prepare_model_audio", lambda waveform, rate, *args: (
        waveform, {"speech_seconds_unpadded": waveform.shape[-1] / rate},
    ))
    engine = FakeEngine()
    result = plugin.nodes.AuKGenerateEdit.execute(
        engine, task, primary, "", 12.4, 42,
        input_audio={"waveform": torch.zeros(1, 1, 4 * 24_000), "sample_rate": 24_000},
    ).result
    assert result[3] == pytest.approx(expected)
    metadata = json.loads(result[2])
    assert metadata["generation_seconds"] == pytest.approx(expected)
    assert engine.call[2] == pytest.approx(expected)


@pytest.mark.parametrize("bad_audio", [
    {"waveform": torch.ones(1, 1, 10), "sample_rate": True},
    {"waveform": torch.ones(1, 1, 10), "sample_rate": 0},
    {"waveform": torch.ones(2, 1, 10), "sample_rate": 24000},
    {"waveform": torch.full((1, 1, 10), float("nan")), "sample_rate": 24000},
])
def test_trim_rejects_invalid_audio_contract(plugin, bad_audio):
    with pytest.raises(ValueError):
        plugin.nodes.AuKAudioTrim.execute(bad_audio)


def test_trim_reports_missing_audio_clearly(plugin):
    with pytest.raises(ValueError, match="连接原始音频"):
        plugin.nodes.AuKAudioTrim.execute(None)


def test_peak_protection_preserves_normal_and_whisper_levels_and_prevents_export_clipping(plugin):
    for peak in (.8, .0064):
        original = torch.tensor([[-peak, 0, peak]])
        output, info = plugin.preprocess.protect_audio_output(original)
        assert output is original and not info["applied"]
    original = torch.tensor([[-2., -.5, 0., 1.]])
    output, info = plugin.preprocess.protect_audio_output(original)
    assert output.abs().max() == pytest.approx(.99)
    assert info["applied"] and info["gain_db"] < 0
    assert torch.equal(original, torch.tensor([[-2., -.5, 0., 1.]]))
    assert output[0, 1] / output[0, 0] == pytest.approx(.25)


def test_native_decoder_preserves_raw_peaks_until_output_protection(plugin):
    module = importlib.import_module(f"{plugin.__name__}.auk_core.model.vae.bigvgan_flow_vae")
    vae = module.BigVGANFlowVAE.__new__(module.BigVGANFlowVAE)
    torch.nn.Module.__init__(vae)
    vae.h = SimpleNamespace(latent_dim=1)
    vae.num_upsamples = 0
    vae.conv_pre = vae.activation_post = vae.conv_post = torch.nn.Identity()
    vae.denormalize = lambda value: value
    raw = torch.tensor([[[-2., -.5, 0, 2.]]])
    assert vae.inference_from_latents(raw).abs().max() == 1  # Upstream default unchanged.
    engine = plugin.runtime.AuKEngine.__new__(plugin.runtime.AuKEngine)
    engine.device = torch.device("cpu")
    engine.inference = SimpleNamespace(vae_model=vae)
    engine.vae_patcher = object()
    engine._load = engine._unload = lambda patcher: None
    decoded = engine._decode(raw.permute(0, 2, 1), lambda phase: None)
    assert torch.equal(decoded, raw[0])
    protected, info = plugin.preprocess.protect_audio_output(decoded)
    assert info["applied"] and protected.abs().max() == pytest.approx(.99)
    assert protected[0, 1] / protected[0, 0] == pytest.approx(.25)


@pytest.mark.parametrize("waveform", [torch.empty(1, 0), torch.zeros(1, 1, 2), torch.tensor([[float("nan")]])])
def test_invalid_model_output_is_reported_instead_of_published_as_audio(plugin, waveform):
    with pytest.raises(ValueError, match="模型输出"):
        plugin.preprocess.protect_audio_output(waveform)


@pytest.mark.parametrize(("start", "end", "frames"), [(0, 4, 40), (1.2, 3.4, 22), (2, 0, 460), (47, 60, 10)])
def test_trim_is_sample_exact_and_preserves_channels(plugin, start, end, frames):
    waveform = torch.arange(960, dtype=torch.float32).reshape(1, 2, 480)
    original = waveform.clone()
    audio, seconds, note = plugin.nodes.AuKAudioTrim.execute(
        {"waveform": waveform, "sample_rate": 10}, start, end,
    ).result
    assert audio["sample_rate"] == 10
    assert audio["waveform"].shape == (1, 2, frames)
    assert seconds == frames / 10
    assert "实际输入" in note
    assert torch.equal(audio["waveform"], original[..., round(start * 10):round(start * 10) + frames])
    audio["waveform"].zero_()
    assert torch.equal(waveform, original)


def test_extreme_finite_crop_values_report_or_clamp_without_overflow(plugin):
    audio = {"waveform": torch.ones(1, 1, 40), "sample_rate": 10}
    assert plugin.nodes.AuKAudioTrim.execute(audio, 0, 1e308).result[1] == 4
    with pytest.raises(ValueError, match="开始超出"):
        plugin.nodes.AuKAudioTrim.execute(audio, 1e308, 0)


@pytest.mark.parametrize(("start", "end"), [(-1, 2), (0, -1), (float("nan"), 2), (0, float("inf")), (2, 1), (4, 0), (0, .0001)])
def test_invalid_trim_fails_without_altering_source(plugin, start, end):
    source = torch.ones(1, 1, 40)
    with pytest.raises(ValueError):
        plugin.nodes.AuKAudioTrim.execute({"waveform": source, "sample_rate": 10}, start, end)
    assert source.sum() == 40


def test_trimmed_48_second_input_uses_four_seconds_in_edit(plugin):
    source = {"waveform": torch.zeros(1, 2, 48 * 24_000), "sample_rate": 24_000}
    cropped, _, _ = plugin.nodes.AuKAudioTrim.execute(source, 0, 4).result
    engine = FakeEngine()
    output = plugin.nodes.AuKGenerateEdit.execute(
        engine, "速度编辑", "1.25", "", 48, 42, input_audio=cropped,
    ).result
    metadata = json.loads(output[2])
    assert metadata["original_input_seconds"] == 4
    assert output[3] == pytest.approx(3.2)
    assert engine.call[1][0].shape[-1] == 4 * 24_000
    assert source["waveform"].shape[-1] == 48 * 24_000


def test_recut_uses_original_load_audio_without_reupload(plugin):
    source = {"waveform": torch.rand(1, 2, 480), "sample_rate": 10}
    a = plugin.nodes.AuKAudioTrim.execute(source, 0, 4).result[0]
    b = plugin.nodes.AuKAudioTrim.execute(source, 4, 8).result[0]
    assert torch.equal(a["waveform"], source["waveform"][..., :40])
    assert torch.equal(b["waveform"], source["waveform"][..., 40:80])


def test_long_tts_reports_duration_instead_of_silently_truncating(plugin):
    with pytest.raises(ValueError, match="分段"):
        plugin.duration.estimate_tts_seconds("欢迎使用。" * 100)


def test_exact_30_seconds_accepts_internal_padding_and_rejects_one_extra_sample(plugin):
    engine = FakeEngine()
    prepared = (torch.zeros(1, round(30.2 * 24_000)), 24_000)
    plugin.nodes.validate_sequence_duration(engine, prepared, 30, input_seconds=30)
    with pytest.raises(ValueError, match="裁剪"):
        plugin.nodes.validate_sequence_duration(engine, (torch.zeros(1, 30 * 24_000 + 1), 24_000), 3)


def test_over_limit_raw_input_is_rejected_before_vad_can_hide_it(plugin, monkeypatch):
    def forbidden(*args):
        pytest.fail("VAD must not hide an over-limit original input")

    monkeypatch.setattr(plugin.nodes, "prepare_model_audio", forbidden)
    with pytest.raises(ValueError, match="裁剪"):
        plugin.nodes.AuKGenerateEdit.execute(
            FakeEngine(), "音高编辑", "+1", "", 1, 42,
            input_audio={"waveform": torch.zeros(1, 1, 31 * 24_000), "sample_rate": 24_000},
        )


@pytest.mark.parametrize(("field", "value"), [
    ("seed", -1), ("seed", 1.5), ("seed", True), ("nfe_steps", 3), ("nfe_steps", 65),
    ("cfg_strength", float("nan")), ("cfg_strength", 6), ("sway_sampling_coef", float("inf")),
])
def test_invalid_backend_parameters_fail_before_model_generation(plugin, field, value):
    kwargs = {"engine": FakeEngine(), "task": "描述生成语音", "primary": "你好", "secondary": "",
              "generation_seconds": 1, "seed": 42}
    kwargs[field] = value
    with pytest.raises(ValueError):
        plugin.nodes.AuKGenerateEdit.execute(**kwargs)
    assert kwargs["engine"].call is None


def test_legacy_manual_tts_mode_still_uses_connected_float(plugin):
    result = plugin.nodes.AuKGenerateEdit.execute(
        FakeEngine(), "描述生成语音", "你好", "", 12.4, 42,
        duration_mode=plugin.duration.MANUAL_DURATION_MODE,
    ).result
    assert result[3] == 12.4


def test_nonfinite_stale_widget_is_not_written_as_invalid_json(plugin):
    result = plugin.nodes.AuKGenerateEdit.execute(FakeEngine(), "描述生成语音", "你好", "", float("nan"), 42).result
    metadata = json.loads(result[2], parse_constant=lambda value: pytest.fail(value))
    assert metadata["requested_generation_seconds"] is None


def test_all_workflow_links_are_consistent_and_audio_routes_through_trim(plugin):
    root = Path(plugin.__file__).parent / "example_workflows"
    for path in root.glob("*.json"):
        workflow = json.loads(path.read_text(encoding="utf8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        for link_id, source, source_slot, target, target_slot, kind in workflow["links"]:
            output = nodes[source]["outputs"][source_slot]
            input_ = nodes[target]["inputs"][target_slot]
            assert link_id in output["links"]
            assert input_["link"] == link_id
            assert input_["type"] == output["type"] == kind
        generator = next(node for node in nodes.values() if node["type"] == "AuKGenerateEdit")
        assert generator["widgets_values"][-1] == plugin.duration.AUTO_TASK_DURATION_MODE
        task = plugin.nodes.TASK_BY_LABEL[generator["widgets_values"][0]]
        if task.needs_audio:
            link = next(link for link in workflow["links"] if link[0] == generator["inputs"][1]["link"])
            assert nodes[link[1]]["type"] == "AuKAudioTrim"


def test_workflow_generator_is_idempotent(plugin):
    root = Path(plugin.__file__).parent
    def hashes():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "example_workflows").glob("*.json")}
    before = hashes()
    subprocess.run([sys.executable, str(root / "tools/generate_example_workflows.py")], check=True)
    assert hashes() == before
