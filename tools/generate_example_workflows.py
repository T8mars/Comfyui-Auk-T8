from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "example_workflows"

TASKS = (
    ("01", "描述生成语音", False, "你好，欢迎使用 AuK ComfyUI 原生节点。", "自然、清晰、温暖的年轻女性声音", "描述生成"),
    ("02", "参考声音克隆", True, "你好，这是使用参考声音生成的测试音频。", "", "声音克隆"),
    ("03", "语音文字编辑", True, "把“今天下午开会”改成“明天上午开会”", "今天下午开会，请大家准时参加。", "语音文字编辑"),
    ("04", "歌词编辑", True, "把歌词“明天你好”改成“未来你好”", "明天你好，声音多渺小。", "歌词编辑"),
    ("05", "音高编辑", True, "+1", "", "音高编辑"),
    ("06", "速度编辑", True, "1.25", "", "速度编辑"),
    ("07", "音量编辑", True, "+5", "", "音量编辑"),
    ("08", "情绪编辑", True, "悲伤", "", "情绪编辑"),
    ("09", "音色编辑", True, "低沉磁性的年轻男声", "", "音色编辑"),
    ("10", "去口音", True, "去掉方言口音，转换成标准普通话", "", "去口音"),
    ("11", "非语言声音编辑", True, "在“欢迎回来”后增加笑声", "", "非语言声音编辑"),
    ("12", "耳语转换", True, "转换成耳语", "", "耳语转换"),
    ("13", "语音增强", True, "去噪并去除房间混响", "", "语音增强"),
    ("14", "音质修复", True, "补充高频并提升清晰度", "", "音质修复"),
    ("15", "说话人分离", True, "第一个开始说话的人", "", "说话人分离"),
    ("16", "音乐人声提取", True, "只保留歌声，去掉说话和伴奏", "", "音乐人声提取"),
    ("17", "指定说话人提取", True, "欢迎大家来到今天的节目", "", "指定说话人提取"),
)


def main() -> None:
    no_audio = json.loads((WORKFLOW_DIR / "AuK-01-描述生成语音.json").read_text(encoding="utf-8"))
    with_audio = json.loads((WORKFLOW_DIR / "AuK-03-语音文字编辑.json").read_text(encoding="utf-8"))
    for path in WORKFLOW_DIR.glob("AuK-*.json"):
        path.unlink()
    for number, label, needs_audio, primary, secondary, save_name in TASKS:
        workflow = copy.deepcopy(with_audio if needs_audio else no_audio)
        loader = next(node for node in workflow["nodes"] if node["type"] == "AuKModelLoader")
        generator = next(node for node in workflow["nodes"] if node["type"] == "AuKGenerateEdit")
        saver = next(node for node in workflow["nodes"] if node["type"] == "SaveAudio")
        loader["widgets_values"] = ["AuK Base", "auto", "auto"]
        generator["widgets_values"] = [
            label,
            primary,
            secondary,
            3.0,
            42,
            "randomize",
            32,
            2.0,
            -1.0,
            "自动估算（TTS 推荐）",
        ]
        saver["widgets_values"] = [f"auk/{save_name}"]
        if needs_audio:
            audio_loader = next(node for node in workflow["nodes"] if node["type"] == "LoadAudio")
            audio_loader["widgets_values"] = ["auk_input.wav"]
        path = WORKFLOW_DIR / f"AuK-{number}-{label}.json"
        path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
