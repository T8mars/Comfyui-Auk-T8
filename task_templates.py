from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskTemplate:
    key: str
    category: str
    label: str
    needs_audio: bool
    primary_label: str
    secondary_label: str
    duration_strategy: str = "manual"


TASKS: tuple[TaskTemplate, ...] = (
    TaskTemplate("instruct_tts", "语音生成", "描述生成语音", False, "目标文本", "声音描述"),
    TaskTemplate("zero_shot_tts", "语音生成", "参考声音克隆", True, "目标文本", "参考音频文字（可选）"),
    TaskTemplate("content_edit", "音频编辑", "语音文字编辑", True, "编辑要求", "原文/目标文字（可选）"),
    TaskTemplate("lyric_edit", "音频编辑", "歌词编辑", True, "歌词修改要求", "原歌词/新歌词（可选）", "source"),
    TaskTemplate("pitch", "音频编辑", "音高编辑", True, "半音变化（+1/+2/+3 或 -1/-2/-3）", "附加要求（可选）", "source"),
    TaskTemplate("speed", "音频编辑", "速度编辑", True, "速度倍率（0.5/0.75/1.25/1.5/2.0）", "附加要求（可选）", "speed"),
    TaskTemplate("volume", "音频编辑", "音量编辑", True, "分贝变化（+5/+10/+15 或 -5/-10/-15）", "附加要求（可选）", "source"),
    TaskTemplate("emotion", "音频编辑", "情绪编辑", True, "目标情绪", "附加要求（可选）", "source"),
    TaskTemplate("timbre", "音频编辑", "音色编辑", True, "目标音色描述", "附加要求（可选）", "source"),
    TaskTemplate("deaccent", "音频编辑", "去口音", True, "去口音要求", "附加要求（可选）", "source"),
    TaskTemplate("nonverbal", "音频编辑", "非语言声音编辑", True, "添加/移除要求", "定位文字（可选）"),
    TaskTemplate("whisper", "音频编辑", "耳语转换", True, "转换方向", "附加要求（可选）", "source"),
    TaskTemplate("enhance", "修复与分离", "语音增强", True, "修复要求", "附加要求（可选）", "source"),
    TaskTemplate("speech_separate", "修复与分离", "说话人分离", True, "保留对象", "按顺序或内容描述", "source"),
    TaskTemplate("music_separate", "修复与分离", "音乐人声提取", True, "保留内容", "附加要求（可选）", "source"),
    TaskTemplate("target_speaker", "修复与分离", "指定说话人提取", True, "目标说出的内容", "附加要求（可选）", "source"),
)

TASK_BY_KEY = {task.key: task for task in TASKS}
TASK_BY_LABEL = {task.label: task for task in TASKS}


def _signed_adjustment(value: str, *, allowed: tuple[int, ...], kind: str, unit: str) -> str:
    text = str(value or "").strip().replace("＋", "+").replace("－", "-")
    direction_words = {
        "increase": ("升高", "提高", "增加", "调高", "调大"),
        "decrease": ("降低", "减少", "调低", "调小"),
    }
    direction = None
    for candidate, words in direction_words.items():
        if any(word in text for word in words):
            if direction is not None and direction != candidate:
                raise ValueError(f"{kind}方向互相冲突：{value!r}")
            direction = candidate

    match = re.search(r"[+-]?\d+(?:\.0+)?", text)
    if match is None:
        choices = "/".join(str(item) for item in allowed)
        raise ValueError(f"{kind}请输入带方向的数值，只支持 ±{choices}{unit}")
    remainder = (text[: match.start()] + text[match.end() :]).strip()
    for word in (*direction_words["increase"], *direction_words["decrease"], "个半音", "半音", "分贝", "dB", "db"):
        remainder = remainder.replace(word, "")
    if remainder.strip(" ，,。"):
        raise ValueError(f"无法识别{kind}数值：{value!r}")

    numeric_text = match.group()
    numeric = float(numeric_text)
    if numeric == 0:
        raise ValueError(f"{kind}不能为 0（不会产生变化）")
    sign_direction = "decrease" if numeric < 0 else "increase"
    has_explicit_sign = numeric_text.startswith(("+", "-"))
    if direction is not None and has_explicit_sign and direction != sign_direction:
        raise ValueError(f"{kind}方向与数值符号冲突：{value!r}")
    direction = direction or sign_direction
    magnitude = abs(numeric)
    if magnitude not in allowed:
        choices = "/".join(str(item) for item in allowed)
        raise ValueError(f"{kind}只支持 {choices}{unit}，当前为 {magnitude:g}{unit}")
    verb = "升高" if direction == "increase" else "降低"
    return f"将{kind}{verb}{magnitude:g}{unit}。"


def parse_speed_multiplier(value: str) -> float:
    text = str(value or "").strip().replace("×", "x")
    match = re.fullmatch(r"(0\.5|0\.75|1\.25|1\.5|2(?:\.0)?)(?:\s*(?:倍|[xX]))?", text)
    if match is None:
        raise ValueError("速度倍率只支持 0.5、0.75、1.25、1.5 或 2.0")
    return float(match.group(1))


def _speed_adjustment(value: str) -> str:
    return f"将语速调整为{parse_speed_multiplier(value):g}倍。"


def build_instruction(task_key: str, primary: str, secondary: str = "") -> str:
    primary = str(primary or "").strip()
    secondary = str(secondary or "").strip()
    if task_key not in TASK_BY_KEY:
        raise ValueError(f"未知任务：{task_key}")
    if not primary:
        raise ValueError("主要内容不能为空")
    if task_key == "pitch":
        return f"{_signed_adjustment(primary, allowed=(1, 2, 3), kind='音调', unit='个半音')}{secondary}".strip()
    if task_key == "volume":
        return f"{_signed_adjustment(primary, allowed=(5, 10, 15), kind='音量', unit='分贝')}{secondary}".strip()
    if task_key == "speed":
        return f"{_speed_adjustment(primary)}{secondary}".strip()
    templates = {
        "instruct_tts": f'请基于下面的描述: "{secondary or "自然、清晰的声音"}",生成语音内容"{primary}".',
        "zero_shot_tts": f'Say the following with the same voice: "{primary}"',
        "content_edit": primary if not secondary else f"{primary}。文字信息：{secondary}",
        "lyric_edit": primary if not secondary else f"{primary}。歌词信息：{secondary}",
        "emotion": f"将这段语音的情绪改为：{primary}，保持内容和说话人声音。{secondary}",
        "timbre": f"保持说话内容不变，将音色改为：{primary}。{secondary}",
        "deaccent": f"{primary}，保持说话人音色和内容一致。{secondary}",
        "nonverbal": f"{primary}。{secondary}",
        "whisper": f"将这段语音转换为{primary}，保持说话人和内容。{secondary}",
        "enhance": f"{primary}，保留原本说话内容。{secondary}",
        "speech_separate": f"仅保留{primary}。{secondary}",
        "music_separate": f"从混合音频中{primary}。{secondary}",
        "target_speaker": f"仅保留说出“{primary}”的说话人。{secondary}",
    }
    return templates[task_key].strip()
