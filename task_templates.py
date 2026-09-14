from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskTemplate:
    key: str
    category: str
    label: str
    needs_audio: bool
    primary_label: str
    secondary_label: str


TASKS: tuple[TaskTemplate, ...] = (
    TaskTemplate("instruct_tts", "语音生成", "描述生成语音", False, "目标文本", "声音描述"),
    TaskTemplate("zero_shot_tts", "语音生成", "参考声音克隆", True, "目标文本", "参考音频文字（可选）"),
    TaskTemplate("content_edit", "音频编辑", "语音文字编辑", True, "编辑要求", "原文/目标文字（可选）"),
    TaskTemplate("lyric_edit", "音频编辑", "歌词编辑", True, "歌词修改要求", "原歌词/新歌词（可选）"),
    TaskTemplate("pitch", "音频编辑", "音高编辑", True, "半音变化", "附加要求（可选）"),
    TaskTemplate("speed", "音频编辑", "速度编辑", True, "速度倍率", "附加要求（可选）"),
    TaskTemplate("volume", "音频编辑", "音量编辑", True, "分贝变化", "附加要求（可选）"),
    TaskTemplate("emotion", "音频编辑", "情绪编辑", True, "目标情绪", "附加要求（可选）"),
    TaskTemplate("timbre", "音频编辑", "音色编辑", True, "目标音色描述", "附加要求（可选）"),
    TaskTemplate("deaccent", "音频编辑", "去口音", True, "去口音要求", "附加要求（可选）"),
    TaskTemplate("nonverbal", "音频编辑", "非语言声音编辑", True, "添加/移除要求", "定位文字（可选）"),
    TaskTemplate("whisper", "音频编辑", "耳语转换", True, "转换方向", "附加要求（可选）"),
    TaskTemplate("enhance", "修复与分离", "语音增强", True, "修复要求", "附加要求（可选）"),
    TaskTemplate("speech_separate", "修复与分离", "说话人分离", True, "保留对象", "按顺序或内容描述"),
    TaskTemplate("music_separate", "修复与分离", "音乐人声提取", True, "保留内容", "附加要求（可选）"),
    TaskTemplate("target_speaker", "修复与分离", "指定说话人提取", True, "目标说出的内容", "附加要求（可选）"),
)

TASK_BY_KEY = {task.key: task for task in TASKS}
TASK_BY_LABEL = {task.label: task for task in TASKS}


def build_instruction(task_key: str, primary: str, secondary: str = "") -> str:
    primary = str(primary or "").strip()
    secondary = str(secondary or "").strip()
    if task_key not in TASK_BY_KEY:
        raise ValueError(f"未知任务：{task_key}")
    if not primary:
        raise ValueError("主要内容不能为空")
    templates = {
        "instruct_tts": f'请基于下面的声音描述："{secondary or "自然、清晰的声音"}"，生成语音内容："{primary}"。',
        "zero_shot_tts": (
            f'请使用参考音频中相同的声音说："{primary}"。'
            + (f'参考音频的文字内容为："{secondary}"。' if secondary else "")
        ),
        "content_edit": primary if not secondary else f"{primary}。文字信息：{secondary}",
        "lyric_edit": primary if not secondary else f"{primary}。歌词信息：{secondary}",
        "pitch": f"将音调调整 {primary} 个半音。{secondary}",
        "speed": f"将语速调整为 {primary} 倍。{secondary}",
        "volume": f"将音量调整 {primary} 分贝。{secondary}",
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
