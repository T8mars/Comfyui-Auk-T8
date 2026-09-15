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


@dataclass(frozen=True)
class TaskGuide:
    requirement: str
    example: str
    note: str


TASKS: tuple[TaskTemplate, ...] = (
    TaskTemplate("instruct_tts", "语音生成", "描述生成语音", False, "目标文本", "声音描述", "tts"),
    TaskTemplate("zero_shot_tts", "语音生成", "参考声音克隆", True, "目标文本", "声音克隆无需填写", "tts"),
    TaskTemplate(
        "content_edit", "音频编辑", "语音文字编辑", True,
        "编辑要求（一次只改一处）", "原音频完整文字（可选，仅估算时长）", "content",
    ),
    TaskTemplate(
        "lyric_edit", "音频编辑", "歌词编辑", True,
        "歌词修改要求（一次只改一处）", "原音频完整歌词（可选，仅估算时长）", "content",
    ),
    TaskTemplate("pitch", "音频编辑", "音高编辑", True, "半音变化", "附加要求（可选）", "source"),
    TaskTemplate("speed", "音频编辑", "速度编辑", True, "速度倍率", "附加要求（可选）", "speed"),
    TaskTemplate("volume", "音频编辑", "音量编辑", True, "分贝变化", "附加要求（可选）", "source"),
    TaskTemplate("emotion", "音频编辑", "情绪编辑", True, "目标情绪", "附加要求（可选）", "emotion"),
    TaskTemplate("timbre", "音频编辑", "音色编辑", True, "目标音色描述", "附加要求（可选）", "source"),
    TaskTemplate("deaccent", "音频编辑", "去口音", True, "去口音要求", "附加要求（可选）", "source"),
    TaskTemplate(
        "nonverbal", "音频编辑", "非语言声音编辑", True,
        "添加/移除要求", "无需填写", "nonverbal",
    ),
    TaskTemplate("whisper", "音频编辑", "耳语转换", True, "转换方向", "附加要求（可选）", "source"),
    TaskTemplate("enhance", "修复与分离", "语音增强", True, "修复要求", "附加要求（可选）", "source"),
    TaskTemplate("quality", "修复与分离", "音质修复", True, "音质问题或修复要求", "无需填写", "source"),
    TaskTemplate("speech_separate", "修复与分离", "说话人分离", True, "保留对象", "按顺序或内容描述", "source"),
    TaskTemplate("music_separate", "修复与分离", "音乐人声提取", True, "保留内容", "附加要求（可选）", "source"),
    TaskTemplate("target_speaker", "修复与分离", "指定说话人提取", True, "目标说出的内容", "附加要求（可选）", "source"),
)

TASK_BY_KEY = {task.key: task for task in TASKS}
TASK_BY_LABEL = {task.label: task for task in TASKS}

TASK_GUIDES: dict[str, TaskGuide] = {
    "instruct_tts": TaskGuide(
        "不需要音频；填写要念的目标文本和声音描述。",
        "目标文本：欢迎回来，今天辛苦了。｜声音描述：年轻女生，温柔、关心、语速稍慢。",
        "这是按文字描述随机生成声音；要克隆真人音色请选择“参考声音克隆”。",
    ),
    "zero_shot_tts": TaskGuide(
        "必须上传单人、清晰的参考语音；只填写要念的新文本。",
        "目标文本：大家好，欢迎收看今天的节目。",
        "无需填写参考音频原文或声音描述；程序使用 AuK 官方固定克隆提示词。",
    ),
    "content_edit": TaskGuide(
        "必须上传普通说话录音；一次只做一处插入、删除或替换，并写出准确原词。",
        "把“今天下午开会”改成“明天上午开会”",
        "也可写：删掉“那个”；在“你好”后面加上“呀”。完整原文只填到下方可选框，用于估算时长。",
    ),
    "lyric_edit": TaskGuide(
        "必须上传歌唱录音；一次只替换一处歌词，原歌词必须与音频实际唱词一致。",
        "把歌词“明天你好”改成“未来你好”",
        "普通说话录音请使用“语音文字编辑”。完整歌词只填到下方可选框，用于估算时长。",
    ),
    "pitch": TaskGuide(
        "必须上传音频；只支持带方向的 ±1、±2、±3 个半音。",
        "+1（升高一个半音）或 -2（降低两个半音）",
        "0 或其他数值不属于模型官方档位。",
    ),
    "speed": TaskGuide(
        "必须上传音频；只支持 0.5、0.75、1.25、1.5、2.0 倍。",
        "0.75（稍慢）或 1.25（稍快）",
        "1.0 不会产生变化；目标时长会锁定，并按“原音频时长 ÷ 速度倍率”自动计算。",
    ),
    "volume": TaskGuide(
        "必须上传音频；只支持带方向的 ±5、±10、±15 分贝。",
        "+5（增大音量）或 -10（减小音量）",
        "0 dB 不会产生变化。",
    ),
    "emotion": TaskGuide(
        "必须上传说话音频；目标情感限开心、愤怒、悲伤、恐惧、惊讶、厌恶、平静、兴奋。",
        "开心",
        "保持原说话内容和音色，只改变情感。",
    ),
    "timbre": TaskGuide(
        "必须上传说话音频；用文字描述目标音色。",
        "低沉磁性的年轻男声",
        "保持原内容，只改变音色；如要念新文本请选择语音生成任务。",
    ),
    "deaccent": TaskGuide(
        "必须上传带口音或方言腔的说话音频。",
        "去掉方言口音，转换成标准普通话",
        "保持原说话人音色和内容。",
    ),
    "nonverbal": TaskGuide(
        "必须上传音频；一次添加或删除一种笑声、呼吸声、叹气、咳嗽等非语言声音。",
        "在“欢迎回来”后增加笑声",
        "也可写：删除音频中所有呼吸声；在语音开头增加叹气声。",
    ),
    "whisper": TaskGuide(
        "必须上传说话音频；明确选择转为耳语或转回正常说话。",
        "转换成耳语",
        "保持原说话内容和音色。",
    ),
    "enhance": TaskGuide(
        "必须上传说话音频；用于去噪、去底噪和去混响。",
        "去噪并去除房间混响",
        "会保留所有说话人；只保留某个人请使用说话人分离。",
    ),
    "quality": TaskGuide(
        "必须上传说话音频；用于补高频/带宽扩展，或去电话、扩音器、水下闷声等音色缺陷。",
        "补充高频并提升清晰度；或：去掉电话感",
        "普通背景噪声和房间混响请选择“语音增强”。",
    ),
    "speech_separate": TaskGuide(
        "必须上传多说话人音频；按开始顺序指定保留对象。",
        "第一个开始说话的人",
        "输出保留所选说话人并去掉其他说话人。",
    ),
    "music_separate": TaskGuide(
        "必须上传含伴奏的混合音频；明确要保留哪类人声。",
        "只保留歌声，去掉说话和伴奏",
        "如需保留说话和歌唱，可填“保留所有人声，去掉伴奏”。",
    ),
    "target_speaker": TaskGuide(
        "必须上传多说话人音频；填写目标说话人讲过的一小段准确内容。",
        "欢迎大家来到今天的节目",
        "程序用这段内容定位说话人，然后只保留该说话人。",
    ),
}


def _clean_replacement_slot(value: str) -> str:
    return value.strip().strip("\"'“”‘’ ").rstrip("。.!！").strip()


def _quoted_slot(value: str) -> str:
    cleaned = _clean_replacement_slot(value)
    if not cleaned:
        raise ValueError("编辑内容不能为空")
    return cleaned


def _canonical_replacement(value: str, *, lyrics: bool) -> str:
    text = str(value or "").strip()
    if lyrics:
        text = re.sub(r"^(?:把|将)?\s*(?:这段)?歌词(?:中(?:的)?)?\s*", "把", text, count=1)
    patterns = (
        r"(?:把|将)?\s*(.+?)\s*(?:改成|改为|替换成|替换为|换成)\s*(.+)",
        r"(?:replace|change)\s+(.+?)\s+(?:with|to)\s+(.+)",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        original = _clean_replacement_slot(match.group(1))
        replacement = _clean_replacement_slot(match.group(2))
        if not original or not replacement:
            break
        if original == replacement:
            raise ValueError("原词和替换词相同，不会产生变化")
        if lyrics:
            return f"把这段歌词中的“{original}”改成“{replacement}”。"
        return f"把‘{original}’改成‘{replacement}’"
    task_name = "歌词编辑" if lyrics else "语音文字编辑"
    raise ValueError(f"{task_name}格式不正确，请按上方示例填写，并且一次只改一处")


def _canonical_content_edit(value: str) -> str:
    text = str(value or "").strip()
    try:
        return _canonical_replacement(text, lyrics=False)
    except ValueError as replacement_error:
        if "原词和替换词相同" in str(replacement_error):
            raise

    quoted = r"[\"'“‘]?(.+?)[\"'”’]?"
    insert_match = re.fullmatch(
        rf"(?:在)?\s*{quoted}\s*(前面|前|后面|后)\s*(?:加上|加入|添加|插入)\s*{quoted}\s*[。.!！]?",
        text,
    )
    if insert_match is not None:
        anchor = _quoted_slot(insert_match.group(1))
        side = "前面" if insert_match.group(2).startswith("前") else "后面"
        added = _quoted_slot(insert_match.group(3))
        return f"在‘{anchor}’{side}加上‘{added}’"

    anchored_delete = re.fullmatch(
        rf"(?:删掉|删除|去掉)\s*{quoted}\s*(前面|前|后面|后)(?:的|那个)?\s*{quoted}\s*[。.!！]?",
        text,
    )
    if anchored_delete is not None:
        anchor = _quoted_slot(anchored_delete.group(1))
        side = "前" if anchored_delete.group(2).startswith("前") else "后"
        target = _quoted_slot(anchored_delete.group(3))
        return f"删掉‘{anchor}’{side}的‘{target}’"

    delete_match = re.fullmatch(r"(?:删掉|删除|去掉)\s*[\"'“‘]?(.+?)[\"'”’]?\s*[。.!！]?", text)
    if delete_match is not None:
        return f"删掉‘{_quoted_slot(delete_match.group(1))}’"

    raise ValueError("语音文字编辑格式不正确，请按上方替换、插入或删除示例填写，并且一次只改一处")


def _signed_adjustment(value: str, *, allowed: tuple[int, ...], kind: str, unit: str) -> str:
    text = str(value or "").strip().replace("＋", "+").replace("－", "-")
    directions = {
        "increase": ("升高", "提高", "增加", "调高", "调大"),
        "decrease": ("降低", "减少", "调低", "调小"),
    }
    direction = None
    for candidate, words in directions.items():
        if any(word in text for word in words):
            if direction is not None and direction != candidate:
                raise ValueError(f"{kind}方向互相冲突：{value!r}")
            direction = candidate
    match = re.search(r"[+-]?\d+(?:\.0+)?", text)
    if match is None:
        choices = "/".join(str(item) for item in allowed)
        raise ValueError(f"{kind}请输入带方向的数值，只支持 ±{choices}{unit}")
    remainder = (text[: match.start()] + text[match.end() :]).strip()
    for word in (*directions["increase"], *directions["decrease"], "个半音", "半音", "分贝", "dB", "db"):
        remainder = remainder.replace(word, "")
    if remainder.strip(" ，,。"):
        raise ValueError(f"无法识别{kind}数值：{value!r}")
    numeric_text = match.group()
    numeric = float(numeric_text)
    if numeric == 0:
        raise ValueError(f"{kind}不能为 0（不会产生变化）")
    sign_direction = "decrease" if numeric < 0 else "increase"
    if direction is not None and numeric_text.startswith(("+", "-")) and direction != sign_direction:
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


EMOTION_ALIASES = {
        "开心": "开心", "高兴": "开心", "愉快": "开心",
        "愤怒": "愤怒", "生气": "愤怒", "恼怒": "愤怒",
        "悲伤": "悲伤", "难过": "悲伤", "伤心": "悲伤",
        "恐惧": "恐惧", "害怕": "恐惧", "惊讶": "惊讶", "吃惊": "惊讶",
        "厌恶": "厌恶", "嫌弃": "厌恶", "反感": "厌恶",
        "平静": "平静", "冷静": "平静", "淡定": "平静",
        "兴奋": "兴奋", "激动": "兴奋",
}

EMOTION_DURATION_MULTIPLIERS = {
    "悲伤": 1.22,
    "恐惧": 1.16,
    "开心": 1.06,
    "愤怒": 1.06,
    "惊讶": 1.06,
    "厌恶": 1.06,
    "平静": 1.06,
    "兴奋": 1.06,
}


def normalize_emotion(value: str) -> str:
    text = str(value or "").strip().strip("。.!！")
    for alias, label in EMOTION_ALIASES.items():
        if alias in text:
            return label
    raise ValueError("目标情感只支持：开心、愤怒、悲伤、恐惧、惊讶、厌恶、平静、兴奋")


def emotion_duration_multiplier(value: str) -> float:
    return EMOTION_DURATION_MULTIPLIERS[normalize_emotion(value)]


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


def spoken_duration_seconds(value: str | None) -> float:
    """Match AuK's prompt enhancer heuristic for spoken text duration."""
    text = str(value or "")
    duration = len(_CJK_RE.findall(text)) * 0.21 + len(_EN_WORD_RE.findall(text)) * 0.30
    if duration > 0:
        return duration
    return len(re.findall(r"\S", text)) * 0.21


def _content_duration_slots(task_key: str, primary: str) -> tuple[str | None, str | None]:
    instruction = build_instruction(task_key, primary)
    if task_key == "lyric_edit":
        match = re.fullmatch(r"把这段歌词中的“(.+?)”改成“(.+?)”。", instruction)
        if match is None:
            raise ValueError("无法解析歌词编辑要求")
        return match.group(2), match.group(1)

    replace_match = re.fullmatch(r"把‘(.+?)’改成‘(.+?)’", instruction)
    if replace_match is not None:
        return replace_match.group(2), replace_match.group(1)
    insert_match = re.fullmatch(r"在‘.+?’(?:前面|后面)加上‘(.+?)’", instruction)
    if insert_match is not None:
        return insert_match.group(1), None
    delete_match = re.fullmatch(r"删掉(?:‘.+?’[前后]的)?‘(.+?)’", instruction)
    if delete_match is not None:
        return None, delete_match.group(1)
    raise ValueError("无法解析语音文字编辑要求")


def content_scaled_seconds(task_key: str, primary: str, source_seconds: float, transcript: str = "") -> float:
    """Apply AuK's official content-edit duration heuristic without running ASR."""
    if task_key not in {"content_edit", "lyric_edit"}:
        raise ValueError(f"不支持的内容时长任务：{task_key}")
    add_text, delete_text = _content_duration_slots(task_key, primary)
    transcript_duration = spoken_duration_seconds(transcript)
    if transcript_duration > 0:
        edited_duration = transcript_duration
        if add_text:
            edited_duration += spoken_duration_seconds(add_text)
        if delete_text:
            edited_duration -= spoken_duration_seconds(delete_text)
        return source_seconds * max(0.05, edited_duration) / transcript_duration
    if add_text and delete_text:
        original_duration = spoken_duration_seconds(delete_text)
        replacement_duration = spoken_duration_seconds(add_text)
        if original_duration > 0:
            return source_seconds * replacement_duration / original_duration
    return source_seconds


def nonverbal_duration_delta(primary: str) -> float:
    """Apply AuK's official event-family duration adjustment."""
    text = str(primary or "").casefold()
    operation = "delete" if any(word in text for word in ("删除", "删掉", "去掉", "remove", "delete")) else "add"
    families = (
        (("呼吸", "换气", "喘", "breath", "breathing", "pant", "inhale", "exhale"), 0.35, -0.60),
        (("咂嘴", "咂舌", "啧", "吸鼻", "倒吸", "惊喘", "tsk", "smack", "sniff", "gasp"), 0.50, -1.00),
        (("笑", "叹气", "叹息", "咳", "清嗓", "语气", "laugh", "laughter", "chuckle", "sigh", "cough", "throat", "clearing"), 0.75, -1.05),
    )
    for keywords, add_delta, delete_delta in families:
        if any(keyword in text for keyword in keywords):
            return delete_delta if operation == "delete" else add_delta
    return -0.90 if operation == "delete" else 0.55


def _emotion_instruction(value: str) -> str:
    return f"将情感转变为{normalize_emotion(value)}。"


def _whisper_instruction(value: str) -> str:
    text = str(value or "").strip()
    if any(word in text for word in ("正常", "别耳语", "非耳语")):
        return "把这段耳语转换成正常说话的声音。"
    if any(word in text for word in ("耳语", "悄悄", "气声")):
        return "用小声耳语的方式把这段话说出来。"
    raise ValueError("耳语转换请填写“转换成耳语”或“转换成正常说话”")


def _enhance_instruction(value: str) -> str:
    text = str(value or "").strip()
    has_noise = any(word in text for word in ("噪", "杂音", "底噪"))
    has_reverb = any(word in text for word in ("混响", "回声"))
    if has_noise and not has_reverb:
        return "请只去除这段音频中的背景噪声，保留说话人原有的房间混响以及其它音色，输出等长的去噪结果。"
    if has_reverb and not has_noise:
        return "请只去除这段音频中的房间混响，保留原有的背景噪声以及其它音色，输出等长的去混响结果。"
    return "请对这段语音做纯净化处理，保留所有说话人的人声，并去除其中的噪声和混响，输出与输入等长的干净人声。"


def _music_separation_instruction(value: str) -> str:
    text = str(value or "").strip()
    if "所有人声" in text:
        return "请保留所有人声，说话和歌唱都算，其余声音都去掉。"
    if any(word in text for word in ("歌声", "歌唱", "唱歌")):
        return "请只保留歌声，其余声音都去掉。"
    raise ValueError("音乐人声提取请填写“只保留歌声”或“保留所有人声（说话和歌唱）”")


def _quality_instruction(value: str) -> str:
    text = str(value or "").strip()
    if any(word in text for word in ("带宽", "高频", "超分辨率", "清晰度", "补频")):
        return "请对这段语音做超分辨率/带宽扩展处理，恢复被削掉的高频成分，输出宽带纯净人声。"
    effects = (
        ("电话", ("电话", "手机", "窄带")),
        ("扩音器", ("扩音器", "喇叭", "广播")),
        ("水下闷声", ("水下", "闷声", "发闷")),
        ("削波破音", ("削波", "破音", "爆音")),
        ("丢包瞬断", ("丢包", "瞬断", "断续")),
        ("直流偏置", ("直流", "偏置")),
    )
    for effect, aliases in effects:
        if any(alias in text for alias in aliases):
            return (
                f"请消除这段音频的{effect}音色，这段音频带有混响，请恢复成无混响的干声，"
                "输出自然清晰的人声。"
            )
    raise ValueError("音质修复请填写“补充高频并提升清晰度”，或明确电话、扩音器、水下闷声等音色问题")


def build_instruction(task_key: str, primary: str, secondary: str = "") -> str:
    primary = str(primary or "").strip()
    secondary = str(secondary or "").strip()
    if task_key not in TASK_BY_KEY:
        raise ValueError(f"未知任务：{task_key}")
    if not primary:
        raise ValueError("主要内容不能为空")
    if task_key == "content_edit":
        return _canonical_content_edit(primary)
    if task_key == "lyric_edit":
        return _canonical_replacement(primary, lyrics=True)
    if task_key == "pitch":
        return _signed_adjustment(primary, allowed=(1, 2, 3), kind="音调", unit="个半音")
    if task_key == "speed":
        return _speed_adjustment(primary)
    if task_key == "volume":
        return _signed_adjustment(primary, allowed=(5, 10, 15), kind="音量", unit="分贝")
    if task_key == "emotion":
        return _emotion_instruction(primary)
    if task_key == "whisper":
        return _whisper_instruction(primary)
    if task_key == "enhance":
        return _enhance_instruction(primary)
    if task_key == "music_separate":
        return _music_separation_instruction(primary)
    if task_key == "quality":
        return _quality_instruction(primary)
    templates = {
        "instruct_tts": f'请基于下面的描述: "{secondary or "自然、清晰的声音"}",生成语音内容"{primary}".',
        # Match AuK's training prompt exactly. Extra transcript or descriptive
        # prose can make the model continue the reference audio's content.
        "zero_shot_tts": f'Say the following with the same voice: "{primary}"',
        "timbre": f"请将这段音频的音色修改为符合以下描述的声音：“{primary}”。",
        "deaccent": "请去掉这段语音里的方言口音，保持说话人音色一致。",
        "nonverbal": f"{primary.rstrip('。.!！')}。",
        "speech_separate": f"这段音频中只保留{primary}对应的语音，去掉其余说话人。",
        "target_speaker": f"请只保留说'{primary}'的人，去掉其他说话人，输出等长纯净人声。",
    }
    return templates[task_key].strip()
