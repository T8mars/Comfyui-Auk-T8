from __future__ import annotations

import math
import re

TTS_TASK_KEYS = frozenset({"instruct_tts", "zero_shot_tts"})
AUTO_DURATION_MODE = "自动估算（TTS 推荐）"
MANUAL_DURATION_MODE = "手动指定"

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
_DIGIT_RE = re.compile(r"\d")
_SHORT_PAUSE_RE = re.compile(r"[,，、;；:：]")
_LONG_PAUSE_RE = re.compile(r"[.!?。！？…]")


def estimate_tts_seconds(text: str, *, max_seconds: float = 30.0) -> float:
    """Estimate spoken length with a small tail margin, rounded up to 0.1 s."""
    value = str(text or "").strip()
    if not value:
        raise ValueError("自动估算时长需要填写目标文本")

    cjk_count = len(_CJK_RE.findall(value))
    latin_words = len(_LATIN_WORD_RE.findall(value))
    digit_count = len(_DIGIT_RE.findall(value))
    short_pauses = len(_SHORT_PAUSE_RE.findall(value))
    long_pauses = len(_LONG_PAUSE_RE.findall(value))
    other_units = len(
        re.findall(
            r"[^\s\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaffA-Za-z\d,，、;；:：.!?。！？…]",
            value,
        )
    )
    seconds = (
        0.30
        + cjk_count * 0.20
        + latin_words * 0.36
        + digit_count * 0.18
        + other_units * 0.16
        + short_pauses * 0.12
        + long_pauses * 0.20
    )
    return min(float(max_seconds), max(0.6, math.ceil(seconds * 10.0 - 1e-9) / 10.0))
