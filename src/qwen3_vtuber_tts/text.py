from __future__ import annotations

import re
from pathlib import Path


STYLE_TAG_PATTERN = re.compile(
    r"^\s*\[(neutral|default|warm|playful|cold|serious)\]\s*", re.IGNORECASE
)
VOICE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def strip_style_tag(text: str) -> str:
    if not text:
        return text
    match = STYLE_TAG_PATTERN.match(text)
    if not match:
        return text
    return text[match.end() :].strip()


def sanitize_voice_name(voice_name: str, fallback: str = "default") -> str:
    cleaned = VOICE_NAME_PATTERN.sub("-", (voice_name or "").strip()).strip("-")
    return cleaned or fallback


def resolve_prompt_path(
    asset_dir: Path, voice_name: str, default_prompt_name: str = "default_prompt.pkl"
) -> Path:
    sanitized_voice = sanitize_voice_name(voice_name)
    candidate = asset_dir / f"{sanitized_voice}_prompt.pkl"
    if candidate.exists():
        return candidate
    return asset_dir / default_prompt_name
