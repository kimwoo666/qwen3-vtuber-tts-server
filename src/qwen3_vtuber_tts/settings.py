from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


def _read_env(environ: Optional[Mapping[str, str]] = None) -> Mapping[str, str]:
    return environ if environ is not None else os.environ


def _merge_env(
    environ: Optional[Mapping[str, str]] = None,
    overrides: Optional[Mapping[str, str | int]] = None,
) -> Mapping[str, str]:
    env = dict(_read_env(environ))
    if not overrides:
        return env
    for key, value in overrides.items():
        if value is None:
            continue
        env[key] = str(value)
    return env


def parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    return int(value.strip())


def _default_root(environ: Mapping[str, str]) -> Path:
    raw = parse_optional(environ.get("QWEN3_TTS_ROOT_DIR"))
    return Path(raw).expanduser() if raw else Path.cwd()


@dataclass(frozen=True, slots=True)
class ServerSettings:
    root_dir: Path
    base_model_dir: Path
    asset_dir: Path
    default_prompt_name: str
    default_voice: str
    default_model_name: str
    language: str
    host: str
    port: int
    device: str
    dtype: str
    attn_implementation: str
    warmup_enabled: bool
    warmup_text: str
    skip_model_load: bool

    @property
    def default_prompt_path(self) -> Path:
        return self.asset_dir / self.default_prompt_name

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        overrides: Optional[Mapping[str, str | int]] = None,
    ) -> "ServerSettings":
        env = _merge_env(environ, overrides)
        root_dir = _default_root(env)
        base_model_dir = Path(
            parse_optional(env.get("QWEN3_TTS_BASE_MODEL_DIR"))
            or root_dir / "Qwen3-TTS-12Hz-1.7B-Base"
        ).expanduser()
        asset_dir = Path(
            parse_optional(env.get("QWEN3_TTS_ASSET_DIR")) or root_dir / "voice_assets"
        ).expanduser()
        return cls(
            root_dir=root_dir.expanduser(),
            base_model_dir=base_model_dir,
            asset_dir=asset_dir,
            default_prompt_name=env.get(
                "QWEN3_TTS_DEFAULT_PROMPT_NAME", "default_prompt.pkl"
            ),
            default_voice=env.get("QWEN3_TTS_DEFAULT_VOICE", "default"),
            default_model_name=env.get(
                "QWEN3_TTS_DEFAULT_MODEL_NAME", "qwen3-tts-en-single"
            ),
            language=env.get("QWEN3_TTS_LANGUAGE", "English"),
            host=env.get("QWEN3_TTS_HOST", "127.0.0.1"),
            port=parse_int(env.get("QWEN3_TTS_PORT"), 8000),
            device=env.get("QWEN3_TTS_DEVICE", "auto"),
            dtype=env.get("QWEN3_TTS_DTYPE", "auto"),
            attn_implementation=env.get("QWEN3_TTS_ATTN_IMPL", "eager"),
            warmup_enabled=parse_bool(env.get("QWEN3_TTS_WARMUP"), True),
            warmup_text=env.get("QWEN3_TTS_WARMUP_TEXT", "Hello. I'm ready."),
            skip_model_load=parse_bool(env.get("QWEN3_TTS_SKIP_MODEL_LOAD"), False),
        )


@dataclass(frozen=True, slots=True)
class PromptSettings:
    root_dir: Path
    base_model_dir: Path
    voice_design_model_dir: Path
    asset_dir: Path
    voice_name: str
    language: str
    reference_text: str
    reference_instruction: str
    reference_file_name: str
    prompt_file_name: str
    metadata_file_name: str
    device: str
    dtype: str
    attn_implementation: str

    @property
    def reference_output_path(self) -> Path:
        return self.asset_dir / self.reference_file_name

    @property
    def prompt_output_path(self) -> Path:
        return self.asset_dir / self.prompt_file_name

    @property
    def metadata_output_path(self) -> Path:
        return self.asset_dir / self.metadata_file_name

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        overrides: Optional[Mapping[str, str | int]] = None,
    ) -> "PromptSettings":
        env = _merge_env(environ, overrides)
        root_dir = _default_root(env)
        asset_dir = Path(
            parse_optional(env.get("QWEN3_TTS_ASSET_DIR")) or root_dir / "voice_assets"
        ).expanduser()
        voice_name = env.get("QWEN3_TTS_VOICE_NAME", "default").strip() or "default"
        reference_text = parse_optional(env.get("QWEN3_TTS_REFERENCE_TEXT"))
        if not reference_text:
            raise ValueError(
                "QWEN3_TTS_REFERENCE_TEXT is required. Provide it with an environment variable "
                "or --reference-text when running qwen3-vtuber-tts-prompt."
            )
        reference_instruction = parse_optional(env.get("QWEN3_TTS_REFERENCE_INSTRUCT"))
        if not reference_instruction:
            raise ValueError(
                "QWEN3_TTS_REFERENCE_INSTRUCT is required. Provide it with an environment variable "
                "or --reference-instruction when running qwen3-vtuber-tts-prompt."
            )
        return cls(
            root_dir=root_dir.expanduser(),
            base_model_dir=Path(
                parse_optional(env.get("QWEN3_TTS_BASE_MODEL_DIR"))
                or root_dir / "Qwen3-TTS-12Hz-1.7B-Base"
            ).expanduser(),
            voice_design_model_dir=Path(
                parse_optional(env.get("QWEN3_TTS_VOICE_DESIGN_MODEL_DIR"))
                or root_dir / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            ).expanduser(),
            asset_dir=asset_dir,
            voice_name=voice_name,
            language=env.get("QWEN3_TTS_LANGUAGE", "English"),
            reference_text=reference_text,
            reference_instruction=reference_instruction,
            reference_file_name=f"{voice_name}_reference.wav",
            prompt_file_name=f"{voice_name}_prompt.pkl",
            metadata_file_name=f"{voice_name}_metadata.json",
            device=env.get("QWEN3_TTS_DEVICE", "auto"),
            dtype=env.get("QWEN3_TTS_DTYPE", "auto"),
            attn_implementation=env.get("QWEN3_TTS_ATTN_IMPL", "eager"),
        )
