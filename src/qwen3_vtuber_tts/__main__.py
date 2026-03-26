from __future__ import annotations

import argparse

import uvicorn

from .server import create_app
from .settings import ServerSettings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3-vtuber-tts",
        description="Run an OpenAI-compatible Qwen3 TTS FastAPI server.",
    )
    parser.add_argument("--host", help="Bind host. Overrides QWEN3_TTS_HOST.")
    parser.add_argument("--port", type=int, help="Bind port. Overrides QWEN3_TTS_PORT.")
    parser.add_argument(
        "--root-dir",
        help="Root directory that contains Qwen3 models and voice assets.",
    )
    parser.add_argument(
        "--base-model-dir",
        help="Path to the Qwen3 base model directory.",
    )
    parser.add_argument(
        "--asset-dir",
        help="Directory that contains generated prompt assets.",
    )
    parser.add_argument(
        "--default-voice",
        help="Default voice prompt name prefix. Overrides QWEN3_TTS_DEFAULT_VOICE.",
    )
    parser.add_argument(
        "--language",
        help="Default synthesis language. Overrides QWEN3_TTS_LANGUAGE.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Disable warmup on startup for faster boot during testing.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    overrides = {
        "QWEN3_TTS_HOST": args.host,
        "QWEN3_TTS_PORT": args.port,
        "QWEN3_TTS_ROOT_DIR": args.root_dir,
        "QWEN3_TTS_BASE_MODEL_DIR": args.base_model_dir,
        "QWEN3_TTS_ASSET_DIR": args.asset_dir,
        "QWEN3_TTS_DEFAULT_VOICE": args.default_voice,
        "QWEN3_TTS_LANGUAGE": args.language,
        "QWEN3_TTS_WARMUP": "false" if args.skip_warmup else None,
    }
    settings = ServerSettings.from_env(overrides=overrides)
    uvicorn.run(
        create_app(settings=settings),
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
