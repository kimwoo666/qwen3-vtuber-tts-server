from __future__ import annotations

import argparse

from .runtime import create_voice_assets
from .settings import PromptSettings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3-vtuber-tts-prompt",
        description="Generate reusable Qwen3 voice clone prompt assets.",
    )
    parser.add_argument(
        "--root-dir",
        help="Root directory that contains Qwen3 models and voice assets.",
    )
    parser.add_argument(
        "--base-model-dir",
        help="Path to the Qwen3 base model directory.",
    )
    parser.add_argument(
        "--voice-design-model-dir",
        help="Path to the Qwen3 voice design model directory.",
    )
    parser.add_argument(
        "--asset-dir",
        help="Directory where generated prompt assets will be written.",
    )
    parser.add_argument(
        "--voice-name",
        help="Logical voice name. Output files use this as the filename prefix.",
    )
    parser.add_argument(
        "--language",
        help="Language name passed to the Qwen3 models.",
    )
    parser.add_argument(
        "--reference-text",
        help="Reference text used for voice design and clone prompt creation.",
    )
    parser.add_argument(
        "--reference-instruction",
        help="Natural language description of the target voice persona.",
    )
    parser.add_argument(
        "--device",
        help="Torch device override such as auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        help="Torch dtype override such as auto, float16, float32, or bfloat16.",
    )
    parser.add_argument(
        "--attn-implementation",
        help="Attention implementation override such as eager or flash_attention_2.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    overrides = {
        "QWEN3_TTS_ROOT_DIR": args.root_dir,
        "QWEN3_TTS_BASE_MODEL_DIR": args.base_model_dir,
        "QWEN3_TTS_VOICE_DESIGN_MODEL_DIR": args.voice_design_model_dir,
        "QWEN3_TTS_ASSET_DIR": args.asset_dir,
        "QWEN3_TTS_VOICE_NAME": args.voice_name,
        "QWEN3_TTS_LANGUAGE": args.language,
        "QWEN3_TTS_REFERENCE_TEXT": args.reference_text,
        "QWEN3_TTS_REFERENCE_INSTRUCT": args.reference_instruction,
        "QWEN3_TTS_DEVICE": args.device,
        "QWEN3_TTS_DTYPE": args.dtype,
        "QWEN3_TTS_ATTN_IMPL": args.attn_implementation,
    }
    settings = PromptSettings.from_env(overrides=overrides)
    outputs = create_voice_assets(settings)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
