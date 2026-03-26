from __future__ import annotations

import asyncio
import gc
import io
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

from .settings import PromptSettings, ServerSettings
from .text import resolve_prompt_path


class RuntimeDependencyError(RuntimeError):
    pass


def _import_qwen_runtime():
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeDependencyError(
            "Missing runtime dependency. Install torch, numpy, soundfile, and qwen_tts."
        ) from exc
    return np, sf, torch, Qwen3TTSModel


def load_prompt(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    if isinstance(saved, dict) and "voice_clone_prompt" in saved:
        return saved["voice_clone_prompt"]
    return saved


def write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_wav_file(
    path: Path, wav: Any, sample_rate: int, tail_seconds: float = 0.03
) -> None:
    np, sf, _, _ = _import_qwen_runtime()
    waveform = np.asarray(wav, dtype=np.float32).flatten()
    tail = np.zeros(int(sample_rate * tail_seconds), dtype=np.float32)
    waveform = np.concatenate([waveform, tail])
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0.97:
        waveform = waveform * (0.97 / peak)
    sf.write(str(path), waveform, sample_rate, format="WAV", subtype="PCM_16")


def wav_to_bytes(wav: Any, sample_rate: int, tail_seconds: float = 0.03) -> bytes:
    np, sf, _, _ = _import_qwen_runtime()
    waveform = np.asarray(wav, dtype=np.float32).flatten()
    tail = np.zeros(int(sample_rate * tail_seconds), dtype=np.float32)
    waveform = np.concatenate([waveform, tail])
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return audio_buffer.getvalue()


def _resolve_torch_dtype(torch_module: Any, dtype_name: str, use_cuda: bool) -> Any:
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    return torch_module.float16 if use_cuda else torch_module.float32


def _resolve_device_and_dtype(device_name: str, dtype_name: str) -> Tuple[str, Any]:
    _, _, torch_module, _ = _import_qwen_runtime()
    use_cuda = torch_module.cuda.is_available()
    if device_name == "auto":
        resolved_device = "cuda:0" if use_cuda else "cpu"
    else:
        resolved_device = device_name
    return resolved_device, _resolve_torch_dtype(
        torch_module=torch_module,
        dtype_name=dtype_name.lower(),
        use_cuda="cuda" in resolved_device,
    )


def _clear_torch_cache() -> None:
    _, _, torch_module, _ = _import_qwen_runtime()
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


class QwenVoiceRuntime:
    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self._generation_lock = asyncio.Lock()
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_model(self) -> Any:
        if self.settings.skip_model_load:
            raise RuntimeDependencyError(
                "Model loading is disabled because QWEN3_TTS_SKIP_MODEL_LOAD is enabled."
            )
        if self._model is not None:
            return self._model
        _, _, _, model_class = _import_qwen_runtime()
        device, dtype = _resolve_device_and_dtype(
            self.settings.device, self.settings.dtype
        )
        self._model = model_class.from_pretrained(
            str(self.settings.base_model_dir),
            device_map=device,
            dtype=dtype,
            attn_implementation=self.settings.attn_implementation,
        )
        self._loaded = True
        return self._model

    async def warmup(self) -> None:
        if not self.settings.warmup_enabled or self.settings.skip_model_load:
            return
        await self.synthesize(
            text=self.settings.warmup_text,
            voice_name=self.settings.default_voice,
            language=self.settings.language,
        )

    async def synthesize(
        self, text: str, voice_name: str, language: str | None = None
    ) -> bytes:
        async with self._generation_lock:
            model = self.ensure_model()
            prompt_path = resolve_prompt_path(
                self.settings.asset_dir,
                voice_name=voice_name,
                default_prompt_name=self.settings.default_prompt_name,
            )
            voice_clone_prompt = load_prompt(prompt_path)
            language_name = language or self.settings.language
            wavs, sample_rate = await asyncio.to_thread(
                model.generate_voice_clone,
                text=text,
                language=language_name,
                voice_clone_prompt=voice_clone_prompt,
            )
            return wav_to_bytes(wavs[0], sample_rate)


def create_voice_assets(settings: PromptSettings) -> Dict[str, str]:
    _, _, _, model_class = _import_qwen_runtime()
    settings.asset_dir.mkdir(parents=True, exist_ok=True)

    design_device, design_dtype = _resolve_device_and_dtype(
        settings.device, settings.dtype
    )
    design_model = model_class.from_pretrained(
        str(settings.voice_design_model_dir),
        device_map=design_device,
        dtype=design_dtype,
        attn_implementation=settings.attn_implementation,
    )
    ref_wavs, ref_sample_rate = design_model.generate_voice_design(
        text=settings.reference_text,
        language=settings.language,
        instruct=settings.reference_instruction,
    )
    save_wav_file(settings.reference_output_path, ref_wavs[0], ref_sample_rate)
    del design_model
    _clear_torch_cache()

    base_device, base_dtype = _resolve_device_and_dtype(settings.device, settings.dtype)
    base_model = model_class.from_pretrained(
        str(settings.base_model_dir),
        device_map=base_device,
        dtype=base_dtype,
        attn_implementation=settings.attn_implementation,
    )
    voice_clone_prompt = base_model.create_voice_clone_prompt(
        ref_audio=(ref_wavs[0], ref_sample_rate),
        ref_text=settings.reference_text,
    )
    with settings.prompt_output_path.open("wb") as handle:
        pickle.dump(
            {
                "voice_clone_prompt": voice_clone_prompt,
                "ref_text": settings.reference_text,
                "language": settings.language,
                "reference_path": str(settings.reference_output_path),
                "sample_rate": ref_sample_rate,
            },
            handle,
        )
    write_metadata(
        settings.metadata_output_path,
        {
            "voice_name": settings.voice_name,
            "language": settings.language,
            "reference_text": settings.reference_text,
            "reference_instruction": settings.reference_instruction,
            "reference_path": str(settings.reference_output_path),
            "prompt_path": str(settings.prompt_output_path),
            "sample_rate": ref_sample_rate,
        },
    )
    del base_model
    _clear_torch_cache()

    return {
        "reference_path": str(settings.reference_output_path),
        "prompt_path": str(settings.prompt_output_path),
        "metadata_path": str(settings.metadata_output_path),
    }
