import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .runtime import QwenVoiceRuntime, RuntimeDependencyError
from .settings import ServerSettings
from .text import strip_style_tag


async def _parse_request_payload(request: Request) -> Dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return await request.json()
    if (
        "application/x-www-form-urlencoded" in content_type
        or "multipart/form-data" in content_type
    ):
        form = await request.form()
        return dict(form)
    raw = await request.body()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def create_app(
    settings: Optional[ServerSettings] = None,
    runtime: Optional[QwenVoiceRuntime] = None,
) -> FastAPI:
    resolved_settings = settings or ServerSettings.from_env()
    resolved_runtime = runtime or QwenVoiceRuntime(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            await resolved_runtime.warmup()
        except RuntimeDependencyError:
            raise
        except Exception:
            pass
        yield

    app = FastAPI(title="qwen3-vtuber-tts-server", lifespan=lifespan)

    @app.get("/")
    async def root() -> Dict[str, str]:
        return {
            "status": "ok",
            "message": "Qwen3 TTS server is running",
            "default_model": resolved_settings.default_model_name,
        }

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": resolved_runtime.is_loaded,
            "base_model_dir": str(resolved_settings.base_model_dir),
            "asset_dir": str(resolved_settings.asset_dir),
        }

    @app.post("/v1/audio/speech")
    async def audio_speech(request: Request) -> Response:
        try:
            payload = await _parse_request_payload(request)
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"failed to parse request body: {exc}"},
            )

        text = payload.get("input") or payload.get("text")
        if not text:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "missing input text",
                    "received_keys": sorted(payload.keys()),
                    "hint": "expected 'input' or 'text'",
                },
            )

        response_format = (payload.get("response_format") or "wav").lower()
        if response_format != "wav":
            return JSONResponse(
                status_code=422,
                content={
                    "error": "unsupported response_format",
                    "supported_formats": ["wav"],
                },
            )

        clean_text = strip_style_tag(str(text))
        if not clean_text:
            return JSONResponse(
                status_code=422,
                content={"error": "input text is empty after preprocessing"},
            )

        voice_name = payload.get("voice", resolved_settings.default_voice)
        language = payload.get("language", resolved_settings.language)

        try:
            audio_bytes = await resolved_runtime.synthesize(
                text=clean_text,
                voice_name=str(voice_name),
                language=str(language),
            )
        except FileNotFoundError as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
        except RuntimeDependencyError as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"error": f"TTS generation failed: {exc}"},
            )

        return Response(content=audio_bytes, media_type="audio/wav")

    return app
