import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from qwen3_vtuber_tts.server import create_app
from qwen3_vtuber_tts.settings import ServerSettings


class StubRuntime:
    def __init__(self) -> None:
        self.is_loaded = True
        self.warmup_called = False
        self.calls = []

    async def warmup(self) -> None:
        self.warmup_called = True

    async def synthesize(self, *, text: str, voice_name: str, language: str) -> bytes:
        self.calls.append(
            {
                "text": text,
                "voice_name": voice_name,
                "language": language,
            }
        )
        return b"RIFFstub-wave-data"


def make_settings() -> ServerSettings:
    return ServerSettings(
        root_dir=Path("D:/models/qwen3"),
        base_model_dir=Path("D:/models/qwen3/Qwen3-TTS-12Hz-1.7B-Base"),
        asset_dir=Path("D:/models/qwen3/voice_assets"),
        default_prompt_name="default_prompt.pkl",
        default_voice="default",
        default_model_name="qwen3-tts-en-single",
        language="English",
        host="127.0.0.1",
        port=8000,
        device="auto",
        dtype="auto",
        attn_implementation="eager",
        warmup_enabled=True,
        warmup_text="Hello. I'm ready.",
        skip_model_load=False,
    )


class ServerTest(unittest.TestCase):
    def test_root_and_health(self) -> None:
        runtime = StubRuntime()

        with TestClient(
            create_app(settings=make_settings(), runtime=runtime)
        ) as client:
            root_response = client.get("/")
            health_response = client.get("/health")

        self.assertTrue(runtime.warmup_called)
        self.assertEqual(root_response.status_code, 200)
        self.assertEqual(root_response.json()["default_model"], "qwen3-tts-en-single")
        self.assertEqual(health_response.status_code, 200)
        self.assertTrue(health_response.json()["model_loaded"])

    def test_audio_speech_json_request(self) -> None:
        runtime = StubRuntime()

        with TestClient(
            create_app(settings=make_settings(), runtime=runtime)
        ) as client:
            response = client.post(
                "/v1/audio/speech",
                json={
                    "input": "[warm] Hello there",
                    "voice": "hero",
                    "language": "English",
                    "response_format": "wav",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(
            runtime.calls,
            [
                {
                    "text": "Hello there",
                    "voice_name": "hero",
                    "language": "English",
                }
            ],
        )

    def test_audio_speech_form_request(self) -> None:
        runtime = StubRuntime()

        with TestClient(
            create_app(settings=make_settings(), runtime=runtime)
        ) as client:
            response = client.post(
                "/v1/audio/speech",
                data={"text": "Hello from form data"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime.calls[0]["text"], "Hello from form data")
        self.assertEqual(runtime.calls[0]["voice_name"], "default")

    def test_audio_speech_requires_input(self) -> None:
        runtime = StubRuntime()

        with TestClient(
            create_app(settings=make_settings(), runtime=runtime)
        ) as client:
            response = client.post("/v1/audio/speech", json={"voice": "default"})

        self.assertEqual(response.status_code, 422)
        self.assertIn("missing input text", response.json()["error"])

    def test_audio_speech_rejects_unsupported_response_format(self) -> None:
        runtime = StubRuntime()

        with TestClient(
            create_app(settings=make_settings(), runtime=runtime)
        ) as client:
            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello", "response_format": "mp3"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["supported_formats"], ["wav"])


if __name__ == "__main__":
    unittest.main()
