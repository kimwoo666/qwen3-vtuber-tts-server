import unittest
from pathlib import Path

from qwen3_vtuber_tts.settings import PromptSettings, ServerSettings, parse_bool


class SettingsTest(unittest.TestCase):
    def test_parse_bool(self) -> None:
        self.assertTrue(parse_bool("true", False))
        self.assertTrue(parse_bool("1", False))
        self.assertFalse(parse_bool("no", True))

    def test_server_settings_from_env(self) -> None:
        settings = ServerSettings.from_env(
            {
                "QWEN3_TTS_ROOT_DIR": "D:/models/qwen3",
                "QWEN3_TTS_PORT": "9000",
                "QWEN3_TTS_DEFAULT_VOICE": "hero",
            }
        )
        self.assertEqual(settings.root_dir, Path("D:/models/qwen3"))
        self.assertEqual(settings.port, 9000)
        self.assertEqual(settings.default_voice, "hero")
        self.assertEqual(
            settings.base_model_dir,
            Path("D:/models/qwen3/Qwen3-TTS-12Hz-1.7B-Base"),
        )

    def test_server_settings_support_overrides(self) -> None:
        settings = ServerSettings.from_env(
            {"QWEN3_TTS_ROOT_DIR": "D:/models/qwen3"},
            overrides={"QWEN3_TTS_PORT": 9100, "QWEN3_TTS_HOST": "0.0.0.0"},
        )
        self.assertEqual(settings.port, 9100)
        self.assertEqual(settings.host, "0.0.0.0")

    def test_prompt_settings_from_env(self) -> None:
        settings = PromptSettings.from_env(
            {
                "QWEN3_TTS_ROOT_DIR": "D:/models/qwen3",
                "QWEN3_TTS_VOICE_NAME": "guide",
                "QWEN3_TTS_REFERENCE_TEXT": "Stay calm. We can fix this.",
                "QWEN3_TTS_REFERENCE_INSTRUCT": "Young adult voice with clear diction.",
            }
        )
        self.assertEqual(settings.voice_name, "guide")
        self.assertEqual(settings.reference_file_name, "guide_reference.wav")
        self.assertEqual(settings.prompt_file_name, "guide_prompt.pkl")

    def test_prompt_settings_require_reference_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "QWEN3_TTS_REFERENCE_TEXT is required"):
            PromptSettings.from_env({"QWEN3_TTS_ROOT_DIR": "D:/models/qwen3"})


if __name__ == "__main__":
    unittest.main()
