import unittest
from pathlib import Path
from unittest.mock import patch

from qwen3_vtuber_tts.text import (
    resolve_prompt_path,
    sanitize_voice_name,
    strip_style_tag,
)


class TextHelpersTest(unittest.TestCase):
    def test_strip_style_tag_removes_known_tag(self) -> None:
        self.assertEqual(strip_style_tag("[warm] Hello there"), "Hello there")

    def test_strip_style_tag_keeps_unknown_prefix(self) -> None:
        self.assertEqual(strip_style_tag("[robot] Hello there"), "[robot] Hello there")

    def test_sanitize_voice_name_normalizes_text(self) -> None:
        self.assertEqual(sanitize_voice_name("main voice 01"), "main-voice-01")

    def test_resolve_prompt_path_prefers_voice_specific_file(self) -> None:
        asset_dir = Path("C:/voice_assets")

        def fake_exists(path: Path) -> bool:
            return path.name == "hero_prompt.pkl"

        with patch("pathlib.Path.exists", autospec=True, side_effect=fake_exists):
            self.assertEqual(
                resolve_prompt_path(asset_dir, "hero"),
                asset_dir / "hero_prompt.pkl",
            )

    def test_resolve_prompt_path_falls_back_to_default(self) -> None:
        asset_dir = Path("C:/voice_assets")

        def fake_exists(path: Path) -> bool:
            return False

        with patch("pathlib.Path.exists", autospec=True, side_effect=fake_exists):
            self.assertEqual(
                resolve_prompt_path(asset_dir, "missing"),
                asset_dir / "default_prompt.pkl",
            )


if __name__ == "__main__":
    unittest.main()
