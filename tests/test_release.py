"""Fast release-contract tests that do not load neural models."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import realtime
from webui import Bus


class ReleaseContractTests(unittest.TestCase):
    def test_sentence_stream_yields_complete_sentences_and_tail(self):
        pieces = ["Hello", " there. How", " are you?"]
        self.assertEqual(
            list(realtime.sentence_stream(pieces)),
            ["Hello there.", "How are you?"],
        )

    def test_model_discovery_is_recursive_and_prefers_release_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "gemma_4_e2b"
            nested.mkdir()
            fallback = root / "other-Q8_0.gguf"
            model = nested / "gemma-4-E2B-it-test-Q4_K_M.gguf"
            projector = nested / "mmproj-gemma-4-E2B-it-BF16.gguf"
            fallback.touch()
            model.touch()
            projector.touch()
            with patch.object(realtime, "MODELS_DIR", str(root)):
                self.assertEqual(Path(realtime._find_gguf(False)), model)
                self.assertEqual(Path(realtime._find_gguf(True)), projector)

    def test_event_bus_replays_latest_state(self):
        bus = Bus()
        bus.publish(type="state", state="listen")
        event = json.loads(bus.subscribe().get_nowait())
        self.assertEqual(event, {"type": "state", "state": "listen"})

    def test_pinned_manifest_has_full_llm_hashes(self):
        manifest_path = Path(__file__).parents[1] / "scripts" / "models.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["llama_cpp"]["tag"], "b9867")
        for entry in manifest["llm"]["files"]:
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertGreater(entry["size"], 0)


if __name__ == "__main__":
    unittest.main()
