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

    def test_skip_llm_fetches_the_voice_pipeline_without_the_bundled_llm(self):
        """--skip-llm must drop Gemma and keep every voice-pipeline artifact.

        LitheVoice can run its whole VAD/turn/STT/TTS path against another
        program's brain (realtime.py --llm-bridge), and such a host usually
        already runs a far larger model than the bundled 2B demo. The bundled
        LLM is the single largest download, so skipping it has to be a real
        option -- and it must not quietly take Parakeet or Kokoro with it.
        """
        import importlib.util
        root = Path(__file__).parents[1]
        spec = importlib.util.spec_from_file_location(
            "dl_models", root / "scripts" / "download_models.py")
        dl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dl)
        manifest = json.loads((root / "scripts" / "models.json").read_text(encoding="utf-8"))

        import io, contextlib
        def plan(skip):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                dl.print_plan(manifest, "cpu", False, skip)
            return buf.getvalue()

        kept, skipped = plan(False), plan(True)
        self.assertIn(manifest["llm"]["repo_id"], kept)
        self.assertNotIn(manifest["llm"]["repo_id"], skipped)
        self.assertIn("skipped", skipped)
        # the voice pipeline must survive the skip
        for marker in ("Parakeet", "Kokoro", "Smart Turn"):
            self.assertIn(marker, skipped, f"{marker} lost to --skip-llm")

    def test_every_downloaded_artifact_is_attributed(self):
        """THIRD_PARTY_NOTICES must name every artifact setup fetches.

        The notices table is hand-maintained but the manifest is the truth, so
        a new model added to models.json without a licence line here fails the
        release rather than shipping unattributed. Parakeet is CC BY 4.0:
        attribution is a licence condition, not a courtesy.
        """
        root = Path(__file__).parents[1]
        manifest = json.loads((root / "scripts" / "models.json").read_text(encoding="utf-8"))
        notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        missing = []
        for key, entry in manifest.items():
            if key == "schema_version" or not isinstance(entry, dict):
                continue
            repo = entry.get("repo_id")
            if repo and repo not in notices:
                missing.append(f"{key}: {repo} not named in THIRD_PARTY_NOTICES.md")
            licence = entry.get("license")
            if licence and licence.split()[0] not in notices:
                missing.append(f"{key}: licence {licence.split()[0]} not stated")
            rev = entry.get("revision")
            if rev and rev[:12] not in notices:
                missing.append(f"{key}: pinned revision {rev[:12]} not stated")
        self.assertEqual(missing, [], "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
