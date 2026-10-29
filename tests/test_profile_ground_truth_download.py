from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import assemble_ahmedml_schema_v3_candidate as assembler
from scripts import fetch_profile_ground_truth as downloader


class Response(io.BytesIO):
    def geturl(self):
        return "https://example.org/manifest.json"


class ProfileGroundTruthDownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.payload = b'{"data_release":{"id":"test-release"}}\n'
        self.release = self.root / "release.json"
        self.release.write_text(json.dumps({"data_release": {"profile_ground_truth": {
            "release_id": "test-release",
            "manifest_url": "https://example.org/manifest.json",
            "manifest_sha256": hashlib.sha256(self.payload).hexdigest(),
        }}}))

    def test_download_preserves_the_exact_verified_bytes(self):
        output = self.root / "download/manifest.json"
        with patch.object(downloader, "urlopen", return_value=Response(self.payload)):
            self.assertEqual(downloader.fetch_manifest(output, release_manifest=self.release), output.resolve())
        self.assertEqual(output.read_bytes(), self.payload)

    def test_invalid_download_does_not_replace_existing_file(self):
        output = self.root / "manifest.json"
        output.write_bytes(b"previous file")
        with patch.object(downloader, "urlopen", return_value=Response(b"404 page")):
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                downloader.fetch_manifest(output, release_manifest=self.release)
        self.assertEqual(output.read_bytes(), b"previous file")

    def test_matching_digest_with_wrong_release_is_rejected(self):
        release = json.loads(self.release.read_text())
        release["data_release"]["profile_ground_truth"]["release_id"] = "different-release"
        self.release.write_text(json.dumps(release))
        with patch.object(downloader, "urlopen", return_value=Response(self.payload)):
            with self.assertRaisesRegex(ValueError, "release ID differs"):
                downloader.fetch_manifest(self.root / "manifest.json", release_manifest=self.release)

    def test_assembler_downloads_when_local_manifest_is_omitted(self):
        config = self.root / "config.json"
        config.write_text("{}")
        def fetch(path):
            path.write_bytes(self.payload)
            return path
        def assemble(**kwargs):
            self.assertEqual(kwargs["profile_truth_manifest_path"].read_bytes(), self.payload)
            return {"status": "test-only"}
        args = ["--config", str(config), "--case-evidence-directory", str(self.root),
                "--discretization-cases", str(self.root / "cases.jsonl"), "--output", str(self.root / "package")]
        with patch.object(assembler, "fetch_manifest", side_effect=fetch) as download, patch.object(assembler, "assemble_package", side_effect=assemble):
            self.assertEqual(assembler.main(args), 0)
            download.assert_called_once()
        local = self.root / "local.json"
        local.write_bytes(self.payload)
        with patch.object(assembler, "fetch_manifest") as download, patch.object(assembler, "assemble_package", side_effect=assemble):
            self.assertEqual(assembler.main(args + ["--profile-ground-truth-manifest", str(local)]), 0)
            download.assert_not_called()
