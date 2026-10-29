from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_hiliftaeroml_submission_zip import (
    FIXED_MEMBER_MODE,
    FIXED_ZIP_TIMESTAMP,
    RECEIPT_SCHEMA,
    ZIP_COMPRESSION_LEVEL,
    DeterministicZipError,
    build_deterministic_submission_zip,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_hiliftaeroml_submission_zip.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8192):
            digest.update(block)
    return digest.hexdigest()


class HiLiftAeroMLDeterministicZipTests(unittest.TestCase):
    def test_builds_byte_identical_sorted_archives_with_fixed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "assembled"
            (source / "zeta").mkdir(parents=True)
            (source / "alpha").mkdir()
            (source / "zeta" / "last.json").write_text(
                '{"last":true}\n', encoding="utf-8"
            )
            (source / "alpha" / "second.bin").write_bytes(b"\x02\x00\x01")
            (source / "manifest.json").write_text(
                '{"schema":"v3"}\n', encoding="utf-8"
            )
            os.chmod(source / "zeta" / "last.json", 0o600)
            os.chmod(source / "alpha" / "second.bin", 0o755)

            first = root / "first.zip"
            second = root / "second.zip"
            first_receipt = build_deterministic_submission_zip(source, first)
            os.utime(source / "manifest.json", (1_700_000_000, 1_700_000_000))
            os.chmod(source / "manifest.json", 0o700)
            second_receipt = build_deterministic_submission_zip(source, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_receipt["archive_sha256"], _sha256(first))
            self.assertEqual(
                first_receipt["archive_sha256"], second_receipt["archive_sha256"]
            )
            self.assertEqual(first_receipt["schema"], RECEIPT_SCHEMA)
            self.assertEqual(first_receipt["member_count"], 3)
            self.assertEqual(second_receipt["member_count"], 3)

            with zipfile.ZipFile(first) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "alpha/second.bin",
                        "manifest.json",
                        "zeta/last.json",
                    ],
                )
                for member in archive.infolist():
                    self.assertEqual(member.date_time, FIXED_ZIP_TIMESTAMP)
                    self.assertEqual(member.compress_type, zipfile.ZIP_DEFLATED)
                    self.assertEqual(member.create_system, 3)
                    self.assertEqual(member.external_attr >> 16, FIXED_MEMBER_MODE)
                    self.assertNotIn("\\", member.filename)
                    self.assertFalse(member.filename.startswith("/"))

    def test_same_destination_can_be_rebuilt_byte_identically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "assembled"
            source.mkdir()
            (source / "payload.txt").write_text("stable\n", encoding="utf-8")
            output = root / "submission.zip"

            first = build_deterministic_submission_zip(source, output)
            first_bytes = output.read_bytes()
            second = build_deterministic_submission_zip(source, output)

            self.assertEqual(output.read_bytes(), first_bytes)
            self.assertEqual(first, second)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o644)

    def test_rejects_output_inside_source_without_creating_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "assembled"
            source.mkdir()
            (source / "payload.txt").write_text("data", encoding="utf-8")
            output = source / "artifacts" / "submission.zip"

            with self.assertRaisesRegex(DeterministicZipError, "outside"):
                build_deterministic_submission_zip(source, output)
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())

    def test_rejects_source_symlinks_and_non_regular_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.txt"
            target.write_text("target", encoding="utf-8")

            source_with_link = root / "source-link"
            source_with_link.mkdir()
            (source_with_link / "payload.txt").write_text("data", encoding="utf-8")
            (source_with_link / "linked.txt").symlink_to(target)
            with self.assertRaisesRegex(DeterministicZipError, "symbolic links"):
                build_deterministic_submission_zip(
                    source_with_link,
                    root / "link.zip",
                )
            self.assertFalse((root / "link.zip").exists())

            source_with_fifo = root / "source-fifo"
            source_with_fifo.mkdir()
            fifo = source_with_fifo / "stream"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(DeterministicZipError, "non-regular"):
                build_deterministic_submission_zip(
                    source_with_fifo,
                    root / "fifo.zip",
                )
            self.assertFalse((root / "fifo.zip").exists())

    def test_rejects_symlink_source_root_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "assembled"
            source.mkdir()
            (source / "payload.txt").write_text("data", encoding="utf-8")
            source_link = root / "assembled-link"
            source_link.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(DeterministicZipError, "must not be"):
                build_deterministic_submission_zip(source_link, root / "root-link.zip")

            output_target = root / "target.zip"
            output_target.write_bytes(b"preserve")
            output_link = root / "output.zip"
            output_link.symlink_to(output_target)
            with self.assertRaisesRegex(DeterministicZipError, "must not be"):
                build_deterministic_submission_zip(source, output_link)
            self.assertEqual(output_target.read_bytes(), b"preserve")

    def test_cli_prints_canonical_sha256_and_member_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "assembled"
            source.mkdir()
            (source / "submission.json").write_text("{}\n", encoding="utf-8")
            output = root / "submission.zip"

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), str(output)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads(completed.stdout)
            self.assertEqual(receipt["schema"], RECEIPT_SCHEMA)
            self.assertEqual(receipt["archive"], str(output.resolve()))
            self.assertEqual(receipt["archive_sha256"], _sha256(output))
            self.assertEqual(receipt["member_count"], 1)
            self.assertEqual(completed.stdout.count("\n"), 1)
            with zipfile.ZipFile(output) as archive:
                member = archive.getinfo("submission.json")
                self.assertEqual(member.date_time, FIXED_ZIP_TIMESTAMP)
                self.assertEqual(member._compresslevel, None)
                self.assertEqual(member.compress_type, zipfile.ZIP_DEFLATED)
            self.assertEqual(ZIP_COMPRESSION_LEVEL, 9)


if __name__ == "__main__":
    unittest.main()
