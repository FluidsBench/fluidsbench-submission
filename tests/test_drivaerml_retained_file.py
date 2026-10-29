from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reference.drivaerml.retained_file import (
    RetainedFileError,
    RetainedVerifiedFile,
)


class RetainedVerifiedFileTests(unittest.TestCase):
    def test_regular_file_is_retained_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.bin"
            source.write_bytes(b"retained bytes")
            with RetainedVerifiedFile.open(source, label="test input") as retained:
                self.assertEqual(retained.source_path, source.absolute())
                self.assertEqual(
                    retained.sha256(),
                    "fb256b25e0c6295ffe7f2d26c984d128075acbf4ac614fbed75887eebe83fc0b",
                )

    def test_final_component_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.bin"
            target.write_bytes(b"target")
            link = root / "input.bin"
            link.symlink_to(target)
            with self.assertRaisesRegex(RetainedFileError, "cannot open test input"):
                RetainedVerifiedFile.open(link, label="test input")

    def test_path_replacement_by_symlink_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.bin"
            replacement = root / "replacement.bin"
            source.write_bytes(b"original")
            replacement.write_bytes(b"replacement")
            with RetainedVerifiedFile.open(source, label="test input") as retained:
                source.unlink()
                source.symlink_to(replacement)
                with self.assertRaisesRegex(
                    RetainedFileError, "pathname no longer names the retained inode"
                ):
                    retained.assert_unchanged(context="during test")

    def test_symlink_swap_immediately_before_open_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.bin"
            target = root / "target.bin"
            source.write_bytes(b"original")
            target.write_bytes(b"target")
            original_open = os.open

            def replace_then_open(path: object, flags: int, *args: object) -> int:
                source.unlink()
                source.symlink_to(target)
                return original_open(path, flags, *args)

            with mock.patch.object(os, "open", side_effect=replace_then_open):
                with self.assertRaisesRegex(
                    RetainedFileError, "cannot open test input"
                ):
                    RetainedVerifiedFile.open(source, label="test input")


if __name__ == "__main__":
    unittest.main()
