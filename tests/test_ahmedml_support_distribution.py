"""Public cache installation, integrity failures and fresh-machine selection."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import http.client
import io
import json
import os
import stat
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from reference.ahmedml import distribution as d
from scripts import fetch_ahmedml_support as cli
from scripts import evaluate_ahmedml_candidate_case as evaluator_cli


class SupportInstallationTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.payloads = {
            "case-support.json": b'{"case_id":"run_4"}\n',
            "surface_area_vector_m2.npy": b"area-vector-fixture",
            "volume_cell_volume_m3.npy": b"cell-volume-fixture",
            "volume_region_code.npy": b"region-fixture",
            "profile_mapping_and_truth.npz": b"profile-fixture",
        }
        self.files = {
            n: {"size_bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
            for n, b in self.payloads.items()
        }
        self.stack.enter_context(
            patch.object(d, "expected_files", return_value=self.files)
        )
        self.archive = self.make_archive()
        self.entry = self.transport(self.archive)
        self.index = {"cases": {"run_4": self.entry}}
        self.stack.enter_context(
            patch.object(d, "load_distribution", return_value=self.index)
        )
        self.opener = Mock()
        self.opener.open.side_effect = lambda *a, **kw: io.BytesIO(self.archive)
        self.stack.enter_context(
            patch.object(d.urllib.request, "build_opener", return_value=self.opener)
        )

    def make_archive(self, extra=None):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in self.payloads.items():
                archive.writestr(name, payload)
            if extra:
                archive.writestr(*extra)
        return buffer.getvalue()

    def transport(self, payload):
        return {
            "url": "https://example.org/run_4.zip",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "expanded_bytes": sum(x["size_bytes"] for x in self.files.values()),
        }

    def install(self, **kwargs):
        return d.ensure_case_support("run_4", destination=self.root / "cache", **kwargs)

    def test_fresh_install_then_verified_reuse_requires_one_download(self):
        result = self.install()
        self.assertEqual(result.read_bytes(), self.payloads["case-support.json"])
        self.assertEqual(self.install(), result)
        self.assertEqual(self.install(offline=True), result)
        self.assertEqual(self.opener.open.call_count, 1)
        self.assertFalse(list((self.root / "cache").glob("*-install-*")))

    def test_missing_offline_case_performs_no_write_or_network(self):
        with self.assertRaises(d.SupportDistributionError):
            self.install(offline=True)
        self.opener.open.assert_not_called()
        self.assertFalse((self.root / "cache").exists())

    def test_corrupt_cache_is_replaced_and_preserved(self):
        result = self.install()
        result.write_bytes(b"damaged")
        self.install()
        self.assertEqual(result.read_bytes(), self.payloads["case-support.json"])
        old = list(result.parent.parent.glob(".run_4-replaced-*"))
        self.assertEqual(len(old), 1)
        self.assertEqual((old[0] / "case-support.json").read_bytes(), b"damaged")

    def test_failed_download_never_replaces_existing_cache(self):
        result = self.install()
        result.write_bytes(b"damaged")
        self.entry["sha256"] = "0" * 64
        with self.assertRaisesRegex(d.SupportDistributionError, "SHA-256"):
            self.install()
        self.assertEqual(result.read_bytes(), b"damaged")
        self.assertFalse(list(result.parent.parent.glob(".run_4-install-*")))

    def test_pinned_archive_does_not_override_individual_file_identities(self):
        self.payloads["volume_region_code.npy"] = b"WRONG!-fixture"
        self.archive = self.make_archive()
        self.entry.update(self.transport(self.archive))
        with self.assertRaises(d.SupportDistributionError):
            self.install()
        self.assertFalse((self.root / "cache/run_4").exists())

    def test_archive_path_traversal_and_unexpected_members_are_rejected(self):
        for name in ("../outside", "/outside", "nested/file", "extra.txt"):
            with self.subTest(name=name):
                self.archive = self.make_archive((name, b"bad"))
                self.entry.update(self.transport(self.archive))
                with self.assertRaises(d.SupportDistributionError):
                    self.install()
        self.assertFalse((self.root / "outside").exists())

    def test_archive_symlink_is_rejected(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, payload in self.payloads.items():
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, payload)
        self.archive = buffer.getvalue()
        self.entry.update(self.transport(self.archive))
        with self.assertRaisesRegex(d.SupportDistributionError, "unsafe"):
            self.install()

    def test_existing_case_symlink_is_never_followed(self):
        root = self.root / "cache"
        root.mkdir()
        (root / "run_4").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(d.SupportDistributionError):
            self.install()
        self.opener.open.assert_not_called()

    def test_wrong_archive_length_is_rejected(self):
        for delta in (-1, 1):
            self.entry["size_bytes"] = len(self.archive) + delta
            with self.assertRaises(d.SupportDistributionError):
                self.install()
        self.assertFalse((self.root / "cache/run_4").exists())

    def test_network_failure_retries_and_a_later_invocation_can_resume(self):
        self.opener.open.side_effect = urllib.error.URLError("interrupted")
        with patch.object(d.time, "sleep"), self.assertRaisesRegex(
            d.SupportDistributionError, "three attempts"
        ):
            self.install()
        self.assertEqual(self.opener.open.call_count, 3)
        self.opener.open.side_effect = lambda *a, **kw: io.BytesIO(self.archive)
        self.install()
        self.assertEqual(self.opener.open.call_count, 4)

    def test_truncated_http_response_retries_before_installation(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.side_effect = http.client.IncompleteRead(b"partial", 99)
        self.opener.open.side_effect = [response, io.BytesIO(self.archive)]
        with patch.object(d.time, "sleep"):
            self.install()
        self.assertEqual(self.opener.open.call_count, 2)

    def test_non_https_and_downgrade_redirects_are_rejected(self):
        self.entry["url"] = "http://example.org/support.zip"
        with self.assertRaisesRegex(d.SupportDistributionError, "HTTPS"):
            self.install()
        with self.assertRaisesRegex(d.SupportDistributionError, "HTTPS"):
            d._HTTPSRedirects().redirect_request(
                None, None, 302, "", {}, "http://example.org"
            )
        self.opener.open.assert_not_called()

    def test_explicit_case_support_never_attempts_automatic_installation(self):
        explicit = self.root / "custom/case-support.json"
        self.assertEqual(
            d.resolve_case_support("run_4", explicit=explicit), explicit.resolve()
        )
        self.opener.open.assert_not_called()


class DistributionContractTests(unittest.TestCase):
    def test_real_distribution_is_bound_to_all_existing_scientific_records(self):
        index = d.load_distribution()
        self.assertEqual(len(index["cases"]), 316)
        self.assertEqual(
            sum(c["expanded_bytes"] for c in index["cases"].values()), 41454139431
        )
        self.assertEqual(d.select_cases(split="full"), d.select_cases(split="medium"))
        self.assertEqual(len(d.select_cases(split="full")), 50)

    def test_distribution_cannot_redefine_the_scientific_case_manifest(self):
        index = copy.deepcopy(d.load_distribution())
        index["cases"]["run_4"]["case_manifest_sha256"] = "0" * 64
        original = d.read_json

        def read(path, **kwargs):
            return index if path == d.INDEX_PATH else original(path, **kwargs)

        with patch.object(d, "read_json", side_effect=read), self.assertRaises(
            d.SupportDistributionError
        ):
            d.load_distribution()

    def test_bad_selection_is_rejected(self):
        for cases in ([], ["run_4", "run_4"], ["../../outside"], ["run_999"]):
            with self.subTest(cases=cases), self.assertRaises(
                d.SupportDistributionError
            ):
                d.select_cases(case_ids=cases)

    def test_cache_location_is_revision_scoped_and_overridable(self):
        with patch.dict(
            os.environ,
            {"XDG_CACHE_HOME": "/tmp/test-xdg", "FLUIDSBENCH_AHMEDML_SUPPORT_ROOT": ""},
        ):
            self.assertEqual(
                d.cache_root(),
                Path("/tmp/test-xdg/fluidsbench/ahmedml", d.RELEASE_SHA256).resolve(),
            )
        with patch.dict(
            os.environ, {"FLUIDSBENCH_AHMEDML_SUPPORT_ROOT": "/tmp/hpc-cache"}
        ):
            self.assertEqual(d.cache_root(), Path("/tmp/hpc-cache").resolve())

    def test_dry_run_does_not_install_or_download(self):
        with patch("sys.argv", ["fetch", "--split", "full", "--dry-run"]), patch.object(
            cli, "ensure_case_support"
        ) as install, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(), 0)
        install.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["cases"], 50)

    def test_evaluator_resolves_support_and_preserves_explicit_offline_options(self):
        base_args = [
            "evaluate",
            "--case-id",
            "run_4",
            "--dataset-root",
            "/dataset",
            "--surface-prediction-manifest",
            "/predictions/surface.json",
            "--volume-prediction-manifest",
            "/predictions/volume.json",
            "--output",
            "/evidence/run_4.json",
        ]
        scenarios = [
            ([], None, None, False),
            (
                ["--support-root", "/scratch/support", "--offline"],
                None,
                Path("/scratch/support"),
                True,
            ),
            (
                ["--case-support", "/cache/run_4/case-support.json"],
                Path("/cache/run_4/case-support.json"),
                None,
                False,
            ),
        ]
        for extra_args, explicit, destination, offline in scenarios:
            with self.subTest(extra_args=extra_args), contextlib.ExitStack() as stack:
                stack.enter_context(patch("sys.argv", base_args + extra_args))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                identity = stack.enter_context(
                    patch.object(evaluator_cli, "load_source_identity")
                )
                resolved = Path("/managed/run_4/case-support.json")
                resolver = stack.enter_context(
                    patch.object(
                        evaluator_cli, "resolve_case_support", return_value=resolved
                    )
                )
                loader = stack.enter_context(
                    patch.object(evaluator_cli, "load_case_support")
                )
                evaluate = stack.enter_context(
                    patch.object(evaluator_cli, "evaluate_candidate_case")
                )
                writer = stack.enter_context(
                    patch.object(evaluator_cli, "write_atomic")
                )
                self.assertEqual(evaluator_cli.main(), 0)
                resolver.assert_called_once_with(
                    "run_4", explicit=explicit, destination=destination, offline=offline
                )
                loader.assert_called_once_with(
                    resolved, source_identity=identity.return_value
                )
                self.assertIs(
                    evaluate.call_args.kwargs["case_support"], loader.return_value
                )
                writer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
