from __future__ import annotations

import copy
import hashlib
import io
import json
import struct
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

import reference.drivaerml.prediction_chunks as prediction_chunks_module

from reference.drivaerml.prediction_chunks import (
    CANDIDATE_ARTIFACT_ROLE,
    CANDIDATE_FORMAT,
    DEFAULT_MAX_NPY_HEADER_BYTES,
    DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES,
    PredictionChunkError,
    iter_prediction_chunks,
    load_prediction_chunk_manifest,
    validate_prediction_chunks,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8192):
            digest.update(block)
    return digest.hexdigest()


def _field_components(support_id: str) -> dict[str, int]:
    if support_id == "surface_native_cells":
        return {"pMeanTrim": 1, "wallShearStressMeanTrim": 3}
    if support_id == "volume_native_cells":
        return {"pMeanTrim": 1, "UMeanTrim": 3}
    raise AssertionError(support_id)


def _arrays(support_id: str, start: int, stop: int) -> dict[str, np.ndarray]:
    raw_ids = np.arange(start, stop, dtype=np.int64)
    index = raw_ids.astype(np.float64)
    result: dict[str, np.ndarray] = {
        "raw_cell_id": raw_ids,
        "pMeanTrim": (0.25 + index).astype(np.float32),
    }
    vector = np.column_stack((index + 1.0, index - 0.5, -index)).astype(
        np.float64
    )
    if support_id == "surface_native_cells":
        result["wallShearStressMeanTrim"] = vector
    else:
        result["UMeanTrim"] = vector
    return result


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _npy_bytes(array: np.ndarray) -> bytes:
    destination = io.BytesIO()
    np.lib.format.write_array(destination, array, allow_pickle=True)
    return destination.getvalue()


def _write_npz_members(
    path: Path,
    members: list[tuple[str, bytes]],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w", compression=compression) as archive:
            for name, payload in members:
                archive.writestr(name, payload)


def _members_for_arrays(
    arrays: dict[str, np.ndarray],
) -> list[tuple[str, bytes]]:
    return [(f"{name}.npy", _npy_bytes(array)) for name, array in arrays.items()]


def _refresh_chunk_hash(
    manifest_path: Path,
    document: dict,
    files: list[Path],
    chunk_index: int = 0,
) -> None:
    document["chunks"][chunk_index]["sha256"] = _sha256(files[chunk_index])
    _write_manifest(manifest_path, document)


def _mutate_first_central_directory_entry(
    path: Path,
    *,
    flag_bits_or: int = 0,
    local_flag_bits_or: int = 0,
    compression_method: int | None = None,
    uncompressed_size: int | None = None,
) -> None:
    payload = bytearray(path.read_bytes())
    local_offset = payload.index(b"PK\x03\x04")
    if local_flag_bits_or:
        local_flags = struct.unpack_from("<H", payload, local_offset + 6)[0]
        struct.pack_into(
            "<H",
            payload,
            local_offset + 6,
            local_flags | local_flag_bits_or,
        )
    offset = payload.index(b"PK\x01\x02")
    if flag_bits_or:
        flags = struct.unpack_from("<H", payload, offset + 8)[0]
        struct.pack_into("<H", payload, offset + 8, flags | flag_bits_or)
    if compression_method is not None:
        struct.pack_into("<H", payload, offset + 10, compression_method)
    if uncompressed_size is not None:
        struct.pack_into("<I", payload, offset + 24, uncompressed_size)
    path.write_bytes(payload)


def _write_manifest(path: Path, document: dict) -> None:
    path.write_text(f"{json.dumps(document, indent=2)}\n", encoding="utf-8")


def _candidate_fixture(
    root: Path,
    *,
    support_id: str,
    row_counts: tuple[int, ...] = (3, 2),
) -> tuple[Path, dict, list[Path]]:
    chunks: list[dict] = []
    files: list[Path] = []
    start = 0
    for index, row_count in enumerate(row_counts):
        stop = start + row_count
        relative = Path("chunks") / f"chunk-{index:05d}.npz"
        file_path = root / relative
        _write_npz(file_path, _arrays(support_id, start, stop))
        chunks.append(
            {
                "chunk_index": index,
                "file": relative.as_posix(),
                "sha256": _sha256(file_path),
                "row_count": row_count,
                "raw_cell_id_start": start,
                "raw_cell_id_stop": stop,
            }
        )
        files.append(file_path)
        start = stop
    document = {
        "format": CANDIDATE_FORMAT,
        "format_version": 1,
        "artifact_role": CANDIDATE_ARTIFACT_ROLE,
        "case_id": "run_1",
        "support_id": support_id,
        "association": "CellData",
        "total_row_count": start,
        "field_components": _field_components(support_id),
        "chunks": chunks,
    }
    manifest_path = root / "manifest.json"
    _write_manifest(manifest_path, document)
    return manifest_path, document, files


def _replace_chunk(
    manifest_path: Path,
    document: dict,
    files: list[Path],
    chunk_index: int,
    arrays: dict[str, np.ndarray],
) -> None:
    _write_npz(files[chunk_index], arrays)
    document["chunks"][chunk_index]["sha256"] = _sha256(files[chunk_index])
    _write_manifest(manifest_path, document)


class DrivAerMLPredictionChunkTests(unittest.TestCase):
    def test_manifest_in_place_mutation_during_read_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, _, _ = _candidate_fixture(
                Path(directory), support_id="surface_native_cells"
            )
            original = prediction_chunks_module._assert_snapshot
            mutated = False

            def mutate_before_identity_check(*args: object, **kwargs: object) -> None:
                nonlocal mutated
                if not mutated and kwargs.get("context") == "prediction chunk manifest":
                    mutated = True
                    with manifest_path.open("ab") as destination:
                        destination.write(b" ")
                original(*args, **kwargs)

            with mock.patch.object(
                prediction_chunks_module,
                "_assert_snapshot",
                side_effect=mutate_before_identity_check,
            ), self.assertRaisesRegex(PredictionChunkError, "changed while"):
                load_prediction_chunk_manifest(manifest_path)

    def test_npz_in_place_mutation_between_hash_and_parse_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, _, files = _candidate_fixture(
                Path(directory),
                support_id="volume_native_cells",
                row_counts=(3,),
            )
            original = prediction_chunks_module._validate_npz_members

            def validate_then_mutate(*args: object, **kwargs: object) -> None:
                original(*args, **kwargs)
                with files[0].open("ab") as destination:
                    destination.write(b"changed-after-hash")

            with mock.patch.object(
                prediction_chunks_module,
                "_validate_npz_members",
                side_effect=validate_then_mutate,
            ), self.assertRaisesRegex(PredictionChunkError, "changed while"):
                list(iter_prediction_chunks(manifest_path))

    def test_surface_manifest_streams_exact_fields_and_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _, _ = _candidate_fixture(
                root, support_id="surface_native_cells"
            )
            manifest = load_prediction_chunk_manifest(manifest_path)
            self.assertEqual(manifest.case_id, "run_1")
            self.assertEqual(manifest.total_row_count, 5)
            self.assertEqual(
                dict(manifest.field_components),
                {"pMeanTrim": 1, "wallShearStressMeanTrim": 3},
            )

            seen: list[int] = []
            for chunk in iter_prediction_chunks(
                manifest, hash_chunk_bytes=7, validation_block_rows=2
            ):
                seen.extend(chunk.raw_cell_id.tolist())
                self.assertEqual(
                    set(chunk.fields),
                    {"pMeanTrim", "wallShearStressMeanTrim"},
                )
                self.assertFalse(chunk.raw_cell_id.flags.writeable)
                self.assertFalse(chunk.field("pMeanTrim").flags.writeable)
                self.assertEqual(
                    chunk.field("wallShearStressMeanTrim").shape[1], 3
                )
            self.assertEqual(seen, list(range(5)))

            receipt = validate_prediction_chunks(
                manifest, hash_chunk_bytes=11, validation_block_rows=3
            )
            self.assertEqual(receipt.total_row_count, 5)
            self.assertEqual(receipt.chunk_count, 2)
            self.assertTrue(receipt.complete_gap_free_duplicate_free_coverage)
            self.assertEqual(len(receipt.chunk_sha256), 2)
            self.assertRegex(receipt.manifest_sha256, r"^[0-9a-f]{64}$")

    def test_volume_manifest_requires_pressure_and_three_component_velocity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _, _ = _candidate_fixture(
                root,
                support_id="volume_native_cells",
                row_counts=(4,),
            )
            chunks = list(iter_prediction_chunks(manifest_path))
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].field("pMeanTrim").shape, (4,))
            self.assertEqual(chunks[0].field("UMeanTrim").shape, (4, 3))
            np.testing.assert_array_equal(
                chunks[0].raw_cell_id, np.arange(4, dtype=np.int64)
            )

    def test_manifest_rejects_gap_overlap_and_interval_row_mismatch(self) -> None:
        mutations = (
            (
                "gap",
                lambda value: value["chunks"][1].update(
                    raw_cell_id_start=4,
                    raw_cell_id_stop=6,
                ),
                "coverage gap",
            ),
            (
                "overlap",
                lambda value: value["chunks"][1].update(
                    raw_cell_id_start=2,
                    raw_cell_id_stop=4,
                ),
                "overlap or duplicate",
            ),
            (
                "row count",
                lambda value: value["chunks"][0].update(row_count=2),
                "interval length",
            ),
            (
                "incomplete",
                lambda value: value.update(total_row_count=6),
                "coverage stops",
            ),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, document, _ = _candidate_fixture(
                    root, support_id="volume_native_cells"
                )
                mutate(document)
                _write_manifest(manifest_path, document)
                with self.assertRaisesRegex(PredictionChunkError, message):
                    load_prediction_chunk_manifest(manifest_path)

    def test_npz_rejects_wrong_raw_ids_dtype_and_values(self) -> None:
        cases = (
            (
                "int32",
                lambda arrays: arrays.update(
                    raw_cell_id=arrays["raw_cell_id"].astype(np.int32)
                ),
                "signed int64",
            ),
            (
                "duplicate",
                lambda arrays: arrays.update(
                    raw_cell_id=np.array([0, 1, 1], dtype=np.int64)
                ),
                "exact contiguous interval",
            ),
            (
                "out of order",
                lambda arrays: arrays.update(
                    raw_cell_id=np.array([0, 2, 1], dtype=np.int64)
                ),
                "exact contiguous interval",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, document, files = _candidate_fixture(
                    root,
                    support_id="volume_native_cells",
                    row_counts=(3,),
                )
                arrays = _arrays("volume_native_cells", 0, 3)
                mutate(arrays)
                _replace_chunk(manifest_path, document, files, 0, arrays)
                with self.assertRaisesRegex(PredictionChunkError, message):
                    list(iter_prediction_chunks(manifest_path))

    def test_npz_rejects_nonfinite_wrong_components_object_and_integer_fields(self) -> None:
        def nonfinite(arrays: dict[str, np.ndarray]) -> None:
            arrays["pMeanTrim"][1] = np.nan

        def wrong_components(arrays: dict[str, np.ndarray]) -> None:
            arrays["UMeanTrim"] = arrays["UMeanTrim"][:, :2]

        def object_field(arrays: dict[str, np.ndarray]) -> None:
            arrays["pMeanTrim"] = np.array(["a", "b", "c"], dtype=object)

        def integer_field(arrays: dict[str, np.ndarray]) -> None:
            arrays["pMeanTrim"] = np.arange(3, dtype=np.int64)

        def fortran_order(arrays: dict[str, np.ndarray]) -> None:
            arrays["UMeanTrim"] = np.asfortranarray(arrays["UMeanTrim"])

        cases = (
            ("nonfinite", nonfinite, "non-finite"),
            ("components", wrong_components, r"shape \(3, 3\)"),
            ("object", object_field, "object dtype"),
            ("integer", integer_field, "Float32 or Float64"),
            ("fortran order", fortran_order, "must use C-order"),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, document, files = _candidate_fixture(
                    root,
                    support_id="volume_native_cells",
                    row_counts=(3,),
                )
                arrays = _arrays("volume_native_cells", 0, 3)
                mutate(arrays)
                _replace_chunk(manifest_path, document, files, 0, arrays)
                with self.assertRaisesRegex(PredictionChunkError, message):
                    list(iter_prediction_chunks(manifest_path))

    def test_hashes_are_checked_lazily_one_file_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, document, _ = _candidate_fixture(
                root, support_id="surface_native_cells"
            )
            document["chunks"][1]["sha256"] = "0" * 64
            _write_manifest(manifest_path, document)
            chunks = iter_prediction_chunks(manifest_path, hash_chunk_bytes=5)
            first = next(chunks)
            self.assertEqual(first.descriptor.chunk_index, 0)
            with self.assertRaisesRegex(PredictionChunkError, "SHA-256 mismatch"):
                next(chunks)

    def test_unknown_fields_and_path_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, document, files = _candidate_fixture(
                root,
                support_id="surface_native_cells",
                row_counts=(3,),
            )

            extra_manifest = copy.deepcopy(document)
            extra_manifest["official"] = True
            _write_manifest(manifest_path, extra_manifest)
            with self.assertRaisesRegex(PredictionChunkError, "unknown keys"):
                load_prediction_chunk_manifest(manifest_path)

            extra_field = copy.deepcopy(document)
            extra_field["field_components"]["temperature"] = 1
            _write_manifest(manifest_path, extra_field)
            with self.assertRaisesRegex(PredictionChunkError, "unknown fields"):
                load_prediction_chunk_manifest(manifest_path)

            escaped = copy.deepcopy(document)
            escaped["chunks"][0]["file"] = "../outside.npz"
            _write_manifest(manifest_path, escaped)
            with self.assertRaisesRegex(PredictionChunkError, "without traversal"):
                load_prediction_chunk_manifest(manifest_path)

            arrays = _arrays("surface_native_cells", 0, 3)
            arrays["temperature"] = np.ones(3, dtype=np.float32)
            _replace_chunk(manifest_path, document, files, 0, arrays)
            with self.assertRaisesRegex(PredictionChunkError, "unknown arrays"):
                list(iter_prediction_chunks(manifest_path))

    def test_npz_preflight_rejects_duplicate_and_path_like_members(self) -> None:
        mutations = ("duplicate", "path")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, document, files = _candidate_fixture(
                    root,
                    support_id="volume_native_cells",
                    row_counts=(3,),
                )
                members = _members_for_arrays(
                    _arrays("volume_native_cells", 0, 3)
                )
                if mutation == "duplicate":
                    members.append(members[0])
                    expected = "duplicate ZIP members"
                else:
                    members[0] = ("nested/raw_cell_id.npy", members[0][1])
                    expected = "non-flat ZIP member"
                _write_npz_members(files[0], members)
                _refresh_chunk_hash(manifest_path, document, files)
                with self.assertRaisesRegex(PredictionChunkError, expected):
                    list(iter_prediction_chunks(manifest_path))

    def test_npz_preflight_rejects_data_descriptor_and_unknown_compression(self) -> None:
        mutations = (
            ("encrypted", {"flag_bits_or": 0x1}, "encrypted ZIP"),
            ("data descriptor", {"flag_bits_or": 0x8}, "data-descriptor"),
            (
                "hidden local data descriptor",
                {"local_flag_bits_or": 0x8},
                "inconsistent local and central ZIP flags",
            ),
            (
                "compression",
                {"compression_method": zipfile.ZIP_BZIP2},
                "unsupported ZIP compression",
            ),
        )
        for label, arguments, expected in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, document, files = _candidate_fixture(
                    root,
                    support_id="volume_native_cells",
                    row_counts=(3,),
                )
                _mutate_first_central_directory_entry(files[0], **arguments)
                _refresh_chunk_hash(manifest_path, document, files)
                with self.assertRaisesRegex(PredictionChunkError, expected):
                    list(iter_prediction_chunks(manifest_path))

    def test_npz_preflight_rejects_inflated_central_size_before_decompression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, document, files = _candidate_fixture(
                root,
                support_id="volume_native_cells",
                row_counts=(3,),
            )
            np.savez_compressed(
                files[0],
                **_arrays("volume_native_cells", 0, 3),
            )
            _mutate_first_central_directory_entry(
                files[0],
                uncompressed_size=DEFAULT_MAX_NPZ_UNCOMPRESSED_BYTES + 1,
            )
            _refresh_chunk_hash(manifest_path, document, files)
            with mock.patch(
                "reference.drivaerml.prediction_chunks.np.load",
                side_effect=AssertionError("np.load must not run"),
            ):
                with self.assertRaisesRegex(
                    PredictionChunkError,
                    "exceeding the .*per-chunk limit",
                ):
                    list(iter_prediction_chunks(manifest_path))

    def test_npz_preflight_rejects_oversized_header_before_numpy_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, document, files = _candidate_fixture(
                root,
                support_id="volume_native_cells",
                row_counts=(3,),
            )
            arrays = _arrays("volume_native_cells", 0, 3)
            oversized_length = DEFAULT_MAX_NPY_HEADER_BYTES + 1
            oversized_header = (
                b"\x93NUMPY\x01\x00"
                + struct.pack("<H", oversized_length)
                + b" " * (oversized_length - 1)
                + b"\n"
            )
            members = _members_for_arrays(arrays)
            members[0] = ("raw_cell_id.npy", oversized_header)
            _write_npz_members(files[0], members)
            _refresh_chunk_hash(manifest_path, document, files)
            with mock.patch(
                "reference.drivaerml.prediction_chunks.np.load",
                side_effect=AssertionError("np.load must not run"),
            ):
                with self.assertRaisesRegex(
                    PredictionChunkError,
                    "NPY header length .* exceeds",
                ):
                    list(iter_prediction_chunks(manifest_path))

    def test_npz_preflight_rejects_huge_shape_and_truncated_payload(self) -> None:
        mutations: tuple[tuple[str, bytes, str], ...]
        header_only = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header_only,
            {
                "descr": "<i8",
                "fortran_order": False,
                "shape": (10**12,),
            },
        )
        valid_raw = _npy_bytes(np.arange(3, dtype=np.int64))
        mutations = (
            ("huge shape", header_only.getvalue(), r"shape \(3,\)"),
            ("truncated payload", valid_raw[:-1], "requires exactly"),
        )
        for label, raw_member, expected in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, document, files = _candidate_fixture(
                    root,
                    support_id="volume_native_cells",
                    row_counts=(3,),
                )
                arrays = _arrays("volume_native_cells", 0, 3)
                members = _members_for_arrays(arrays)
                members[0] = ("raw_cell_id.npy", raw_member)
                _write_npz_members(files[0], members)
                _refresh_chunk_hash(manifest_path, document, files)
                with mock.patch(
                    "reference.drivaerml.prediction_chunks.np.load",
                    side_effect=AssertionError("np.load must not run"),
                ):
                    with self.assertRaisesRegex(PredictionChunkError, expected):
                        list(iter_prediction_chunks(manifest_path))

    def test_npz_preflight_enforces_caller_total_uncompressed_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _, _ = _candidate_fixture(
                root,
                support_id="surface_native_cells",
                row_counts=(3,),
            )
            with self.assertRaisesRegex(PredictionChunkError, "per-chunk limit"):
                list(
                    iter_prediction_chunks(
                        manifest_path,
                        max_npz_uncompressed_bytes=1,
                    )
                )

    def test_symlink_escape_and_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / "inside"
            inside.mkdir()
            manifest_path, document, _ = _candidate_fixture(
                inside,
                support_id="volume_native_cells",
                row_counts=(2,),
            )
            outside = root / "outside.npz"
            _write_npz(outside, _arrays("volume_native_cells", 0, 2))
            link = inside / "chunks" / "escape.npz"
            link.symlink_to(outside)
            escaped = copy.deepcopy(document)
            escaped["chunks"][0]["file"] = "chunks/escape.npz"
            escaped["chunks"][0]["sha256"] = _sha256(outside)
            _write_manifest(manifest_path, escaped)
            with self.assertRaisesRegex(PredictionChunkError, "escapes"):
                load_prediction_chunk_manifest(manifest_path)

            valid_text = json.dumps(document)
            duplicate_text = valid_text.replace(
                '"case_id": "run_1"',
                '"case_id": "run_1", "case_id": "run_2"',
                1,
            )
            manifest_path.write_text(duplicate_text, encoding="utf-8")
            with self.assertRaisesRegex(PredictionChunkError, "duplicate JSON key"):
                load_prediction_chunk_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
