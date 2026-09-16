from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from reference.hiliftaeroml import (
    native_profile_truth_materializer as truth_materializer,
)
from reference.hiliftaeroml.native_profile_truth import (
    TRUTH_ARRAYS,
    CaseUniverse,
    NativeProfileTruthError,
    array_identity,
    assemble_release,
    build_case_truth,
    build_source_index,
    load_case_truth_arrays,
    load_case_universe,
    load_source_index,
    preflight,
    validate_release,
)
from reference.hiliftaeroml.native_profile_truth_materializer import (
    _historical_pdmsh_root,
    _line_weights,
    _reconstruct_cp,
    _reconstruct_velocity,
    _SparseVTUReader,
    load_compact_profile_support_inputs,
)
from reference.hiliftaeroml.native_profiles import canonical_json_bytes
from tests.test_hiliftaeroml_native_profiles import CASE_ID, make_case

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_MANIFEST = (
    ROOT
    / "benchmark-specs"
    / "hiliftaeroml"
    / "scoring-support"
    / "hiliftaeroml-native-all-splits-support-v1-candidate"
    / "manifest.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def small_universe(*case_ids: str) -> CaseUniverse:
    ordered = tuple(sorted(case_ids))
    return CaseUniverse(
        case_ids=ordered,
        case_sets=(("caseset-test", ordered, "9" * 64),),
        support_manifest_sha256="8" * 64,
        case_universe_sha256=hashlib.sha256(
            ("\n".join(ordered) + "\n").encode()
        ).hexdigest(),
    )


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_official_case_universe_is_eight_sets_and_1355_unique_cases() -> None:
    universe = load_case_universe(SUPPORT_MANIFEST)
    assert len(universe.case_sets) == 8
    assert len(universe.case_ids) == 1355
    assert CASE_ID in universe.case_ids
    assert len(universe.case_ids) == len(set(universe.case_ids))


def test_historical_training_points_stat_is_not_a_truth_authority(
    tmp_path: Path,
) -> None:
    pdmsh_root = tmp_path / "case.pdmsh"
    pdmsh_root.mkdir()
    historical_points = (
        pdmsh_root / "_tensordict" / "interior" / "_tensordict" / "points.memmap"
    )
    metadata = {
        "training_points_path": str(historical_points),
        "training_points_role": "historical PDMsh provenance only",
        "training_points_used_for_native_filter": False,
        "training_points_used_for_partition_order": False,
        "training_points_coordinate_order_equality_required": False,
        # Deliberately stale and no live points file: neither is a production
        # truth input after the direct native avg(u) replay pivot.
        "training_points_size_bytes": 1,
        "training_points_mtime_ns": 2,
    }
    assert _historical_pdmsh_root(metadata, CASE_ID) == pdmsh_root.resolve()

    metadata["training_points_used_for_native_filter"] = True
    with pytest.raises(NativeProfileTruthError, match="role differs"):
        _historical_pdmsh_root(metadata, CASE_ID)


def test_truth_release_is_lossless_deduplicated_and_deterministic(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    make_case(outputs)
    universe = small_universe(CASE_ID)
    source_index_path = tmp_path / "source-index.json"
    source_body = build_source_index(
        universe=universe,
        outputs_roots=[outputs],
        output_path=source_index_path,
    )
    assert source_body["status"] == "complete"
    source, source_sha = load_source_index(
        source_index_path, universe=universe, require_complete=True
    )

    releases = [tmp_path / "release-a", tmp_path / "release-b"]
    for release in releases:
        record = build_case_truth(
            case_id=CASE_ID,
            source=source,
            output_root=release,
        )
        assert record["truth_boundary"] == {
            "benchmark_owned": True,
            "participant_visible": False,
            "participant_prediction_artifacts_include_truth": False,
        }
        manifest = assemble_release(
            universe=universe,
            source_index_sha256=source_sha,
            output_root=release,
            cases_per_chunk=1,
        )
        assert manifest["storage"]["topology_payloads_duplicated"] == 0
        assert manifest["storage"]["case_artifacts_duplicated_across_case_sets"] == 0
        assert manifest["activation"] == {
            "owner_approval_complete": False,
            "published": False,
            "submissions_opened": False,
        }
        result = validate_release(
            universe=universe,
            source=source,
            source_index_sha256=source_sha,
            output_root=release,
        )
        assert result["status"] == "pass"
        assert result["source_replay_complete"] is True

    assert tree_hashes(releases[0]) == tree_hashes(releases[1])
    build_case_truth(
        case_id=CASE_ID,
        source=source,
        output_root=releases[0],
        check=True,
    )
    assemble_release(
        universe=universe,
        source_index_sha256=source_sha,
        output_root=releases[0],
        cases_per_chunk=1,
        check=True,
    )

    artifact = releases[0] / "cases" / f"{CASE_ID}.npz"
    with np.load(artifact, allow_pickle=False) as archive:
        assert archive.files == list(TRUTH_ARRAYS)
        assert "prediction_cp" not in archive.files
        assert "predicted_velocity_nd" not in archive.files
        assert np.all(np.isfinite(archive["truth_cp"]))
        assert archive["reference_velocity_nd"].shape == (4005, 3)
        expected_truth_cp = np.array(archive["truth_cp"], copy=True)
        expected_velocity = np.array(archive["reference_velocity_nd"], copy=True)
    loaded_record, loaded_record_sha, loaded_arrays, loaded_artifact_sha = (
        load_case_truth_arrays(releases[0], CASE_ID)
    )
    assert loaded_record["case_id"] == CASE_ID
    assert loaded_record_sha == digest(
        releases[0] / "case-records" / f"{CASE_ID}.json"
    )
    assert loaded_artifact_sha == digest(artifact)
    assert np.array_equal(loaded_arrays["truth_cp"], expected_truth_cp)
    assert np.array_equal(
        loaded_arrays["reference_velocity_nd"], expected_velocity, equal_nan=True
    )
    record = json.loads(
        (releases[0] / "case-records" / f"{CASE_ID}.json").read_text(encoding="utf-8")
    )
    assert len(record["volume_velocity"]["alignment"]["stations"]) == 5
    assert [
        station["station_id"]
        for station in record["volume_velocity"]["alignment"]["stations"]
    ] == ["B.2", "B.3", "C.1", "C.2", "C.3"]
    assert record["surface_cp"]["alignment"]["station_rows"] == list("ABCDEFGHIJ")


def test_source_inventory_preflight_is_fail_closed_when_one_case_is_missing(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    make_case(outputs)
    missing_case = "geo_LHC002_AoA_6"
    universe = small_universe(CASE_ID, missing_case)
    source_index_path = tmp_path / "source-index.json"
    source_body = build_source_index(
        universe=universe,
        outputs_roots=[outputs],
        output_path=source_index_path,
    )
    source, source_sha = load_source_index(
        source_index_path, universe=universe, require_complete=False
    )
    receipt = preflight(
        universe=universe,
        source=source,
        source_index_sha256=source_sha,
    )
    assert source_body["status"] == "incomplete"
    assert receipt["status"] == "blocked_missing_materialized_sources"
    assert receipt["missing_source_case_ids"] == [missing_case]
    assert receipt["case_array_job"]["array"] == "0-1"
    with pytest.raises(NativeProfileTruthError, match="incomplete"):
        load_source_index(source_index_path, universe=universe, require_complete=True)


def test_truth_array_identity_binds_dtype_shape_and_values() -> None:
    base = np.asarray([0.0, 1.0], dtype=np.float64)
    assert array_identity(base, namespace="test") == array_identity(
        base.copy(), namespace="test"
    )
    assert array_identity(base, namespace="test") != array_identity(
        base.astype(np.float32), namespace="test"
    )
    assert array_identity(base, namespace="test") != array_identity(
        base.reshape(1, 2), namespace="test"
    )
    changed = base.copy()
    changed[1] = 2.0
    assert array_identity(base, namespace="test") != array_identity(
        changed, namespace="test"
    )


def test_release_validator_rejects_changed_truth_bytes(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    make_case(outputs)
    universe = small_universe(CASE_ID)
    source_path = tmp_path / "source.json"
    build_source_index(
        universe=universe, outputs_roots=[outputs], output_path=source_path
    )
    source, source_sha = load_source_index(
        source_path, universe=universe, require_complete=True
    )
    release = tmp_path / "release"
    build_case_truth(case_id=CASE_ID, source=source, output_root=release)
    assemble_release(
        universe=universe,
        source_index_sha256=source_sha,
        output_root=release,
        cases_per_chunk=1,
    )
    artifact = release / "cases" / f"{CASE_ID}.npz"
    payload = bytearray(artifact.read_bytes())
    payload[-1] ^= 1
    artifact.write_bytes(payload)
    with pytest.raises(NativeProfileTruthError, match="live bytes differ"):
        validate_release(
            universe=universe,
            source=source,
            source_index_sha256=source_sha,
            output_root=release,
            replay_sources=False,
        )


def test_source_index_rejects_conflicting_complete_duplicate_roots(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    make_case(first)
    make_case(second)
    universe = small_universe(CASE_ID)
    with pytest.raises(NativeProfileTruthError, match="conflicting complete"):
        build_source_index(
            universe=universe,
            outputs_roots=[first, second],
            output_path=tmp_path / "source.json",
        )


def _one_case_release(tmp_path: Path) -> tuple[CaseUniverse, dict, str, Path]:
    outputs = tmp_path / "outputs"
    make_case(outputs)
    universe = small_universe(CASE_ID)
    source_path = tmp_path / "source.json"
    build_source_index(
        universe=universe, outputs_roots=[outputs], output_path=source_path
    )
    source, source_sha = load_source_index(
        source_path, universe=universe, require_complete=True
    )
    release = tmp_path / "release"
    build_case_truth(case_id=CASE_ID, source=source, output_root=release)
    assemble_release(
        universe=universe,
        source_index_sha256=source_sha,
        output_root=release,
        cases_per_chunk=1,
    )
    return universe, source, source_sha, release


def test_release_validator_rejects_intermediate_symlink_escape(
    tmp_path: Path,
) -> None:
    universe, source, source_sha, release = _one_case_release(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "index.json").write_bytes((release / "index.json").read_bytes())
    (release / "escape").symlink_to(outside, target_is_directory=True)
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["master_index"]["file"] = "escape/index.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(NativeProfileTruthError, match="outside the release root"):
        validate_release(
            universe=universe,
            source=source,
            source_index_sha256=source_sha,
            output_root=release,
            replay_sources=False,
        )


def test_release_validator_rejects_final_symlink_inside_release(
    tmp_path: Path,
) -> None:
    universe, source, source_sha, release = _one_case_release(tmp_path)
    index = release / "index.json"
    target = release / "index-target.json"
    index.rename(target)
    index.symlink_to(target.name)
    with pytest.raises(NativeProfileTruthError, match="symlink"):
        validate_release(
            universe=universe,
            source=source,
            source_index_sha256=source_sha,
            output_root=release,
            replay_sources=False,
        )


def test_release_validator_rejects_untracked_crash_file(tmp_path: Path) -> None:
    universe, source, source_sha, release = _one_case_release(tmp_path)
    (release / "cases" / ".orphan.tmp-123").write_bytes(b"interrupted")
    with pytest.raises(NativeProfileTruthError, match="release inventory differs"):
        validate_release(
            universe=universe,
            source=source,
            source_index_sha256=source_sha,
            output_root=release,
            replay_sources=False,
        )


def test_release_validator_rejects_duplicate_case_set_descriptors(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    make_case(outputs)
    universe = CaseUniverse(
        case_ids=(CASE_ID,),
        case_sets=(
            ("caseset-a", (CASE_ID,), "7" * 64),
            ("caseset-b", (CASE_ID,), "6" * 64),
        ),
        support_manifest_sha256="8" * 64,
        case_universe_sha256=hashlib.sha256(f"{CASE_ID}\n".encode()).hexdigest(),
    )
    source_path = tmp_path / "source.json"
    build_source_index(
        universe=universe, outputs_roots=[outputs], output_path=source_path
    )
    source, source_sha = load_source_index(
        source_path, universe=universe, require_complete=True
    )
    release = tmp_path / "release"
    build_case_truth(case_id=CASE_ID, source=source, output_root=release)
    assemble_release(
        universe=universe,
        source_index_sha256=source_sha,
        output_root=release,
        cases_per_chunk=1,
    )
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["case_sets"][1] = dict(manifest["case_sets"][0])
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    receipt_path = release / "release-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = digest(manifest_path)
    receipt_path.write_bytes(canonical_json_bytes(receipt))

    with pytest.raises(NativeProfileTruthError, match="descriptor is duplicated"):
        validate_release(
            universe=universe,
            source=source,
            source_index_sha256=source_sha,
            output_root=release,
            replay_sources=False,
        )


def test_release_validator_binds_chunk_case_descriptors_to_case_records(
    tmp_path: Path,
) -> None:
    universe, source, source_sha, release = _one_case_release(tmp_path)
    chunk_path = release / "chunks" / "chunk-000.json"
    chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
    chunk["cases"][0]["case_record_sha256"] = "0" * 64
    chunk_path.write_bytes(canonical_json_bytes(chunk))

    index_path = release / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["chunks"][0]["sha256"] = digest(chunk_path)
    index["chunks"][0]["byte_size"] = chunk_path.stat().st_size
    index["case_locations"][CASE_ID]["chunk_sha256"] = digest(chunk_path)
    index_path.write_bytes(canonical_json_bytes(index))

    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["master_index"]["sha256"] = digest(index_path)
    manifest["master_index"]["byte_size"] = index_path.stat().st_size
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    receipt_path = release / "release-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = digest(manifest_path)
    receipt_path.write_bytes(canonical_json_bytes(receipt))

    with pytest.raises(NativeProfileTruthError, match="master index location"):
        validate_release(
            universe=universe,
            source=source,
            source_index_sha256=source_sha,
            output_root=release,
            replay_sources=False,
        )


def test_truth_materializer_sparse_vtu_reader_reads_only_selected_field_rows(
    tmp_path: Path,
) -> None:
    coordinates = np.arange(15, dtype="<f4").reshape(5, 3)
    pressure = np.asarray([5.0, 7.0, 11.0, 13.0, 17.0], dtype="<f4")
    coordinate_payload = coordinates.tobytes(order="C")
    pressure_payload = pressure.tobytes(order="C")
    pressure_offset = 8 + len(coordinate_payload)
    prefix = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" byte_order="LittleEndian" '
        'header_type="UInt64">\n'
        "<UnstructuredGrid>\n"
        '<Piece NumberOfPoints="5" NumberOfCells="0">\n'
        "<PointData>\n"
        '<DataArray type="Float32" Name="PROJ(AVG(P))" format="appended" '
        f'offset="{pressure_offset}"/>\n'
        "</PointData>\n"
        "<Points>\n"
        '<DataArray type="Float32" NumberOfComponents="3" format="appended" '
        'offset="0"/>\n'
        "</Points>\n"
        "</Piece>\n"
        "</UnstructuredGrid>\n"
        '<AppendedData encoding="raw">_'
    ).encode()
    suffix = b"</AppendedData>\n</VTKFile>\n"
    path = tmp_path / "boundary.vtu"
    path.write_bytes(
        prefix
        + struct.pack("<Q", len(coordinate_payload))
        + coordinate_payload
        + struct.pack("<Q", len(pressure_payload))
        + pressure_payload
        + suffix
    )
    reader = _SparseVTUReader(path)
    selected, spec = reader.gather(
        "PROJ(AVG(P))", np.asarray([0, 2, 4], dtype=np.int64)
    )
    assert reader.number_of_points == 5
    assert spec.shape == (5,)
    assert np.array_equal(selected, pressure[[0, 2, 4]])


def test_compact_support_inputs_are_prediction_free_and_authority_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp_names = (
        "cut_xyz_in",
        "branch_closed",
        "branch_vertex_offsets",
        "branch_vertex_ids",
        "branch_segment_offsets",
        "segment_lengths_in",
        "branch_row_code",
        "branch_graph_component_code",
        "branch_component_code",
        "branch_plane_piece_code",
        "branch_side_code",
        "branch_topology_patch_code",
    )
    cp_stencil = {name: np.asarray([index]) for index, name in enumerate(cp_names)}
    velocity_stencil = {
        "requested_xyz_in": np.zeros((3, 3)),
        "valid_mask": np.asarray([True, True, True]),
        "station_names": np.asarray(["B.2"]),
        "station_row_offsets": np.asarray([0, 3]),
        "support_raw_point_ids": np.asarray([7], dtype=np.int64),
    }
    monkeypatch.setattr(
        truth_materializer,
        "_load_cp_stencil",
        lambda entry, case_id: (cp_stencil, {}, "b" * 64),
    )
    monkeypatch.setattr(
        truth_materializer,
        "_load_velocity_stencil",
        lambda entry, case_id: (velocity_stencil, {}, "c" * 64),
    )
    monkeypatch.setattr(
        truth_materializer,
        "_load_validity",
        lambda entry, case_id, support: (
            np.asarray([0], dtype=np.int64),
            {},
            "d" * 64,
        ),
    )
    monkeypatch.setattr(
        truth_materializer,
        "_line_weights",
        lambda stencil: np.asarray([0.5, 1.0, 0.5]),
    )
    authority = {
        "cases": {
            CASE_ID: {
                "authority_case_identity_sha256": "a" * 64,
            }
        }
    }
    cp, velocity, evidence = load_compact_profile_support_inputs(
        case_id=CASE_ID,
        authority=authority,
    )
    assert tuple(cp) == cp_names
    assert "prediction_cp" not in cp
    assert tuple(velocity) == (
        "requested_xyz_in",
        "valid_mask",
        "station_names",
        "station_row_offsets",
        "line_length_weights_in",
    )
    assert "predicted_velocity_nd" not in velocity
    assert evidence == {
        "authority_case_identity_sha256": "a" * 64,
        "cp_stencil_identity_sha256": "b" * 64,
        "velocity_stencil_identity_sha256": "c" * 64,
        "validity_identity_sha256": "d" * 64,
    }


def test_truth_materializer_sparse_profile_formulas_are_lossless() -> None:
    cp_stencil = {
        "cut_support_node_ids": np.asarray([0, 1, 1, 2], dtype=np.int64),
        "cut_support_offsets": np.asarray([0, 2, 4], dtype=np.int64),
        "cut_support_weights": np.asarray([0.25, 0.75, 0.5, 0.5]),
    }
    truth_cp = _reconstruct_cp(
        cp_stencil, np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
    )
    assert truth_cp.dtype == np.dtype("<f8")
    assert np.array_equal(truth_cp, np.asarray([1.75, 3.0]))

    velocity_stencil = {
        "support_raw_point_ids": np.asarray([10, 20], dtype=np.int64),
        "raw_point_ids": np.asarray([10, 20, 10, 20], dtype=np.int64),
        "csr_offsets": np.asarray([0, 2, 2, 4], dtype=np.int64),
        "interpolation_weights": np.asarray([0.25, 0.75, 0.5, 0.5]),
        "valid_mask": np.asarray([True, False, True]),
    }
    support_velocity = np.asarray([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], dtype=np.float32)
    truth_velocity = _reconstruct_velocity(velocity_stencil, support_velocity)
    assert truth_velocity.dtype == np.dtype("<f8")
    assert np.array_equal(
        truth_velocity[[0, 2]], np.asarray([[4.0, 5.0, 6.0], [3.0, 4.0, 5.0]])
    )
    assert np.all(np.isnan(truth_velocity[1]))

    line_stencil = {
        "requested_xyz_in": np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]
        ),
        "valid_mask": np.asarray([True, True, True]),
        "station_row_offsets": np.asarray([0, 3], dtype=np.int64),
    }
    assert np.array_equal(_line_weights(line_stencil), [0.5, 1.0, 0.5])
