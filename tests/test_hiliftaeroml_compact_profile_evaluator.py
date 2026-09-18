from __future__ import annotations

import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

import reference.hiliftaeroml.compact_profile_evaluator as compact_evaluator
from reference.hiliftaeroml.compact_profile_evaluator import (
    COMPACT_PROFILE_CONTRACT_ID,
    COMPACT_PROFILE_CONTRACT_SHA256,
    COMPACT_PROFILE_FORMAT,
    COMPACT_PROFILE_INDEX_SCHEMA_VERSION,
    COMPACT_SUPPORT_RELEASE_ID,
    SOURCE_ARTIFACT_SHA256_KEYS,
    CompactProfileEvaluationError,
    build_compact_profile_directory,
    open_compact_support_release,
    score_compact_profile_directory,
    write_compact_support_release,
)
from reference.hiliftaeroml.compact_profiles import (
    PREDICTION_ARRAYS,
    build_compact_support,
)
from reference.hiliftaeroml.native_profiles import (
    CP_SOURCE_ARRAYS,
    VELOCITY_SOURCE_ARRAYS,
    canonical_json_bytes,
    validate_cp_source,
    validate_velocity_source,
)
from tests.test_hiliftaeroml_native_profiles import (
    CASE_ID,
    make_case,
    retained_profile_hashes,
    write_ordered_npz,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_SET_ID = "caseset-compact-test"
SUBMISSION_ID = "hilift-compact-test"
SPLIT_ID = "full"
SOURCE_TRUTH_RELEASE_ID = "hiliftaeroml-native-profile-truth-v1-candidate"
SOURCE_TRUTH_MANIFEST_SHA256 = "a" * 64
INACTIVE_ACTIVATION = {
    "owner_approval_complete": False,
    "published": False,
    "submissions_opened": False,
    "candidate_validation_may_change_activation": False,
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _directory_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _make_varying_native_case(outputs: Path) -> dict[str, dict[str, str]]:
    """Make the shared native fixture's velocity truth non-constant per station."""

    make_case(outputs)
    volume = outputs / CASE_ID / "volume_submission_stream"
    npz_path = volume / "velocity_profiles.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in VELOCITY_SOURCE_ARRAYS
        }
    valid = arrays["valid_mask"]
    predicted = arrays["predicted_velocity_nd"]
    local_row = np.arange(len(valid), dtype=np.float64) % 801.0
    reference = np.array(predicted, copy=True)
    reference[valid, 0] += 0.1 + 0.001 * (local_row[valid] % 101.0)
    reference[valid, 1] += 0.05 * (local_row[valid] % 7.0)
    reference[valid, 2] -= 0.02 * (local_row[valid] % 5.0)
    reference[~valid] = np.nan
    reference_speed = np.linalg.norm(reference, axis=1)
    arrays["reference_velocity_nd"] = reference
    arrays["reference_velocity_physical"] = reference * 2.0
    arrays["reference_velocity_dimensional"] = reference * 2.0
    arrays["reference_velocity_magnitude_nd"] = reference_speed
    arrays["u_x_over_U_inf_reference"] = reference[:, 0]
    arrays["u_y_over_U_inf_reference"] = reference[:, 1]
    arrays["u_z_over_U_inf_reference"] = reference[:, 2]
    arrays["speed_over_U_inf_reference"] = reference_speed
    npz_path.unlink()
    write_ordered_npz(npz_path, arrays, VELOCITY_SOURCE_ARRAYS)

    metrics_path = volume / "velocity_profile_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["profile_npz"]["sha256"] = _digest(npz_path)
    metrics["profile_npz"]["size_bytes"] = npz_path.stat().st_size
    _write_canonical_json(metrics_path, metrics)
    return retained_profile_hashes(outputs)


def _support_from_native(
    outputs: Path, source_hashes: dict[str, dict[str, str]]
) -> dict[str, np.ndarray]:
    surface = outputs / CASE_ID / "surface_submission_stream"
    volume = outputs / CASE_ID / "volume_submission_stream"
    expected = source_hashes[CASE_ID]
    cp, _ = validate_cp_source(
        metrics_path=surface / "cp_profile_metrics.json",
        npz_path=surface / "cp_cut_values.npz",
        expected_metrics_sha256=expected["cp_profile_metrics"],
        expected_npz_sha256=expected["cp_cut_values"],
    )
    velocity, _ = validate_velocity_source(
        case_id=CASE_ID,
        metrics_path=volume / "velocity_profile_metrics.json",
        npz_path=volume / "velocity_profiles.npz",
        expected_metrics_sha256=expected["velocity_profile_metrics"],
        expected_npz_sha256=expected["velocity_profiles"],
    )
    return build_compact_support(
        cp_native=cp,
        truth_cp=cp["truth_cp"],
        velocity_native=velocity,
        truth_velocity_nd=velocity["reference_velocity_nd"],
    )


def _make_release(
    tmp_path: Path,
    outputs: Path,
    *,
    name: str = "support-release",
) -> tuple[Path, str, dict[str, dict[str, str]]]:
    source_hashes = _make_varying_native_case(outputs)
    support = _support_from_native(outputs, source_hashes)
    release = tmp_path / name
    manifest_sha = write_compact_support_release(
        release_root=release,
        case_ids=[CASE_ID],
        case_sets={CASE_SET_ID: [CASE_ID]},
        supports={CASE_ID: support},
        source_artifact_sha256=source_hashes,
        source_profile_truth_release_id=SOURCE_TRUTH_RELEASE_ID,
        source_profile_truth_manifest_sha256=SOURCE_TRUTH_MANIFEST_SHA256,
    )
    return release, manifest_sha, source_hashes


def _change_cp_prediction_only(outputs: Path) -> dict[str, dict[str, str]]:
    surface = outputs / CASE_ID / "surface_submission_stream"
    npz_path = surface / "cp_cut_values.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True) for name in CP_SOURCE_ARRAYS
        }
    arrays["prediction_cp"] += 0.03125
    npz_path.unlink()
    write_ordered_npz(npz_path, arrays, CP_SOURCE_ARRAYS)
    metrics_path = surface / "cp_profile_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["cut_values"]["sha256"] = _digest(npz_path)
    metrics["cut_values"]["size_bytes"] = npz_path.stat().st_size
    _write_canonical_json(metrics_path, metrics)
    return retained_profile_hashes(outputs)


def _build_directory(
    *,
    outputs: Path,
    release: Path,
    manifest_sha: str,
    source_hashes: dict[str, dict[str, str]],
    destination: Path,
    opened_support_release: compact_evaluator.CompactSupportRelease | None = None,
) -> tuple[str, dict[str, dict[str, float]]]:
    return build_compact_profile_directory(
        submission_id=SUBMISSION_ID,
        split_id=SPLIT_ID,
        case_set_id=CASE_SET_ID,
        case_ids=[CASE_ID],
        support_release_root=release,
        support_manifest_sha256=manifest_sha,
        outputs_root=outputs,
        profiles_root=destination,
        cases_per_chunk=1,
        expected_case_artifact_sha256=source_hashes,
        opened_support_release=opened_support_release,
    )


def _refresh_chunk_binding(profiles: Path, chunk: dict[str, object]) -> None:
    chunk_path = profiles / "chunk-000.json"
    _write_canonical_json(chunk_path, chunk)
    index_path = profiles / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["chunks"][0]["sha256"] = _digest(chunk_path)
    _write_canonical_json(index_path, index)


def _header_only_npy(dtype: np.dtype[object], shape: tuple[int, ...]) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        output,
        {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return output.getvalue()


def test_release_and_participant_directory_are_strict_private_and_deterministic(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    release, manifest_sha, source_hashes = _make_release(tmp_path, outputs)
    handle = open_compact_support_release(
        release_root=release,
        expected_manifest_sha256=manifest_sha,
        expected_case_ids=[CASE_ID],
        case_set_id=CASE_SET_ID,
    )
    assert handle.release_id == COMPACT_SUPPORT_RELEASE_ID
    assert handle.manifest_sha256 == manifest_sha
    assert handle.source_profile_truth_release_id == SOURCE_TRUTH_RELEASE_ID
    assert (
        handle.source_profile_truth_manifest_sha256
        == SOURCE_TRUTH_MANIFEST_SHA256
    )
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    support_index = json.loads(
        (release / "index.json").read_text(encoding="utf-8")
    )
    assert manifest["activation"] == INACTIVE_ACTIVATION
    assert support_index["activation"] == INACTIVE_ACTIVATION
    assert stat.S_IMODE(release.stat().st_mode) == 0o700
    for path in release.rglob("*"):
        expected_mode = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected_mode

    first = tmp_path / "profiles-a"
    second = tmp_path / "profiles-b"
    first_sha, first_metrics = _build_directory(
        outputs=outputs,
        release=release,
        manifest_sha=manifest_sha,
        source_hashes=source_hashes,
        destination=first,
    )
    second_sha, second_metrics = _build_directory(
        outputs=outputs,
        release=release,
        manifest_sha=manifest_sha,
        source_hashes=source_hashes,
        destination=second,
    )
    assert first_sha == second_sha == _digest(first / "index.json")
    assert first_metrics == second_metrics
    assert set(first_metrics[CASE_ID]) == {"cp_cut_r2", "velocity_profile_r2"}
    assert _directory_bytes(first) == _directory_bytes(second)

    index = json.loads((first / "index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == COMPACT_PROFILE_INDEX_SCHEMA_VERSION
    assert index["format"] == COMPACT_PROFILE_FORMAT
    assert index["contract_id"] == COMPACT_PROFILE_CONTRACT_ID
    assert index["contract_sha256"] == COMPACT_PROFILE_CONTRACT_SHA256
    assert index["evaluator_support_release_id"] == COMPACT_SUPPORT_RELEASE_ID
    assert index["evaluator_support_manifest_sha256"] == manifest_sha
    index_schema = json.loads(
        (ROOT / "schemas" / "v1" / "profile-index.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(Draft202012Validator(index_schema).iter_errors(index)) == []
    chunk = json.loads((first / "chunk-000.json").read_text(encoding="utf-8"))
    schema = json.loads(
        (
            ROOT
            / "schemas"
            / "v1"
            / "hiliftaeroml-compact-profile-chunk.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert list(Draft202012Validator(schema).iter_errors(chunk)) == []
    artifact = (
        first / "artifacts" / CASE_ID / "compact-profile-predictions.npz"
    )
    with np.load(artifact, allow_pickle=False) as archive:
        assert archive.files == list(PREDICTION_ARRAYS)
        assert not any(
            "truth" in name
            or "reference" in name
            or "support" in name
            or "coordinate" in name
            for name in archive.files
        )
    assert not any("support" in path.name for path in first.rglob("*.npz"))
    assert score_compact_profile_directory(
        profiles_root=first,
        support_release_root=release,
        support_manifest_sha256=manifest_sha,
        submission_id=SUBMISSION_ID,
        split_id=SPLIT_ID,
        case_set_id=CASE_SET_ID,
        expected_case_ids=[CASE_ID],
    ) == first_metrics


def test_preopened_support_handle_preserves_exact_bindings_and_scores(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    release, manifest_sha, source_hashes = _make_release(tmp_path, outputs)
    handle = open_compact_support_release(
        release_root=release,
        expected_manifest_sha256=manifest_sha,
        expected_case_ids=[CASE_ID],
        case_set_id=CASE_SET_ID,
    )
    profiles = tmp_path / "profiles"
    _index_sha, expected_metrics = _build_directory(
        outputs=outputs,
        release=release,
        manifest_sha=manifest_sha,
        source_hashes=source_hashes,
        destination=profiles,
        opened_support_release=handle,
    )
    assert score_compact_profile_directory(
        profiles_root=profiles,
        support_release_root=release,
        support_manifest_sha256=manifest_sha,
        submission_id=SUBMISSION_ID,
        split_id=SPLIT_ID,
        case_set_id=CASE_SET_ID,
        expected_case_ids=[CASE_ID],
        opened_support_release=handle,
    ) == expected_metrics
    with pytest.raises(
        CompactProfileEvaluationError, match="requested binding"
    ):
        score_compact_profile_directory(
            profiles_root=profiles,
            support_release_root=release,
            support_manifest_sha256="0" * 64,
            submission_id=SUBMISSION_ID,
            split_id=SPLIT_ID,
            case_set_id=CASE_SET_ID,
            expected_case_ids=[CASE_ID],
            opened_support_release=handle,
        )


def test_release_loader_rejects_wrong_binding_case_set_and_extra_file(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    release, manifest_sha, _ = _make_release(tmp_path, outputs)
    with pytest.raises(
        CompactProfileEvaluationError, match="explicit binding"
    ):
        open_compact_support_release(
            release_root=release,
            expected_manifest_sha256="0" * 64,
            expected_case_ids=[CASE_ID],
            case_set_id=CASE_SET_ID,
        )
    with pytest.raises(
        CompactProfileEvaluationError, match="selected case-set binding"
    ):
        open_compact_support_release(
            release_root=release,
            expected_manifest_sha256=manifest_sha,
            expected_case_ids=[CASE_ID],
            case_set_id="caseset-wrong",
        )
    (release / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(
        CompactProfileEvaluationError, match="inventory differs"
    ):
        open_compact_support_release(
            release_root=release,
            expected_manifest_sha256=manifest_sha,
            expected_case_ids=[CASE_ID],
            case_set_id=CASE_SET_ID,
        )


def test_builder_rejects_receipt_mismatch_and_removes_partial_output(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    release, manifest_sha, source_hashes = _make_release(tmp_path, outputs)
    wrong = {
        CASE_ID: {
            key: value
            for key, value in source_hashes[CASE_ID].items()
        }
    }
    assert set(wrong[CASE_ID]) == set(SOURCE_ARTIFACT_SHA256_KEYS)
    wrong[CASE_ID]["velocity_profiles"] = "0" * 64
    destination = tmp_path / "profiles-wrong-receipt"
    with pytest.raises(
        CompactProfileEvaluationError, match="retained receipt"
    ):
        _build_directory(
            outputs=outputs,
            release=release,
            manifest_sha=manifest_sha,
            source_hashes=wrong,
            destination=destination,
        )
    assert not destination.exists()


def test_support_materialization_provenance_does_not_pin_one_surrogate(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    release, manifest_sha, materialization_hashes = _make_release(
        tmp_path, outputs
    )
    current_hashes = _change_cp_prediction_only(outputs)
    assert current_hashes[CASE_ID] != materialization_hashes[CASE_ID]

    destination = tmp_path / "profiles-second-surrogate"
    index_sha, metrics = _build_directory(
        outputs=outputs,
        release=release,
        manifest_sha=manifest_sha,
        source_hashes=current_hashes,
        destination=destination,
    )
    assert index_sha == _digest(destination / "index.json")
    assert set(metrics[CASE_ID]) == {"cp_cut_r2", "velocity_profile_r2"}


def test_scorer_rejects_rebound_metadata_even_with_updated_chunk_hash(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    release, manifest_sha, source_hashes = _make_release(tmp_path, outputs)
    profiles = tmp_path / "profiles"
    _build_directory(
        outputs=outputs,
        release=release,
        manifest_sha=manifest_sha,
        source_hashes=source_hashes,
        destination=profiles,
    )
    chunk = json.loads((profiles / "chunk-000.json").read_text(encoding="utf-8"))
    chunk["cases"][0]["surface_cp"]["support_identity_sha256"] = "0" * 64
    _refresh_chunk_binding(profiles, chunk)
    with pytest.raises(
        CompactProfileEvaluationError, match="metadata differs"
    ):
        score_compact_profile_directory(
            profiles_root=profiles,
            support_release_root=release,
            support_manifest_sha256=manifest_sha,
            submission_id=SUBMISSION_ID,
            split_id=SPLIT_ID,
            case_set_id=CASE_SET_ID,
            expected_case_ids=[CASE_ID],
        )


def test_scorer_rejects_noncanonical_artifact_and_extra_inventory(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    release, manifest_sha, source_hashes = _make_release(tmp_path, outputs)
    profiles = tmp_path / "profiles"
    _build_directory(
        outputs=outputs,
        release=release,
        manifest_sha=manifest_sha,
        source_hashes=source_hashes,
        destination=profiles,
    )
    artifact_path = (
        profiles / "artifacts" / CASE_ID / "compact-profile-predictions.npz"
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    artifact_path.unlink()
    np.savez(
        artifact_path,
        **{name: arrays[name] for name in reversed(PREDICTION_ARRAYS)},
    )
    chunk = json.loads((profiles / "chunk-000.json").read_text(encoding="utf-8"))
    chunk["cases"][0]["artifact"]["sha256"] = _digest(artifact_path)
    chunk["cases"][0]["artifact"]["byte_size"] = artifact_path.stat().st_size
    _refresh_chunk_binding(profiles, chunk)
    with pytest.raises(
        CompactProfileEvaluationError, match="array inventory or order differs"
    ):
        score_compact_profile_directory(
            profiles_root=profiles,
            support_release_root=release,
            support_manifest_sha256=manifest_sha,
            submission_id=SUBMISSION_ID,
            split_id=SPLIT_ID,
            case_set_id=CASE_SET_ID,
            expected_case_ids=[CASE_ID],
        )

    # Restore a fresh valid directory and show that an otherwise innocuous file
    # is also rejected rather than ignored.
    clean = tmp_path / "profiles-clean"
    _build_directory(
        outputs=outputs,
        release=release,
        manifest_sha=manifest_sha,
        source_hashes=source_hashes,
        destination=clean,
    )
    (clean / "notes.txt").write_text("not part of the contract\n", encoding="utf-8")
    with pytest.raises(
        CompactProfileEvaluationError, match="inventory differs"
    ):
        score_compact_profile_directory(
            profiles_root=clean,
            support_release_root=release,
            support_manifest_sha256=manifest_sha,
            submission_id=SUBMISSION_ID,
            split_id=SPLIT_ID,
            case_set_id=CASE_SET_ID,
            expected_case_ids=[CASE_ID],
        )


def test_preflight_rejects_huge_declared_shape_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    release, manifest_sha, source_hashes = _make_release(tmp_path, outputs)
    profiles = tmp_path / "profiles"
    _build_directory(
        outputs=outputs,
        release=release,
        manifest_sha=manifest_sha,
        source_hashes=source_hashes,
        destination=profiles,
    )
    artifact_path = (
        profiles / "artifacts" / CASE_ID / "compact-profile-predictions.npz"
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        speed = np.array(archive["velocity_speed_over_u_inf"], copy=True)
    with zipfile.ZipFile(
        artifact_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr(
            "cp_q_delta.npy",
            _header_only_npy(np.dtype(np.int16), (2_147_483_648,)),
        )
        speed_buffer = io.BytesIO()
        np.lib.format.write_array(speed_buffer, speed, allow_pickle=False)
        archive.writestr(
            "velocity_speed_over_u_inf.npy", speed_buffer.getvalue()
        )
    chunk = json.loads((profiles / "chunk-000.json").read_text(encoding="utf-8"))
    chunk["cases"][0]["artifact"]["sha256"] = _digest(artifact_path)
    chunk["cases"][0]["artifact"]["byte_size"] = artifact_path.stat().st_size
    _refresh_chunk_binding(profiles, chunk)

    def forbidden_numpy_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("NumPy compact loader ran before bounded preflight")

    monkeypatch.setattr(
        compact_evaluator, "load_compact_prediction_npz", forbidden_numpy_load
    )
    with pytest.raises(
        CompactProfileEvaluationError, match="bounded array contract"
    ):
        score_compact_profile_directory(
            profiles_root=profiles,
            support_release_root=release,
            support_manifest_sha256=manifest_sha,
            submission_id=SUBMISSION_ID,
            split_id=SPLIT_ID,
            case_set_id=CASE_SET_ID,
            expected_case_ids=[CASE_ID],
        )


def test_release_open_validates_unselected_live_artifacts(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    source_hashes = _make_varying_native_case(outputs)
    support = _support_from_native(outputs, source_hashes)
    other_case = "geo_LHC002_AoA_4"
    release = tmp_path / "support-release"
    manifest_sha = write_compact_support_release(
        release_root=release,
        case_ids=[CASE_ID, other_case],
        case_sets={CASE_SET_ID: [CASE_ID], "caseset-other": [other_case]},
        supports={CASE_ID: support, other_case: support},
        source_artifact_sha256={
            CASE_ID: source_hashes[CASE_ID],
            other_case: source_hashes[CASE_ID],
        },
        source_profile_truth_release_id=SOURCE_TRUTH_RELEASE_ID,
        source_profile_truth_manifest_sha256=SOURCE_TRUTH_MANIFEST_SHA256,
    )
    unselected = (
        release
        / "artifacts"
        / other_case
        / "compact-profile-support.npz"
    )
    payload = bytearray(unselected.read_bytes())
    payload[-1] ^= 0x01
    unselected.write_bytes(payload)
    assert stat.S_IMODE(unselected.stat().st_mode) == 0o600
    with pytest.raises(CompactProfileEvaluationError, match="live bytes differ"):
        open_compact_support_release(
            release_root=release,
            expected_manifest_sha256=manifest_sha,
            expected_case_ids=[CASE_ID],
            case_set_id=CASE_SET_ID,
        )


def test_exact_json_types_and_private_modes_are_enforced(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    release, manifest_sha, _ = _make_release(tmp_path, outputs)
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["privacy_boundary"]["participant_visible"] = 0
    _write_canonical_json(manifest_path, manifest)
    wrong_manifest_sha = _digest(manifest_path)
    with pytest.raises(
        CompactProfileEvaluationError, match="must be boolean False"
    ):
        open_compact_support_release(
            release_root=release,
            expected_manifest_sha256=wrong_manifest_sha,
            expected_case_ids=[CASE_ID],
            case_set_id=CASE_SET_ID,
        )

    fresh_outputs = tmp_path / "fresh-outputs"
    fresh_release, fresh_sha, _ = _make_release(
        tmp_path, fresh_outputs, name="fresh-support-release"
    )
    (fresh_release / "manifest.json").chmod(0o644)
    with pytest.raises(
        CompactProfileEvaluationError, match="ownership/mode differs"
    ):
        open_compact_support_release(
            release_root=fresh_release,
            expected_manifest_sha256=fresh_sha,
            expected_case_ids=[CASE_ID],
            case_set_id=CASE_SET_ID,
        )


def test_builders_stage_before_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = tmp_path / "outputs"
    release, manifest_sha, source_hashes = _make_release(tmp_path, outputs)
    profiles = tmp_path / "profiles-interrupted"
    with monkeypatch.context() as context:
        context.setattr(
            compact_evaluator,
            "write_compact_prediction_npz",
            lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            _build_directory(
                outputs=outputs,
                release=release,
                manifest_sha=manifest_sha,
                source_hashes=source_hashes,
                destination=profiles,
            )
    assert not profiles.exists()
    assert list(tmp_path.glob(".profiles-interrupted.staging-*")) == []

    support = _support_from_native(outputs, source_hashes)
    interrupted_release = tmp_path / "release-interrupted"
    with monkeypatch.context() as context:
        context.setattr(
            compact_evaluator,
            "write_compact_support_npz",
            lambda *args, **kwargs: (_ for _ in ()).throw(SystemExit(9)),
        )
        with pytest.raises(SystemExit, match="9"):
            write_compact_support_release(
                release_root=interrupted_release,
                case_ids=[CASE_ID],
                case_sets={CASE_SET_ID: [CASE_ID]},
                supports={CASE_ID: support},
                source_artifact_sha256=source_hashes,
                source_profile_truth_release_id=SOURCE_TRUTH_RELEASE_ID,
                source_profile_truth_manifest_sha256=(
                    SOURCE_TRUTH_MANIFEST_SHA256
                ),
            )
    assert not interrupted_release.exists()
    assert list(tmp_path.glob(".release-interrupted.staging-*")) == []
