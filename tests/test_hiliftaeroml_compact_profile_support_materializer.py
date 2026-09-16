from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import scripts.materialize_hiliftaeroml_compact_profile_support as materializer
from reference.hiliftaeroml.compact_profile_evaluator import (
    COMPACT_PROFILE_CONTRACT_ID,
    COMPACT_PROFILE_CONTRACT_SHA256,
    COMPACT_SUPPORT_RELEASE_ID,
    COMPACT_SUPPORT_RELEASE_STATUS,
    COMPACT_SUPPORT_RELEASE_USAGE,
    SOURCE_ARTIFACT_SHA256_KEYS,
    open_compact_support_release,
)
from reference.hiliftaeroml.compact_profiles import (
    COMPACT_PROFILE_FORMAT,
    CP_POINTS_PER_GRAPH,
    load_compact_support_npz,
)
from reference.hiliftaeroml.native_profile_evaluator import CandidateTruthRelease
from reference.hiliftaeroml.native_profile_truth import (
    CaseUniverse,
    assemble_release,
    build_case_truth,
    build_source_index,
    load_source_index,
)
from reference.hiliftaeroml.native_profiles import (
    CP_SOURCE_ARRAYS,
    PROFILE_CONTRACT_ID,
    PROFILE_CONTRACT_SHA256,
    canonical_json_bytes,
)
from tests.test_hiliftaeroml_native_profiles import (
    CASE_ID,
    make_case,
    write_ordered_npz,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_SET_ID = "caseset-test"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _truth_release(outputs: Path, root: Path) -> tuple[CandidateTruthRelease, str]:
    case_ids = (CASE_ID,)
    universe = CaseUniverse(
        case_ids=case_ids,
        case_sets=((CASE_SET_ID, case_ids, "9" * 64),),
        support_manifest_sha256="8" * 64,
        case_universe_sha256=hashlib.sha256(
            f"{CASE_ID}\n".encode("utf-8")
        ).hexdigest(),
    )
    source_path = root / "source-index.json"
    build_source_index(
        universe=universe,
        outputs_roots=[outputs],
        output_path=source_path,
    )
    source, source_sha = load_source_index(
        source_path, universe=universe, require_complete=True
    )
    release_root = root / "release"
    build_case_truth(case_id=CASE_ID, source=source, output_root=release_root)
    manifest = assemble_release(
        universe=universe,
        source_index_sha256=source_sha,
        output_root=release_root,
        cases_per_chunk=1,
    )
    index = json.loads((release_root / "index.json").read_text(encoding="utf-8"))
    manifest_sha = _digest(release_root / "manifest.json")
    return (
        CandidateTruthRelease(
            campaign_root=root,
            release_root=release_root.resolve(),
            binding={},
            manifest=manifest,
            index=index,
            case_set_id=CASE_SET_ID,
            case_ids=case_ids,
        ),
        manifest_sha,
    )


def _poison_raw_truth_copy(outputs: Path) -> None:
    surface = outputs / CASE_ID / "surface_submission_stream"
    npz_path = surface / "cp_cut_values.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["truth_cp"] += 100.0
    write_ordered_npz(npz_path, arrays, CP_SOURCE_ARRAYS)
    metrics_path = surface / "cp_profile_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["cut_values"]["sha256"] = _digest(npz_path)
    metrics["cut_values"]["size_bytes"] = npz_path.stat().st_size
    _write_json(metrics_path, metrics)


def _benchmark_specification(root: Path, truth_manifest_sha: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    contract_source = (
        ROOT / "benchmark-specs/hiliftaeroml/native-profile-format-v2.json"
    )
    contract_path = root / "native-profile-format-v2.json"
    contract_path.write_bytes(contract_source.read_bytes())
    assert _digest(contract_path) == COMPACT_PROFILE_CONTRACT_SHA256

    split = {
        "schema_version": "1.0",
        "dataset_id": "hiliftaeroml",
        "split_id": "full",
        "case_set_id": CASE_SET_ID,
        "split_label": "Full fixture",
        "case_id_status": "official",
        "case_count": 1,
        "case_set_sha256": hashlib.sha256(
            f"{CASE_ID}\n".encode("utf-8")
        ).hexdigest(),
        "case_set_sha256_rule": materializer.CASE_SET_SHA256_RULE,
        "case_ids": [CASE_ID],
    }
    split_path = root / "splits/full.json"
    split_sha = _write_json(split_path, split)
    binding_path = root / "candidate-profile-truth-binding.json"
    binding_sha = _write_json(binding_path, {})
    spec = {
        "dataset_id": "hiliftaeroml",
        "profile_definition": {
            "contract_id": PROFILE_CONTRACT_ID,
            "sha256": PROFILE_CONTRACT_SHA256,
            "candidate_dry_run_profile_ground_truth": {
                "status": "complete_candidate_not_published",
                "usage": "maintainer_local_candidate_dry_run_only",
                "release_id": "hiliftaeroml-native-profile-truth-v1-candidate",
                "manifest_sha256": truth_manifest_sha,
                "binding_file": binding_path.name,
                "binding_sha256": binding_sha,
            },
        },
        "compact_profile_definition": {
            "contract_id": COMPACT_PROFILE_CONTRACT_ID,
            "file": contract_path.name,
            "format": COMPACT_PROFILE_FORMAT,
            "sha256": COMPACT_PROFILE_CONTRACT_SHA256,
            "candidate_dry_run_evaluator_support": {
                "status": COMPACT_SUPPORT_RELEASE_STATUS,
                "usage": COMPACT_SUPPORT_RELEASE_USAGE,
                "release_id": COMPACT_SUPPORT_RELEASE_ID,
                "manifest_sha256": None,
            },
        },
        "splits": [
            {
                "id": "full",
                "label": "Full fixture",
                "index_file": "splits/full.json",
                "case_count": 1,
                "case_set_id": CASE_SET_ID,
                "case_id_status": "official",
                "sha256": split_sha,
            }
        ],
    }
    spec_path = root / "submission-spec.json"
    _write_json(spec_path, spec)
    return spec_path, split_path


def _arguments(
    spec: Path, split: Path, outputs: Path, truth: Path, output: Path
) -> list[str]:
    return [
        "--submission-spec",
        str(spec),
        "--split",
        str(split),
        "--surface-outputs-root",
        str(outputs),
        "--volume-outputs-root",
        str(outputs),
        "--source-truth-release",
        str(truth),
        "--output-root",
        str(output),
    ]


def test_materializer_is_truth_safe_exclusive_and_ab_deterministic(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    outputs = tmp_path / "outputs"
    make_case(outputs)
    truth_handle, truth_manifest_sha = _truth_release(
        outputs, tmp_path / "truth-campaign"
    )
    # A validated native output may carry a truth copy for historical reasons,
    # but the compact materializer must never select it as authority.
    _poison_raw_truth_copy(outputs)
    spec, split = _benchmark_specification(
        tmp_path / "benchmark-specs/hiliftaeroml", truth_manifest_sha
    )

    def fake_open_candidate_truth_release(**kwargs):
        assert kwargs["expected_case_ids"] == (CASE_ID,)
        assert kwargs["case_set_id"] == CASE_SET_ID
        assert kwargs["release_root"] == truth_handle.release_root
        return truth_handle

    monkeypatch.setattr(
        materializer,
        "open_candidate_truth_release",
        fake_open_candidate_truth_release,
    )

    releases = [tmp_path / "support-a", tmp_path / "support-b"]
    receipts = []
    for release in releases:
        assert (
            materializer.main(
                _arguments(
                    spec,
                    split,
                    outputs,
                    truth_handle.release_root,
                    release,
                )
            )
            == 0
        )
        captured = capsys.readouterr()
        assert captured.err == f"compact-support 1/1 {CASE_ID}\n"
        receipts.append(json.loads(captured.out))

    assert _tree_hashes(releases[0]) == _tree_hashes(releases[1])
    assert (
        receipts[0]["evaluator_support_manifest_sha256"]
        == receipts[1]["evaluator_support_manifest_sha256"]
    )
    assert receipts[0]["points_per_physical_graph"] == CP_POINTS_PER_GRAPH == 128
    assert receipts[0]["activation"] == {
        "owner_approval_complete": False,
        "published": False,
        "submissions_opened": False,
        "materialization_changes_activation": False,
    }

    manifest_sha = receipts[0]["evaluator_support_manifest_sha256"]
    opened = open_compact_support_release(
        release_root=releases[0],
        expected_manifest_sha256=manifest_sha,
        expected_case_ids=[CASE_ID],
        case_set_id=CASE_SET_ID,
    )
    assert opened.source_profile_truth_release_id == (
        "hiliftaeroml-native-profile-truth-v1-candidate"
    )
    assert opened.source_profile_truth_manifest_sha256 == truth_manifest_sha
    record = opened.case_records[CASE_ID]
    assert tuple(record["source_artifact_sha256"]) == SOURCE_ARTIFACT_SHA256_KEYS
    identity_digests = (
        record["surface_cp"]["support_identity_sha256"],
        record["surface_cp"]["prediction_order_sha256"],
        record["volume_velocity"]["support_identity_sha256"],
        record["volume_velocity"]["prediction_order_sha256"],
    )
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in identity_digests
    )
    support_path = releases[0] / record["support"]["file"]
    support, support_sha = load_compact_support_npz(support_path)
    assert support_sha == record["support"]["sha256"]
    assert support_path.stat().st_size == record["support"]["byte_size"]
    # The raw source truth copy was shifted by +100 after the hidden release
    # was built; the compact truth still comes from the hidden release.
    assert support["cp_truth"].tolist() == [0.0, 0.25, 0.5]

    # The first pass produces the digest that maintainers pin into the spec.
    # Rebuilding against that pin is byte-identical; a different pin leaves no
    # output release behind.
    spec_body = json.loads(spec.read_text(encoding="utf-8"))
    support_declaration = spec_body["compact_profile_definition"][
        "candidate_dry_run_evaluator_support"
    ]
    support_declaration["manifest_sha256"] = manifest_sha
    _write_json(spec, spec_body)
    pinned = tmp_path / "support-pinned"
    assert (
        materializer.main(
            _arguments(
                spec,
                split,
                outputs,
                truth_handle.release_root,
                pinned,
            )
        )
        == 0
    )
    capsys.readouterr()
    assert _tree_hashes(pinned) == _tree_hashes(releases[0])
    support_declaration["manifest_sha256"] = "0" * 64
    _write_json(spec, spec_body)
    wrong_pin = tmp_path / "support-wrong-pin"
    assert (
        materializer.main(
            _arguments(
                spec,
                split,
                outputs,
                truth_handle.release_root,
                wrong_pin,
            )
        )
        == 2
    )
    assert not wrong_pin.exists()
    capsys.readouterr()
    support_declaration["manifest_sha256"] = manifest_sha
    _write_json(spec, spec_body)

    before = _tree_hashes(releases[0])
    assert (
        materializer.main(
            _arguments(
                spec,
                split,
                outputs,
                truth_handle.release_root,
                releases[0],
            )
        )
        == 2
    )
    assert _tree_hashes(releases[0]) == before
    capsys.readouterr()

    # A source failure after the writer creates its root removes all partial
    # bytes, so no incomplete release can retain a manifest-looking boundary.
    metrics_path = (
        outputs
        / CASE_ID
        / "surface_submission_stream"
        / "cp_profile_metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["status"] = "incomplete"
    _write_json(metrics_path, metrics)
    failed = tmp_path / "support-failed"
    assert (
        materializer.main(
            _arguments(
                spec,
                split,
                outputs,
                truth_handle.release_root,
                failed,
            )
        )
        == 2
    )
    assert not failed.exists()


def test_authority_materializer_is_prediction_free_and_support_identical(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    outputs = tmp_path / "outputs"
    make_case(outputs)
    truth_handle, truth_manifest_sha = _truth_release(
        outputs, tmp_path / "truth-campaign"
    )
    spec, split = _benchmark_specification(
        tmp_path / "benchmark-specs/hiliftaeroml", truth_manifest_sha
    )

    def fake_open_candidate_truth_release(**kwargs):
        assert kwargs["expected_case_ids"] == (CASE_ID,)
        assert kwargs["case_set_id"] == CASE_SET_ID
        return truth_handle

    monkeypatch.setattr(
        materializer,
        "open_candidate_truth_release",
        fake_open_candidate_truth_release,
    )
    native_release = tmp_path / "native-output-support"
    assert (
        materializer.main(
            _arguments(
                spec,
                split,
                outputs,
                truth_handle.release_root,
                native_release,
            )
        )
        == 0
    )
    native_receipt = json.loads(capsys.readouterr().out)
    native_handle = open_compact_support_release(
        release_root=native_release,
        expected_manifest_sha256=native_receipt[
            "evaluator_support_manifest_sha256"
        ],
        expected_case_ids=[CASE_ID],
        case_set_id=CASE_SET_ID,
    )

    surface = outputs / CASE_ID / "surface_submission_stream"
    volume = outputs / CASE_ID / "volume_submission_stream"
    cp_native, _ = materializer.validate_cp_source(
        metrics_path=surface / "cp_profile_metrics.json",
        npz_path=surface / "cp_cut_values.npz",
    )
    velocity_native, _ = materializer.validate_velocity_source(
        case_id=CASE_ID,
        metrics_path=volume / "velocity_profile_metrics.json",
        npz_path=volume / "velocity_profiles.npz",
    )
    authority_evidence = {
        "authority_case_identity_sha256": "6" * 64,
        "cp_stencil_identity_sha256": "7" * 64,
        "velocity_stencil_identity_sha256": "8" * 64,
        "validity_identity_sha256": "9" * 64,
    }
    authority_hashes = {
        "cp_stencil_record": "1" * 64,
        "cp_stencil_payload": "2" * 64,
        "velocity_stencil_record": "3" * 64,
        "velocity_stencil_payload": "4" * 64,
    }
    authority = {
        "cases": {
            CASE_ID: {
                "case_id": CASE_ID,
                "prediction_bearing_evaluator_outputs_used_as_source": False,
                "artifacts": {
                    name: {
                        "file": f"/{name}",
                        "sha256": digest,
                        "byte_size": 1,
                    }
                    for name, digest in authority_hashes.items()
                },
            }
        }
    }
    authority_path = tmp_path / "authority.json"
    _write_json(authority_path, authority)

    def fake_load_bound_prerequisite_authority(**kwargs):
        assert kwargs["authority_index_path"] == authority_path
        assert kwargs["truth_release"] == truth_handle
        return authority, "5" * 64

    def fake_load_compact_profile_support_inputs(**kwargs):
        assert kwargs == {"case_id": CASE_ID, "authority": authority}
        return cp_native, velocity_native, authority_evidence

    def fake_authority_truth_binding(**kwargs):
        assert kwargs["case_id"] == CASE_ID
        assert kwargs["truth_record"]["case_id"] == CASE_ID
        assert kwargs["authority_evidence"] == authority_evidence

    monkeypatch.setattr(
        materializer,
        "_load_bound_prerequisite_authority",
        fake_load_bound_prerequisite_authority,
    )
    monkeypatch.setattr(
        materializer,
        "load_compact_profile_support_inputs",
        fake_load_compact_profile_support_inputs,
    )
    monkeypatch.setattr(
        materializer,
        "_authority_truth_binding",
        fake_authority_truth_binding,
    )
    authority_release = tmp_path / "authority-support"
    assert (
        materializer.main(
            [
                "--submission-spec",
                str(spec),
                "--split",
                str(split),
                "--prerequisite-authority-index",
                str(authority_path),
                "--source-truth-release",
                str(truth_handle.release_root),
                "--output-root",
                str(authority_release),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.err == f"compact-support 1/1 {CASE_ID}\n"
    receipt = json.loads(captured.out)
    assert receipt["case_count"] == receipt["case_set_count"] == 1
    assert receipt["points_per_physical_graph"] == 128
    assert receipt["support_source"] == {
        "kind": "frozen_prerequisite_authority",
        "prediction_bearing_evaluator_outputs_used_as_source": False,
        "authority_index": str(authority_path.resolve()),
        "authority_index_sha256": "5" * 64,
        "legacy_source_hash_slot_mapping": materializer.AUTHORITY_SOURCE_HASH_SLOTS,
    }
    authority_handle = open_compact_support_release(
        release_root=authority_release,
        expected_manifest_sha256=receipt[
            "evaluator_support_manifest_sha256"
        ],
        expected_case_ids=[CASE_ID],
        case_set_id=CASE_SET_ID,
    )
    native_record = native_handle.case_records[CASE_ID]
    authority_record = authority_handle.case_records[CASE_ID]
    assert authority_record["support"] == native_record["support"]
    assert authority_record["surface_cp"] == native_record["surface_cp"]
    assert authority_record["volume_velocity"] == native_record["volume_velocity"]
    assert authority_record["source_artifact_sha256"] == {
        compatibility_slot: authority_hashes[authority_name]
        for compatibility_slot, authority_name in (
            materializer.AUTHORITY_SOURCE_HASH_SLOTS.items()
        )
    }
    mixed_release = tmp_path / "mixed-source-support"
    assert (
        materializer.main(
            [
                "--submission-spec",
                str(spec),
                "--split",
                str(split),
                "--surface-outputs-root",
                str(outputs),
                "--volume-outputs-root",
                str(outputs),
                "--prerequisite-authority-index",
                str(authority_path),
                "--source-truth-release",
                str(truth_handle.release_root),
                "--output-root",
                str(mixed_release),
            ]
        )
        == 2
    )
    assert not mixed_release.exists()


def test_routed_materializer_deduplicates_cases_and_preserves_case_sets(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    outputs = tmp_path / "outputs"
    make_case(outputs)
    truth_handle, truth_manifest_sha = _truth_release(
        outputs, tmp_path / "truth-campaign"
    )
    spec, first_split = _benchmark_specification(
        tmp_path / "benchmark-specs/hiliftaeroml", truth_manifest_sha
    )
    spec_body = json.loads(spec.read_text(encoding="utf-8"))
    prior_manifest_sha = "7" * 64
    spec_body["compact_profile_definition"][
        "candidate_dry_run_evaluator_support"
    ]["manifest_sha256"] = prior_manifest_sha
    mirror_case_set_id = "caseset-test-mirror"
    mirror = {
        "schema_version": "1.0",
        "dataset_id": "hiliftaeroml",
        "split_id": "mirror",
        "case_set_id": mirror_case_set_id,
        "split_label": "Mirror fixture",
        "case_id_status": "official",
        "case_count": 1,
        "case_set_sha256": hashlib.sha256(
            f"{CASE_ID}\n".encode("utf-8")
        ).hexdigest(),
        "case_set_sha256_rule": materializer.CASE_SET_SHA256_RULE,
        "case_ids": [CASE_ID],
    }
    mirror_path = first_split.parent / "mirror.json"
    mirror_sha = _write_json(mirror_path, mirror)
    spec_body["splits"].append(
        {
            "id": "mirror",
            "label": "Mirror fixture",
            "index_file": "splits/mirror.json",
            "case_count": 1,
            "case_set_id": mirror_case_set_id,
            "case_id_status": "official",
            "sha256": mirror_sha,
        }
    )
    _write_json(spec, spec_body)

    def fake_open_candidate_truth_release(**kwargs):
        case_set_id = kwargs["case_set_id"]
        assert case_set_id in {CASE_SET_ID, mirror_case_set_id}
        assert kwargs["expected_case_ids"] == (CASE_ID,)
        return CandidateTruthRelease(
            campaign_root=truth_handle.campaign_root,
            release_root=truth_handle.release_root,
            binding=truth_handle.binding,
            manifest=truth_handle.manifest,
            index=truth_handle.index,
            case_set_id=case_set_id,
            case_ids=(CASE_ID,),
        )

    monkeypatch.setattr(
        materializer,
        "open_candidate_truth_release",
        fake_open_candidate_truth_release,
    )
    release = tmp_path / "routed-support"
    assert (
        materializer.main(
            [
                "--submission-spec",
                str(spec),
                "--split",
                str(first_split),
                "--surface-outputs-root",
                str(outputs),
                "--volume-outputs-root",
                str(outputs),
                "--split",
                str(mirror_path),
                "--surface-outputs-root",
                str(outputs),
                "--volume-outputs-root",
                str(outputs),
                "--source-truth-release",
                str(truth_handle.release_root),
                "--output-root",
                str(release),
                "--allow-manifest-rebind-from",
                prior_manifest_sha,
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.err == f"compact-support 1/1 {CASE_ID}\n"
    receipt = json.loads(captured.out)
    assert receipt["split_count"] == 2
    assert receipt["case_set_count"] == 2
    assert receipt["case_count"] == 1
    assert [item["selected_source_case_count"] for item in receipt["routes"]] == [
        1,
        0,
    ]
    assert receipt["manifest_rebind"] == {
        "allowed": True,
        "prior_manifest_sha256": prior_manifest_sha,
        "materialized_manifest_sha256": receipt[
            "evaluator_support_manifest_sha256"
        ],
    }
    for case_set_id in (CASE_SET_ID, mirror_case_set_id):
        opened = open_compact_support_release(
            release_root=release,
            expected_manifest_sha256=receipt[
                "evaluator_support_manifest_sha256"
            ],
            expected_case_ids=[CASE_ID],
            case_set_id=case_set_id,
        )
        assert opened.case_ids == (CASE_ID,)
