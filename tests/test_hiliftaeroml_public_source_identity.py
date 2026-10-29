from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path

import jsonschema
import pytest

from scripts import pin_hiliftaeroml_public_sources as sources


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sources.canonical_json_bytes(value))


def hf_entry(case_id: str, domain: str, size: int, token: str) -> dict:
    prefix = "boundary" if domain == "surface" else "volume"
    return {
        "type": "file",
        "oid": hashlib.sha1(f"git-{token}".encode()).hexdigest(),
        "size": size,
        "lfs": {
            "oid": hashlib.sha256(f"lfs-{token}".encode()).hexdigest(),
            "size": size,
        },
        "path": f"{case_id}/{prefix}_{case_id}.vtu.tgz",
    }


@pytest.fixture
def tiny_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(sources, "EXPECTED_CASE_COUNT", 2)
    monkeypatch.setattr(sources, "EXPECTED_RELEASE_CASE_COUNT", 2)
    monkeypatch.setattr(sources, "EXPECTED_SPLITS", {"full"})
    case_ids = ["geo_LHC001_AoA_4", "geo_LHC002_AoA_6"]
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    write_json(
        split_dir / "full.json",
        {
            "split_id": "full",
            "case_count": 2,
            "case_set_sha256": sources.ordered_case_digest(case_ids),
            "case_ids": case_ids,
        },
    )
    support_dir = tmp_path / "support"
    support_cases = []
    volume_cases = []
    hf_entries = []
    member_payloads: dict[tuple[str, str], bytes] = {}
    for number, case_id in enumerate(case_ids, 1):
        surface = f"surface-{number}".encode()
        volume = f"volume-{number}-longer".encode()
        member_payloads[(case_id, "surface")] = surface
        member_payloads[(case_id, "volume")] = volume
        support_cases.append(
            {
                "case_id": case_id,
                "support_instances": [
                    {
                        "support_id": "surface-native-points-v1",
                        "parameters": {
                            "source_identity": {
                                "filename": f"boundary_{case_id}.vtu",
                                "size_bytes": len(surface),
                            }
                        },
                    },
                    {
                        "support_id": "volume-native-valid-points-v1",
                        "parameters": {
                            "source_identity": {
                                "filename": f"volume_{case_id}.vtu",
                                "size_bytes": len(volume),
                            }
                        },
                    },
                ],
            }
        )
        surface_entry = hf_entry(case_id, "surface", 1000 + number, f"s-{number}")
        volume_entry = hf_entry(case_id, "volume", 2000 + number, f"v-{number}")
        hf_entries.extend([surface_entry, volume_entry])
        volume_cases.append(
            {
                "case_id": case_id,
                "source": {
                    "volume_archive_path": volume_entry["path"],
                    "volume_archive_size_bytes": volume_entry["size"],
                    "volume_archive_sha256": volume_entry["lfs"]["oid"],
                    "extracted_volume_vtu_filename": f"volume_{case_id}.vtu",
                    "extracted_volume_vtu_size_bytes": len(volume),
                },
            }
        )
    write_json(
        support_dir / "case-sets" / "caseset-tiny" / "chunk-000.json",
        {"cases": support_cases},
    )
    volume_manifest = tmp_path / "volume-manifest.json"
    write_json(
        volume_manifest,
        {
            "case_count": 2,
            "source_dataset": {
                "repository_id": sources.REPOSITORY_ID,
                "revision": sources.REVISION,
            },
            "cases": volume_cases,
        },
    )
    snapshot = sources.build_hf_snapshot(hf_entries)
    snapshot_path = tmp_path / "hf-snapshot.json"
    write_json(snapshot_path, snapshot)
    inventory = sources.build_inventory(
        split_dir=split_dir,
        support_release_dir=support_dir,
        volume_manifest_path=volume_manifest,
        hf_snapshot_path=snapshot_path,
    )
    inventory_path = tmp_path / "inventory.json"
    write_json(inventory_path, inventory)
    return {
        "case_ids": case_ids,
        "payloads": member_payloads,
        "snapshot": snapshot,
        "snapshot_path": snapshot_path,
        "inventory": inventory,
        "inventory_path": inventory_path,
        "tmp_path": tmp_path,
    }


def test_build_is_deterministic_and_binds_snapshot(tiny_inputs: dict) -> None:
    inventory = tiny_inputs["inventory"]
    snapshot_payload = tiny_inputs["snapshot_path"].read_bytes()
    sources.validate_inventory(inventory, hf_snapshot_payload=snapshot_payload)
    assert inventory["status"] == "complete_archive_member_binding"
    assert inventory["case_set"]["case_count"] == 2
    assert inventory["aggregate"]["optional_extracted_content_sha256_count"] == 0
    assert inventory["binding_contract"]["authoritative_tier"] == (
        "verified_archive_exact_member-v1"
    )
    assert sources.canonical_json_bytes(inventory) == tiny_inputs[
        "inventory_path"
    ].read_bytes()


def test_inventory_schema_accepts_real_shape(tiny_inputs: dict) -> None:
    # The production schema fixes 1,355 items, so validate definitions and a
    # production checked-in inventory separately when it is generated.  Here
    # schema validation exercises one case by temporarily checking its $defs.
    schema = json.loads(
        (ROOT / "schemas/hiliftaeroml/public-source-identity-v1.schema.json").read_text()
    )
    resolver = jsonschema.validators.validator_for(schema)(schema)
    resolver.check_schema(schema)
    case_schema = {"$ref": "#/$defs/case", "$defs": schema["$defs"]}
    jsonschema.validate(tiny_inputs["inventory"]["cases"][0], case_schema)


def test_public_archive_swap_and_tamper_fail_closed(tiny_inputs: dict) -> None:
    inventory = copy.deepcopy(tiny_inputs["inventory"])
    left = inventory["cases"][0]["surface"]["archive"]
    right = inventory["cases"][1]["surface"]["archive"]
    left["lfs_sha256"], right["lfs_sha256"] = (
        right["lfs_sha256"],
        left["lfs_sha256"],
    )
    with pytest.raises(sources.SourceIdentityError, match="swap/tamper"):
        sources.validate_inventory(
            inventory, hf_snapshot_payload=tiny_inputs["snapshot_path"].read_bytes()
        )


def test_member_path_ambiguity_fails_closed(tiny_inputs: dict) -> None:
    inventory = copy.deepcopy(tiny_inputs["inventory"])
    inventory["cases"][0]["surface"]["member"]["path"] = "../boundary.vtu"
    with pytest.raises(sources.SourceIdentityError, match="path ambiguity"):
        sources.validate_inventory(inventory)


def test_preexisting_extracted_hash_is_optional_local_evidence(tiny_inputs: dict) -> None:
    dataset = tiny_inputs["tmp_path"] / "dataset"
    for case_id in tiny_inputs["case_ids"]:
        case_dir = dataset / case_id
        case_dir.mkdir(parents=True)
        (case_dir / f"boundary_{case_id}.vtu").write_bytes(
            tiny_inputs["payloads"][(case_id, "surface")]
        )
        (case_dir / f"volume_{case_id}.vtu").write_bytes(
            tiny_inputs["payloads"][(case_id, "volume")]
        )
    inventory_sha = hashlib.sha256(tiny_inputs["inventory_path"].read_bytes()).hexdigest()
    receipt = sources.hash_preexisting_case(
        inventory_path=tiny_inputs["inventory_path"],
        expected_inventory_sha256=inventory_sha,
        dataset_root=dataset,
        case_index=0,
    )
    assert receipt["schema_id"] == sources.LOCAL_RECEIPT_SCHEMA
    assert receipt["domains"]["surface"]["equivalence_to_verified_public_archive"] == (
        "not_established"
    )
    assert receipt["domains"]["surface"]["content_sha256"] == hashlib.sha256(
        tiny_inputs["payloads"][(tiny_inputs["case_ids"][0], "surface")]
    ).hexdigest()


def test_volume_stencil_receipt_chain_supplies_existing_full_hashes(
    tiny_inputs: dict,
) -> None:
    root = tiny_inputs["tmp_path"] / "stencils"
    receipts = root / "receipts"
    receipts.mkdir(parents=True)
    expected_sizes: dict[str, int] = {}
    expected_hashes: dict[str, str] = {}
    for index, case_id in enumerate(tiny_inputs["case_ids"]):
        source_payload = tiny_inputs["payloads"][(case_id, "volume")]
        source_sha = hashlib.sha256(source_payload).hexdigest()
        expected_hashes[case_id] = source_sha
        expected_sizes[case_id] = len(source_payload)
        payload = root / case_id / f"native_velocity_profile_stencil_{case_id}.npz"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(f"npz-{case_id}".encode())
        payload_sha = hashlib.sha256(payload.read_bytes()).hexdigest()
        record = root / case_id / f"native_velocity_profile_stencil_{case_id}.json"
        write_json(
            record,
            {
                "schema_id": "hilift_native_volume_velocity_profile_stencil_v1",
                "case_id": case_id,
                "status": "complete",
                "source_vtu_sha256": source_sha,
                "source_vtu_size_bytes": len(source_payload),
                "npz_sha256": payload_sha,
            },
        )
        audit = root / "audits" / f"{case_id}.json"
        write_json(audit, {"case_id": case_id, "status": "pass"})

        def descriptor(path: Path) -> dict:
            data = path.read_bytes()
            return {
                "path": str(path),
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }

        write_json(
            receipts / f"{case_id}.json",
            {
                "schema_id": "hilift_native_prerequisite_case_receipt_v1",
                "case_id": case_id,
                "domain": "volume",
                "family": "volume_velocity_profile_stencils",
                "origin": "generated" if index == 0 else "reused_audited_aoa4",
                "status": "complete",
                "inventory": {
                    "sha256": "1" * 64,
                    "size_bytes": 100,
                },
                "table5_campaign": {
                    "sha256": "2" * 64,
                    "size_bytes": 200,
                },
                "validation": {
                    "files": {
                        "record": descriptor(record),
                        "audit": descriptor(audit),
                        "payload": descriptor(payload),
                    },
                    "summary": {
                        "case_id": case_id,
                        "status": "complete",
                        "audit_status": "pass",
                        "source": {
                            "relative_path": f"{case_id}/volume_{case_id}.vtu",
                            "size_bytes": len(source_payload),
                            "sha256": source_sha,
                            "full_sha256_reverified_by_finalizer": False,
                        },
                    },
                },
            },
        )
    attestations, evidence = sources.volume_stencil_attestations(
        receipt_dir=receipts,
        case_ids=tiny_inputs["case_ids"],
        expected_volume_sizes=expected_sizes,
    )
    assert {case_id: item["content_sha256"] for case_id, item in attestations.items()} == (
        expected_hashes
    )
    assert evidence["case_count"] == 2
    assert evidence["unique_source_content_sha256_count"] == 2
    assert evidence["origins"] == {"generated": 1, "reused_audited_aoa4": 1}
    assert evidence["source_full_sha256_reverified_by_finalizer_count"] == 0


def write_tgz(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))


def archive_case_record(path: Path, member_name: str, payload: bytes) -> dict:
    return {
        "case_id": "geo_LHC001_AoA_4",
        "surface": {
            "archive": {
                "repository_path": (
                    "geo_LHC001_AoA_4/boundary_geo_LHC001_AoA_4.vtu.tgz"
                ),
                "size_bytes": path.stat().st_size,
                "lfs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            "member": {
                "path": member_name,
                "declared_size_bytes": len(payload),
            },
        },
    }


def test_verified_archive_stream_establishes_member_equivalence(tmp_path: Path) -> None:
    payload = b"exact public member bytes"
    name = "boundary_geo_LHC001_AoA_4.vtu"
    archive = tmp_path / "case.tgz"
    write_tgz(archive, [(name, payload)])
    receipt = sources.audit_local_archive(
        archive_path=archive,
        case_record=archive_case_record(archive, name, payload),
        domain="surface",
    )
    assert receipt["public_equivalence"] == (
        "established_by_verified_archive_stream_extraction"
    )
    assert receipt["member"]["content_sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "members,error",
    [
        (
            [("../boundary_geo_LHC001_AoA_4.vtu", b"x")],
            "ambiguous/nonconforming",
        ),
        (
            [
                ("boundary_geo_LHC001_AoA_4.vtu", b"x"),
                ("extra.vtu", b"y"),
            ],
            "ambiguous/nonconforming",
        ),
    ],
)
def test_archive_member_path_or_multiplicity_ambiguity_is_rejected(
    tmp_path: Path, members: list[tuple[str, bytes]], error: str
) -> None:
    expected_name = "boundary_geo_LHC001_AoA_4.vtu"
    archive = tmp_path / "ambiguous.tgz"
    write_tgz(archive, members)
    record = archive_case_record(archive, expected_name, b"x")
    with pytest.raises(sources.SourceIdentityError, match=error):
        sources.audit_local_archive(
            archive_path=archive,
            case_record=record,
            domain="surface",
        )


def test_local_archive_byte_tamper_is_rejected(tmp_path: Path) -> None:
    payload = b"exact public member bytes"
    name = "boundary_geo_LHC001_AoA_4.vtu"
    archive = tmp_path / "tampered.tgz"
    write_tgz(archive, [(name, payload)])
    record = archive_case_record(archive, name, payload)
    data = bytearray(archive.read_bytes())
    data[-1] ^= 1
    archive.write_bytes(data)
    with pytest.raises(sources.SourceIdentityError, match="LFS identity differs"):
        sources.audit_local_archive(
            archive_path=archive,
            case_record=record,
            domain="surface",
        )
