#!/usr/bin/env python3
"""Validate FluidsBench submission metadata and profile chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.scores import (
    composite_component_group_scores,
    composite_overall_score,
    legacy_aero_scores,
)
from reference.weightings import evaluator_weighting
from reference.drivaerml.dataset_scorer import (
    DrivAerDatasetScorerError,
    RELATIVE_PROFILE_CONTRACT_ID,
    RELATIVE_PROFILE_CONTRACT_SHA256,
    RELATIVE_PROFILE_FORMAT,
    validate_schema_v3_candidate_nonspatial_metrics,
    validate_schema_v3_relative_profile_chunk_candidate,
)
from reference.drivaerml.regional_aggregate import (
    AGGREGATE_REGIONAL_REPORT_SCHEMA,
    REGIONAL_DIAGNOSTICS_CONTRACT_SHA256,
    RegionalAggregateError,
    validate_aggregate_regional_diagnostics,
)
from reference.hiliftaeroml.regional_aggregate import (
    AGGREGATE_REGIONAL_REPORT_SCHEMA as HILIFT_AGGREGATE_REGIONAL_REPORT_SCHEMA,
    REGIONAL_DEFINITION_ID as HILIFT_REGIONAL_DEFINITION_ID,
    REGIONAL_DIAGNOSTICS_CONTRACT_SHA256 as HILIFT_REGIONAL_DIAGNOSTICS_CONTRACT_SHA256,
    HiLiftRegionalAggregateError,
    validate_aggregate_regional_diagnostics as validate_hilift_aggregate_regional_diagnostics,
)
from reference.hiliftaeroml.native_profiles import (
    CP_PUBLISHED_ARRAYS as HILIFT_CP_PROFILE_ARRAYS,
    NativeProfileError as HiLiftNativeProfileError,
    PROFILE_CONTRACT_ID as HILIFT_PROFILE_CONTRACT_ID,
    PROFILE_CONTRACT_SHA256 as HILIFT_PROFILE_CONTRACT_SHA256,
    PROFILE_FORMAT as HILIFT_PROFILE_FORMAT,
    VELOCITY_PUBLISHED_ARRAYS as HILIFT_VELOCITY_PROFILE_ARRAYS,
    validate_prediction_npz as validate_hilift_prediction_npz,
)
from reference.hiliftaeroml.native_profile_evaluator import (
    NativeProfileEvaluationError as HiLiftNativeProfileEvaluationError,
    score_native_profile_directory as score_hilift_native_profile_directory,
)
from reference.hiliftaeroml.compact_profile_evaluator import (
    COMPACT_PROFILE_CONTRACT_ID as HILIFT_COMPACT_PROFILE_CONTRACT_ID,
    COMPACT_PROFILE_CONTRACT_SHA256 as HILIFT_COMPACT_PROFILE_CONTRACT_SHA256,
    COMPACT_PROFILE_FORMAT as HILIFT_COMPACT_PROFILE_FORMAT,
    CompactSupportRelease as HiLiftCompactSupportRelease,
    CompactProfileEvaluationError as HiLiftCompactProfileEvaluationError,
    open_compact_support_release as open_hilift_compact_support_release,
    score_compact_profile_directory as score_hilift_compact_profile_directory,
)
from reference.methodology import methodology_errors

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:  # pragma: no cover - exercised by the command's dependency check
    Draft202012Validator = None
    FormatChecker = None


MANIFEST_PATH = ROOT / "leaderboard" / "manifest.json"
SCHEMA_ROOT = ROOT / "schemas"
OPEN_REPRODUCIBILITY_CONTRACTS = {
    "2.0": "open-reproducibility-2.0",
    "3.0": "open-reproducibility-3.0",
}
HILIFT_COMPACT_PROFILE_IMPLEMENTATION_BINDING = {
    "status": "unbound_worktree_candidate",
    "activation_effect": "none",
    "code_revision": None,
    "implementation_manifest_sha256": None,
    "base_dataset_evaluator_scope": (
        "native_v1_base_field_force_and_noncompact_scoring_only"
    ),
}


def _hilift_preview_configuration(
    *,
    slug: str,
    model_slug: str = "transolver",
    validation_slug: str | None = None,
    split_id: str,
    case_set_id: str,
    case_count: int,
    submission_json_sha256: str,
    archive_sha256: str,
    archive_size_bytes: int,
    archive_member_count: int,
    regular_file_bytes: int,
) -> dict[str, Any]:
    """Build one explicit, code-pinned HiLift pre-release registration."""

    submission_id = f"hiliftaeroml-{model_slug}-{slug}-candidate-v1"
    validation_name = (
        f"compact-profile-{validation_slug or slug}-validation-v1"
    )
    binding = {
        "status": "registered_pre_release_reference",
        "record_type": "pre_release_reference",
        "submission_id": submission_id,
        "submission_path": (
            f"submissions/hiliftaeroml/{submission_id}/submission.json"
        ),
        "submission_json_sha256": submission_json_sha256,
        "deterministic_archive": {
            "sha256": archive_sha256,
            "size_bytes": archive_size_bytes,
            "member_count": archive_member_count,
        },
        "ordinary_validation_scope": (
            "structure_hashes_inventory_and_deterministic_archive_only"
        ),
        "truth_dependent_metric_recomputation": (
            "maintainer_local_candidate_dry_run_only"
        ),
        "claim_eligibility": {
            "academic_citation": False,
            "promotion": False,
        },
    }
    return {
        "binding": binding,
        "validation_record_path": (
            Path("benchmark-specs")
            / "hiliftaeroml"
            / f"{validation_name}.json"
        ),
        "validation_record_schema": f"hiliftaeroml-{validation_name}",
        "split_id": split_id,
        "case_set_id": case_set_id,
        "case_count": case_count,
        "profile_series_count": case_count * 15,
        "regular_file_bytes_each": regular_file_bytes,
    }


HILIFT_REGISTERED_PREVIEW_CONFIGS = (
    _hilift_preview_configuration(
        slug="full360",
        split_id="full",
        case_set_id="caseset-ac791749e527",
        case_count=360,
        submission_json_sha256=(
            "43a4d81dfeb159be2b81ac655647420bc1a275784dd9635f2053f2ab4d8b5b2d"
        ),
        archive_sha256=(
            "e270e73e61d2399f4e954b0b13b09d17a637add7e61d8136c20e168647ebdedd"
        ),
        archive_size_bytes=8_724_067,
        archive_member_count=403,
        regular_file_bytes=11_758_641,
    ),
    _hilift_preview_configuration(
        slug="aoa4",
        split_id="single_aoa_4",
        case_set_id="caseset-7a743a20b3bd",
        case_count=36,
        submission_json_sha256=(
            "4f1fd8b465797f89bcb4dac4791d624e26f7761ca6b8ddd0632d36645c1602e0"
        ),
        archive_sha256=(
            "1b86a5aae3ec1f82080536c0214413eecafec66a452b332ca7e1b56f82aeee40"
        ),
        archive_size_bytes=842_039,
        archive_member_count=47,
        regular_file_bytes=1_258_072,
    ),
    _hilift_preview_configuration(
        slug="aoa12",
        split_id="single_aoa_12",
        case_set_id="caseset-02fc12ff3494",
        case_count=36,
        submission_json_sha256=(
            "22ee1e5e8c5b4cac46825bbca8de0c940563b1f052882503dac1ffcf0e9d6221"
        ),
        archive_sha256=(
            "bb7bf8d7d32eb9122f9d9a0f1234e4ae7033a075dc828709e3ac78b0c154f412"
        ),
        archive_size_bytes=955_829,
        archive_member_count=47,
        regular_file_bytes=1_372_495,
    ),
    _hilift_preview_configuration(
        slug="aoa22",
        split_id="single_aoa_22",
        case_set_id="caseset-85ecccd9ccda",
        case_count=36,
        submission_json_sha256=(
            "df6cb95bcd8c22dfd3647cd696a88824000fdd8a17c0dcb6c476b45c4e58d104"
        ),
        archive_sha256=(
            "56d9d893870062ff584b040a2eced5cdbac7b22f91b806a99004e5ae7fed8ce4"
        ),
        archive_size_bytes=960_142,
        archive_member_count=47,
        regular_file_bytes=1_376_137,
    ),
    _hilift_preview_configuration(
        slug="super-scarce",
        split_id="super_scarce",
        case_set_id="caseset-ac791749e527",
        case_count=360,
        submission_json_sha256=(
            "57087ae77adae29f36bfc868e8e86baa373c03e778516aa602b9466c16bce95d"
        ),
        archive_sha256=(
            "bcb6c2bda00150c039f6aef7019c99e855540abe126c4e9bb02a9f6f4e7656a1"
        ),
        archive_size_bytes=10_532_488,
        archive_member_count=403,
        regular_file_bytes=13_569_255,
    ),
    _hilift_preview_configuration(
        slug="geometry-scarce",
        split_id="geometry_scarce",
        case_set_id="caseset-53990ea68fa6",
        case_count=360,
        submission_json_sha256=(
            "5be3278468041bce62b015d45d77c5482f5d79ca061527418aba999b19b8db2b"
        ),
        archive_sha256=(
            "1313d869cd28466ad47e635ac57cc16325d1dd5839fe155ed2d8de5f001c3073"
        ),
        archive_size_bytes=9_322_198,
        archive_member_count=403,
        regular_file_bytes=12_365_210,
    ),
    _hilift_preview_configuration(
        slug="geometry-super-scarce",
        split_id="geometry_super_scarce",
        case_set_id="caseset-53990ea68fa6",
        case_count=360,
        submission_json_sha256=(
            "68a649ea1933078cf4a7082bdd0daba046aafcc149bceb0489943711ba1c767d"
        ),
        archive_sha256=(
            "903d101210b641ac3b56075f1859f0494184a322e2af1fa183bb688557904d6e"
        ),
        archive_size_bytes=10_205_122,
        archive_member_count=403,
        regular_file_bytes=13_250_900,
    ),
    _hilift_preview_configuration(
        slug="geometry",
        split_id="geometry",
        case_set_id="caseset-53990ea68fa6",
        case_count=360,
        submission_json_sha256=(
            "f58c7a9d7ea549799c3f035b5b93ec66be4ae15d4863d8788e0fda81aaea5a35"
        ),
        archive_sha256=(
            "1ad985b42ff6a7654b1f980498212e1cabbad6e54c25e1213ba8e47e546e0bf2"
        ),
        archive_size_bytes=8_517_360,
        archive_member_count=403,
        regular_file_bytes=11_555_256,
    ),
    _hilift_preview_configuration(
        slug="ood-aoa",
        split_id="aoa",
        case_set_id="caseset-29693354ed8a",
        case_count=900,
        submission_json_sha256=(
            "dc3c9ea0ff17bec23785ece90994010da887f7bb898fe14f5152e82e24b5104d"
        ),
        archive_sha256=(
            "4bd33893d4de01f266372753cbd7c6de65a87f0535dedcf055a6d487161a42ec"
        ),
        archive_size_bytes=24_720_935,
        archive_member_count=997,
        regular_file_bytes=32_105_249,
    ),
    _hilift_preview_configuration(
        slug="ood-deflection",
        split_id="deflection",
        case_set_id="caseset-c0ecb14de138",
        case_count=360,
        submission_json_sha256=(
            "c20c3ab1ede7d3a8f6d95c53402e92f840d71298c3ee3ead47e783f6bcd7bbbb"
        ),
        archive_sha256=(
            "518f1472c77d5bd11054e86f90420d2919a5f5e26ab85d38c9e9bd0c3752788e"
        ),
        archive_size_bytes=8_844_091,
        archive_member_count=403,
        regular_file_bytes=11_886_343,
    ),
    _hilift_preview_configuration(
        slug="ood-stall",
        split_id="stall",
        case_set_id="caseset-804491c8956e",
        case_count=723,
        submission_json_sha256=(
            "3d741145a35729e62c4854610e583c0e62d5543baa96f47d2ee5e3d2a3b25c59"
        ),
        archive_sha256=(
            "dc53ddac42242c828afcffd27bd02999173b6938213504d8abd3ebee641b33c5"
        ),
        archive_size_bytes=19_015_374,
        archive_member_count=803,
        regular_file_bytes=24_979_559,
    ),
    _hilift_preview_configuration(
        slug="full360",
        model_slug="geotransolver",
        validation_slug="geotransolver-full360",
        split_id="full",
        case_set_id="caseset-ac791749e527",
        case_count=360,
        submission_json_sha256=(
            "da274a509a586ce8dd00705d570e993124c564cd4185d997d47428e6d7a3169e"
        ),
        archive_sha256=(
            "97c7150f958788bb1e662d25688c5b1e7529bfb11800c92a5ec542b99c59f1d8"
        ),
        archive_size_bytes=8_964_582,
        archive_member_count=403,
        regular_file_bytes=12_002_142,
    ),
    _hilift_preview_configuration(
        slug="scarce",
        model_slug="geotransolver",
        validation_slug="geotransolver-scarce",
        split_id="scarce",
        case_set_id="caseset-ac791749e527",
        case_count=360,
        submission_json_sha256=(
            "693ad9063ad643abd93be3d306142f0a3d9cfbeec59a34f3389c681232183cf0"
        ),
        archive_sha256=(
            "72864acfe1787eab4c3f70da6b5b4d5b111c4427d5f9b78b156a749d5603af86"
        ),
        archive_size_bytes=10_458_737,
        archive_member_count=403,
        regular_file_bytes=13_495_213,
    ),
    _hilift_preview_configuration(
        slug="aoa4",
        model_slug="geotransolver",
        validation_slug="geotransolver-aoa4",
        split_id="single_aoa_4",
        case_set_id="caseset-7a743a20b3bd",
        case_count=36,
        submission_json_sha256=(
            "333e0867424115c5a3675374597d7bcd193ab7ef2aa0cf16e66e1a3333fc7996"
        ),
        archive_sha256=(
            "fa2d77e2fd77e6920b96fe0a40e684d6a5e53da5911f455b6f939a7674c8671b"
        ),
        archive_size_bytes=891_073,
        archive_member_count=47,
        regular_file_bytes=1_307_990,
    ),
    _hilift_preview_configuration(
        slug="aoa12",
        model_slug="geotransolver",
        validation_slug="geotransolver-aoa12",
        split_id="single_aoa_12",
        case_set_id="caseset-02fc12ff3494",
        case_count=36,
        submission_json_sha256=(
            "5bf17fa2f42c94aefa925bd45b84a34ef1f791aeb7867e27954b1ac62c40bc03"
        ),
        archive_sha256=(
            "87287a12053304a147a2624039365bad3cccb964df79d1105a31396b26165543"
        ),
        archive_size_bytes=968_185,
        archive_member_count=47,
        regular_file_bytes=1_385_492,
    ),
    _hilift_preview_configuration(
        slug="aoa22",
        model_slug="geotransolver",
        validation_slug="geotransolver-aoa22",
        split_id="single_aoa_22",
        case_set_id="caseset-85ecccd9ccda",
        case_count=36,
        submission_json_sha256=(
            "bfcd27cc9ca57e4dac64ce466f7b933451c00be73bc71ca1633217b95514fa52"
        ),
        archive_sha256=(
            "146f28436326fab089afda212b2798033839c2a8c89d5c2a2be37eaf6dd10e8f"
        ),
        archive_size_bytes=1_027_593,
        archive_member_count=47,
        regular_file_bytes=1_444_217,
    ),
    _hilift_preview_configuration(
        slug="geometry",
        model_slug="geotransolver",
        validation_slug="geotransolver-geometry",
        split_id="geometry",
        case_set_id="caseset-53990ea68fa6",
        case_count=360,
        submission_json_sha256=(
            "6c174b7117696c22031e42d60cb787ff7d09d294c42a08c29bc28858730917e7"
        ),
        archive_sha256=(
            "a2e4fd58b889e377be1fb8962b53647fbd0c6429042a9e02c612260a12abca45"
        ),
        archive_size_bytes=8_653_787,
        archive_member_count=403,
        regular_file_bytes=11_694_031,
    ),
    _hilift_preview_configuration(
        slug="geometry-scarce",
        model_slug="geotransolver",
        validation_slug="geotransolver-geometry-scarce",
        split_id="geometry_scarce",
        case_set_id="caseset-53990ea68fa6",
        case_count=360,
        submission_json_sha256=(
            "4755cf5e41f8582444bd27fe2b672c8abdc4730425a5837325ba15ef7b54b8a1"
        ),
        archive_sha256=(
            "4d70db3198e902870a6b95ea45505552e64f3dae692b8ee2f1ae331920d12651"
        ),
        archive_size_bytes=10_114_473,
        archive_member_count=403,
        regular_file_bytes=13_158_992,
    ),
    _hilift_preview_configuration(
        slug="geometry-super-scarce",
        model_slug="geotransolver",
        validation_slug="geotransolver-geometry-super-scarce",
        split_id="geometry_super_scarce",
        case_set_id="caseset-53990ea68fa6",
        case_count=360,
        submission_json_sha256=(
            "dbbe2aab6a43e02e063dff10565f7a1ab059175994074315eb740be2d3f979f7"
        ),
        archive_sha256=(
            "3799fd4bfa747bf35451f84f1562639191aa3e479865e188f19f7a7f1847c36d"
        ),
        archive_size_bytes=11_092_082,
        archive_member_count=403,
        regular_file_bytes=14_137_677,
    ),
    _hilift_preview_configuration(
        slug="super-scarce",
        model_slug="geotransolver",
        validation_slug="geotransolver-super-scarce",
        split_id="super_scarce",
        case_set_id="caseset-ac791749e527",
        case_count=360,
        submission_json_sha256=(
            "e23adaef21962b0a7e16d3de0fe3e36b0975fff85cc441138c1efd8e4acc9983"
        ),
        archive_sha256=(
            "30efc9a3468459862477a4734cf330ac1942b594740346dad0f2d214fb36c10b"
        ),
        archive_size_bytes=10_934_679,
        archive_member_count=403,
        regular_file_bytes=13_971_416,
    ),
    _hilift_preview_configuration(
        slug="ood-aoa",
        model_slug="geotransolver",
        validation_slug="geotransolver-ood-aoa",
        split_id="aoa",
        case_set_id="caseset-29693354ed8a",
        case_count=900,
        submission_json_sha256=(
            "27fa07b886ee3168016cd5ed1356da90bded93364ba174c92887b5ec709073fd"
        ),
        archive_sha256=(
            "ea2e455de205202bd6cf89139f7e5659e0f5f7ce403e847ea4e19a654d32dc3c"
        ),
        archive_size_bytes=26_779_386,
        archive_member_count=997,
        regular_file_bytes=34_170_217,
    ),
    _hilift_preview_configuration(
        slug="ood-deflection",
        model_slug="geotransolver",
        validation_slug="geotransolver-ood-deflection",
        split_id="deflection",
        case_set_id="caseset-c0ecb14de138",
        case_count=360,
        submission_json_sha256=(
            "939072a9c57c3a58563f4a37a41fab96f50ac7d1264bd7a1af7c25aa32161a9c"
        ),
        archive_sha256=(
            "26674828ec6bab1b30a8d34f0491879806eddf79595eb45bbd1630017f7b78bc"
        ),
        archive_size_bytes=9_056_573,
        archive_member_count=403,
        regular_file_bytes=12_101_235,
    ),
    _hilift_preview_configuration(
        slug="ood-stall",
        model_slug="geotransolver",
        validation_slug="geotransolver-ood-stall",
        split_id="stall",
        case_set_id="caseset-804491c8956e",
        case_count=723,
        submission_json_sha256=(
            "bfedac34341e08c7b23ef7206c4c87941defcefc6282c70bc36df6da02d1f2bc"
        ),
        archive_sha256=(
            "bfb8478a52c74227ee89997aad83032669386fb0dd7a78f864db26555aa4847a"
        ),
        archive_size_bytes=19_833_594,
        archive_member_count=803,
        regular_file_bytes=25_800_404,
    ),
)
HILIFT_REGISTERED_PREVIEW = HILIFT_REGISTERED_PREVIEW_CONFIGS[0]["binding"]
HILIFT_REGISTERED_PREVIEW_RECORD_PATH = HILIFT_REGISTERED_PREVIEW_CONFIGS[0][
    "validation_record_path"
]
LEGACY_V1_SUBMISSION_ID = re.compile(r"^(?P<series>[a-z0-9][a-z0-9-]{2,69})-v1$")

DRIVAERML_RELATIVE_ACTIVATION_RECORD_SCHEMA = (
    "drivaerml-relative-diagnostics-activation-release-v1"
)
DRIVAERML_RELATIVE_SUPPORT_INDEX_SCHEMA = (
    "drivaerml-relative-series-support-index-v2"
)
DRIVAERML_CONSTANT_SUPPORT_INDEX_SCHEMA = (
    "drivaerml-constant-series-support-index-v1"
)
DRIVAERML_OFFICIAL_CASE_REGISTRY_SCHEMA = (
    "drivaerml-fluidsbench-public-native-source-pin-v1"
)
DRIVAERML_RELATIVE_ACTIVATION_GATE_IDS = (
    "all_484_velocity_placement_manifest_bound",
    "all_484_velocity_mapping_manifest_bound",
    "all_484_cp_manifest_bound",
    "genuine_model_sensitivity_review_complete",
    "owner_scientific_approval",
    "immutable_evaluator_revision_bound",
)
DRIVAERML_RELATIVE_MANIFEST_SCHEMAS = {
    "velocity_placement_manifest": (
        "drivaerml-relative-velocity-v3-production-input-manifest-v1"
    ),
    "velocity_mapping_manifest": (
        "drivaerml-velocity-relative-v3-mapping-aggregate-v1"
    ),
    "cp_manifest": "drivaerml-relative-cp-native-support-manifest-v3",
}
DRIVAERML_RELATIVE_MANIFEST_FAMILIES = {
    "velocity_placement_manifest": "drivaerml-velocity-relative-v3",
    "velocity_mapping_manifest": "drivaerml-velocity-relative-v3",
    "cp_manifest": "drivaerml_cp_relative_v1",
}
DRIVAERML_RELATIVE_SENSITIVITY_SCHEMA = (
    "drivaerml-relative-diagnostics-sensitivity-evidence-v1"
)
DRIVAERML_TRAINED_CHECKPOINT_PROVENANCE_SCHEMA = (
    "drivaerml-trained-checkpoint-provenance-v1"
)
DRIVAERML_SENSITIVITY_PREDICTION_MANIFEST_SCHEMA = (
    "drivaerml-sensitivity-prediction-manifest-v1"
)
DRIVAERML_RELATIVE_OWNER_APPROVAL_SCHEMA = (
    "drivaerml-relative-diagnostics-owner-approval-v1"
)
DRIVAERML_EVALUATOR_REPOSITORY = (
    "https://github.com/neilashton/fluidsbench-submission"
)
DRIVAERML_RELATIVE_OWNER_APPROVERS = frozenset({"neilashton"})
DRIVAERML_RELATIVE_APPROVAL_TRUSTED_REF = "refs/remotes/origin/dev"
DRIVAERML_RELATIVE_PROFILE_CHUNK_SCHEMA_SHA256 = (
    "ff5c5965bb00633303b9372360879d02535f7946c882b9cfcefe1ee55446a0d2"
)
DRIVAERML_OFFICIAL_CASE_COUNT = 484
LOWER_SHA256 = re.compile(r"^[a-f0-9]{64}$")
LOWER_GIT_SHA1 = re.compile(r"^[a-f0-9]{40}$")

HILIFT_FORCE_SUPPORT_ID = "aerodynamic-case-coefficients-v1"
HILIFT_FORCE_COEFFICIENT_FIELDS = (
    "predicted_c_drag",
    "truth_c_drag",
    "predicted_c_lift",
    "truth_c_lift",
    "predicted_c_pitch",
    "truth_c_pitch",
)
HILIFT_FORCE_MAE_INPUTS = {
    "c_drag_mae": ("truth_c_drag", "predicted_c_drag"),
    "c_lift_mae": ("truth_c_lift", "predicted_c_lift"),
    "c_pitch_mae": ("truth_c_pitch", "predicted_c_pitch"),
}
HILIFT_FORCE_R2_INPUTS = {
    "cd_r2": ("truth_c_drag", "predicted_c_drag"),
    "cl_r2": ("truth_c_lift", "predicted_c_lift"),
}


class SubmissionJSONError(ValueError):
    """Raised for JSON constructs forbidden in authoritative submission metadata."""


def _reject_submission_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SubmissionJSONError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_submission_nonfinite(token: str) -> Any:
    raise SubmissionJSONError(f"forbidden non-finite JSON token {token}")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_submission_json(path: Path) -> Any:
    """Load top-level submission metadata without duplicate or non-finite values."""

    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=_reject_submission_duplicate_keys,
            parse_constant=_reject_submission_nonfinite,
        )


def normalized_result_revision(submission: dict[str, Any]) -> dict[str, Any]:
    """Return public revision metadata, including a safe legacy-v1 fallback."""

    declared = submission.get("result_revision")
    if isinstance(declared, dict):
        return {
            "series_id": declared.get("series_id"),
            "version": declared.get("version"),
            "supersedes": declared.get("supersedes"),
            "change_summary": declared.get("change_summary"),
        }

    submission_id = submission.get("submission_id")
    match = LEGACY_V1_SUBMISSION_ID.fullmatch(submission_id or "")
    if match:
        series_id = match.group("series")
        return {
            "series_id": series_id,
            "version": 1,
            "supersedes": None,
            "change_summary": None,
        }
    return {
        "series_id": submission_id,
        "version": 1,
        "supersedes": None,
        "change_summary": None,
    }


def validate_result_revisions(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    focus_paths: set[Path] | None = None,
) -> list[str]:
    """Validate immutable, sequential result-series relationships across packages."""

    focus = {path.resolve() for path in focus_paths} if focus_paths is not None else None
    errors: list[str] = []
    by_submission_id = {
        submission.get("submission_id"): (path, submission)
        for path, submission in records
        if isinstance(submission.get("submission_id"), str)
    }
    by_revision: dict[tuple[Any, Any, Any, Any], tuple[Path, dict[str, Any]]] = {}

    def selected(path: Path) -> bool:
        return focus is None or path.resolve() in focus

    def add(path: Path, message: str) -> None:
        if selected(path):
            prefix = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            errors.append(f"{prefix}: {message}")

    for path, submission in records:
        revision = normalized_result_revision(submission)
        key = (
            submission.get("dataset_id"),
            submission.get("split_id"),
            revision.get("series_id"),
            revision.get("version"),
        )
        previous = by_revision.get(key)
        if previous is not None:
            previous_path, previous_submission = previous
            if isinstance(submission.get("result_revision"), dict) or isinstance(
                previous_submission.get("result_revision"), dict
            ):
                add(
                    path,
                    "result_revision duplicates the dataset/split/series/version used by "
                    f"{previous_path.relative_to(ROOT) if previous_path.is_relative_to(ROOT) else previous_path}",
                )
                if selected(previous_path):
                    add(previous_path, "result_revision is duplicated by another submission package")
        else:
            by_revision[key] = (path, submission)

        declared = submission.get("result_revision")
        if not isinstance(declared, dict):
            continue
        series_id = declared.get("series_id")
        version = declared.get("version")
        supersedes = declared.get("supersedes")
        if not isinstance(series_id, str) or not isinstance(version, int) or isinstance(version, bool):
            continue  # JSON Schema reports structural failures.
        expected_submission_id = f"{series_id}-v{version}"
        if submission.get("submission_id") != expected_submission_id:
            add(
                path,
                "submission_id must equal result_revision.series_id followed by "
                f"'-v{version}' ({expected_submission_id!r})",
            )
        if version == 1:
            if supersedes is not None:
                add(path, "result_revision version 1 must set supersedes to null")
            continue

        expected_predecessor_id = f"{series_id}-v{version - 1}"
        allowed_predecessor_ids = [expected_predecessor_id]
        legacy_predecessor = by_submission_id.get(series_id) if version == 2 else None
        if (
            legacy_predecessor is not None
            and not isinstance(legacy_predecessor[1].get("result_revision"), dict)
            and normalized_result_revision(legacy_predecessor[1]).get("version") == 1
        ):
            allowed_predecessor_ids.append(series_id)
        if supersedes not in allowed_predecessor_ids:
            expected = " or ".join(repr(identifier) for identifier in allowed_predecessor_ids)
            add(
                path,
                f"result_revision version {version} must supersede {expected}",
            )
            continue
        predecessor_record = by_submission_id.get(supersedes)
        if predecessor_record is None:
            add(path, f"result_revision predecessor {supersedes!r} does not exist")
            continue
        _, predecessor = predecessor_record
        predecessor_revision = normalized_result_revision(predecessor)
        if (
            predecessor_revision.get("series_id") != series_id
            or predecessor_revision.get("version") != version - 1
        ):
            add(path, "result_revision predecessor does not belong to the immediately preceding series version")
        for field in (
            "dataset_id",
            "dataset_version",
            "split_id",
            "split_sha256",
            "case_set_id",
            "submitter_name",
            "institution",
        ):
            if predecessor.get(field) != submission.get(field):
                add(path, f"result_revision predecessor must have the same {field}")
        predecessor_status = predecessor.get("approval", {}).get("status")
        if predecessor_status not in {"approved", "prototype"}:
            add(path, "result_revision predecessor must already be published as approved or prototype")
        try:
            submitted_at = date.fromisoformat(str(submission.get("submitted_at")))
            predecessor_date = date.fromisoformat(str(predecessor.get("submitted_at")))
        except ValueError:
            pass  # JSON Schema reports invalid dates.
        else:
            if submitted_at < predecessor_date:
                add(path, "submitted_at must not be earlier than the superseded result")

    return errors


def manifest_with_benchmark_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Overlay benchmark-owned contract fields onto generated manifest metadata.

    The compact leaderboard manifest is generated output, while each dataset's
    submission specification is the source of truth for accepted metric IDs,
    ranking, metric equations, and scoring-support status.  Applying this small
    overlay before validation prevents a changed specification from being
    rejected solely because the generated manifest has not yet been rebuilt.

    Presentation-only metric metadata (labels, descriptions, groups, and display
    precision) remains owned by the leaderboard manifest.  A newly introduced
    metric therefore still needs a corresponding presentation definition there.
    """

    updated = deepcopy(manifest)
    definitions = {
        definition.get("id"): definition
        for definition in updated.get("metric_definitions", [])
        if isinstance(definition, dict) and isinstance(definition.get("id"), str)
    }
    for dataset in updated.get("datasets", []):
        slug = dataset.get("slug")
        if not isinstance(slug, str):
            continue
        spec_path = ROOT / "benchmark-specs" / slug / "submission-spec.json"
        if not spec_path.is_file():
            continue
        specification = load_json(spec_path)
        metrics = [
            metric
            for metric in specification.get("metrics", [])
            if isinstance(metric, dict) and isinstance(metric.get("id"), str)
        ]
        dataset["metric_ids"] = [metric["id"] for metric in metrics]
        if isinstance(specification.get("ranking"), dict):
            dataset["ranking"] = deepcopy(specification["ranking"])
        composite = specification.get("overall_score_composite")
        if isinstance(composite, dict):
            dataset["overall_score_composite"] = deepcopy(composite)
        else:
            dataset.pop("overall_score_composite", None)
        component_groups = specification.get("component_score_groups")
        if isinstance(component_groups, dict):
            dataset["component_score_groups"] = deepcopy(component_groups)
        else:
            dataset.pop("component_score_groups", None)
        profile_definition = specification.get("profile_definition")
        if isinstance(profile_definition, dict):
            dataset["profile_definition"] = deepcopy(profile_definition)
        else:
            dataset.pop("profile_definition", None)
        scoring_support = specification.get("scoring_support")
        if isinstance(scoring_support, dict):
            dataset["scoring_support"] = {
                key: deepcopy(value)
                for key, value in scoring_support.items()
                if key != "manifest_file"
            }
        for metric in metrics:
            definition = definitions.get(metric["id"])
            if definition is None:
                continue
            for key in ("unit", "direction", "kind", "equation"):
                if key in metric:
                    definition[key] = deepcopy(metric[key])
    return updated


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_registered_hiliftaeroml_preview_path(
    path: Path,
    *,
    root: Path | None = None,
) -> bool:
    """Return whether *path* is a maintainer-configured preview path."""

    return _registered_hiliftaeroml_preview_config(path, root=root) is not None


def _registered_hiliftaeroml_preview_config(
    path: Path,
    *,
    root: Path | None = None,
) -> dict[str, Any] | None:
    """Resolve one code-pinned preview configuration by repository path."""

    effective_root = (root or ROOT).resolve()
    try:
        relative_path = path.resolve().relative_to(effective_root).as_posix()
    except (OSError, ValueError):
        return None
    return next(
        (
            configuration
            for configuration in HILIFT_REGISTERED_PREVIEW_CONFIGS
            if relative_path == configuration["binding"]["submission_path"]
        ),
        None,
    )


def registered_hiliftaeroml_preview(
    path: Path,
    submission: dict[str, Any],
    manifest: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any] | None:
    """Return the maintainer binding for a checked-in HiLift preview.

    This is deliberately not a general candidate acceptance path.  The source
    path, submission metadata, historical validation receipt, and archive
    identity of every permitted entry are pinned in code so another closed
    candidate cannot enter the public feed merely by declaring itself a
    preview.
    """

    effective_root = (root or ROOT).resolve()
    configuration = _registered_hiliftaeroml_preview_config(
        path, root=effective_root
    )
    if configuration is None:
        return None
    binding = configuration["binding"]
    if manifest.get("data_release", {}).get("status") != "prototype_dummy_data":
        return None
    if (
        submission.get("submission_id")
        != binding["submission_id"]
        or submission.get("dataset_id") != "hiliftaeroml"
        or submission.get("dataset") != "HiLiftAeroML"
        or submission.get("split_id") != configuration["split_id"]
        or submission.get("case_set_id") != configuration["case_set_id"]
        or submission.get("schema_version") != "3.0"
        or submission.get("approval") is not None
        or submission.get("profile_data", {}).get("format")
        != HILIFT_COMPACT_PROFILE_FORMAT
    ):
        return None
    try:
        if sha256_file(path) != binding["submission_json_sha256"]:
            return None
        receipt = load_json(
            effective_root / configuration["validation_record_path"]
        )
        dataset_spec = load_json(
            effective_root
            / "benchmark-specs"
            / "hiliftaeroml"
            / "submission-spec.json"
        )
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, dict) or not isinstance(dataset_spec, dict):
        return None
    if (
        receipt.get("schema")
        != configuration["validation_record_schema"]
        or receipt.get("status") != "complete_candidate_not_published"
        or receipt.get("usage") != "maintainer_local_candidate_dry_run_only"
        or receipt.get("activation")
        != {
            "owner_approval_complete": False,
            "published": False,
            "submissions_opened": False,
            "validation_changes_activation": False,
        }
        or receipt.get("registered_preview") != binding
    ):
        return None
    scoring_support = dataset_spec.get("scoring_support", {})
    profile_definition = dataset_spec.get("profile_definition", {})
    compact_definition = dataset_spec.get("compact_profile_definition", {})
    if (
        dataset_spec.get("status") != "owner_review_required"
        or scoring_support.get("status") != "owner_review_required"
        or scoring_support.get("submissions_open") is not False
        or profile_definition.get("profile_ground_truth")
        != {
            "status": "not_published",
            "release_id": None,
            "manifest_sha256": None,
        }
        or compact_definition.get("status") != "additive_candidate_not_bound"
        or compact_definition.get("evaluator_support")
        != {
            "status": "not_published",
            "release_id": None,
            "manifest_sha256": None,
        }
    ):
        return None
    assembly = receipt.get("assembly", {})
    archives = receipt.get("deterministic_archives", {})
    scope = receipt.get("scope", {})
    representation = receipt.get("representation", {})
    validation = receipt.get("candidate_validation", {})
    metric_values = receipt.get("metric_values", {})
    evaluation = submission.get("evaluation", {})
    profile_data = submission.get("profile_data", {})
    evidence_path = path.parent / str(evaluation.get("evidence_file", ""))
    profile_index_path = path.parent / str(profile_data.get("index_file", ""))
    try:
        evidence = load_json(evidence_path)
        evidence_sha256 = sha256_file(evidence_path)
        profile_index_sha256 = sha256_file(profile_index_path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(evidence, dict) or not isinstance(metric_values, dict):
        return None
    if (
        assembly.get("submission_json_sha256")
        != binding["submission_json_sha256"]
        or assembly.get("regular_file_bytes_each")
        != configuration["regular_file_bytes_each"]
        or evidence.get("status") != "submitted_evaluation"
        or evidence_sha256 != assembly.get("evaluation_evidence_sha256")
        or evidence_sha256 != evaluation.get("evidence_sha256")
        or profile_index_sha256 != assembly.get("profile_index_sha256")
        or scope.get("split_id") != configuration["split_id"]
        or scope.get("case_set_id") != configuration["case_set_id"]
        or scope.get("case_count") != configuration["case_count"]
        or scope.get("profile_series_count")
        != configuration["profile_series_count"]
        or representation.get("participant_artifacts_contain_evaluator_support")
        is not False
        or representation.get("participant_artifacts_contain_truth") is not False
        or metric_values
        != {
            metric_id: submission.get("metric_values", {}).get(metric_id)
            for metric_id in metric_values
        }
        or profile_data.get("evaluator_support_release_id")
        != receipt.get("bindings", {}).get("evaluator_support_release_id")
        or profile_data.get("evaluator_support_manifest_sha256")
        != receipt.get("bindings", {}).get("evaluator_support_manifest_sha256")
        or profile_data.get("profile_ground_truth_release_id")
        != receipt.get("bindings", {}).get("source_profile_truth_release_id")
        or profile_data.get("profile_ground_truth_manifest_sha256")
        != receipt.get("bindings", {}).get("source_profile_truth_manifest_sha256")
        or archives.get("sha256_each")
        != binding["deterministic_archive"]["sha256"]
        or archives.get("size_bytes_each")
        != binding["deterministic_archive"]["size_bytes"]
        or archives.get("member_count_each")
        != binding["deterministic_archive"]["member_count"]
        or archives.get("byte_identical") is not True
        or any(
            validation.get(package, {}).get("valid") is not True
            for package in ("package_a", "package_b")
        )
    ):
        return None
    return deepcopy(binding)


def validate_registered_hiliftaeroml_preview_archive(
    add: Any,
    directory: Path,
    binding: dict[str, Any],
) -> None:
    """Bind every checked-in preview member to the reviewed deterministic ZIP."""

    from scripts.build_hiliftaeroml_submission_zip import (
        DeterministicZipError,
        build_deterministic_submission_zip,
    )

    expected = binding["deterministic_archive"]
    try:
        with tempfile.TemporaryDirectory(
            prefix="fluidsbench-hilift-preview-"
        ) as temporary:
            archive_path = Path(temporary) / "submission.zip"
            receipt = build_deterministic_submission_zip(directory, archive_path)
            archive_size = archive_path.stat().st_size
    except (DeterministicZipError, OSError) as error:
        add(f"registered HiLiftAeroML preview archive build failed: {error}")
        return
    if receipt.get("archive_sha256") != expected["sha256"]:
        add("registered HiLiftAeroML preview deterministic archive SHA-256 differs")
    if receipt.get("member_count") != expected["member_count"]:
        add("registered HiLiftAeroML preview deterministic archive member count differs")
    if archive_size != expected["size_bytes"]:
        add("registered HiLiftAeroML preview deterministic archive byte size differs")


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def json_path(parts: list[Any]) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def schema_errors(value: Any, schema_name: str, *, schema_version: str = "v1") -> list[str]:
    if Draft202012Validator is None:
        return ["Python dependency jsonschema is missing; run: python3 -m pip install -r requirements.txt"]
    schema = load_json(SCHEMA_ROOT / schema_version / schema_name)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        f"{json_path(list(error.absolute_path))}: {error.message}"
        for error in sorted(validator.iter_errors(value), key=lambda item: json_path(list(item.absolute_path)))
    ]


def methodology_schema_errors(value: Any) -> list[str]:
    """Validate a method record using the canonical schema-v3 definition."""

    if Draft202012Validator is None:
        return ["Python dependency jsonschema is missing; run: python3 -m pip install -r requirements.txt"]
    submission_schema = load_json(SCHEMA_ROOT / "v3" / "submission.schema.json")
    fragment = {
        "$schema": submission_schema["$schema"],
        "$defs": submission_schema["$defs"],
        "$ref": "#/$defs/fluidsbench_methodology",
    }
    validator = Draft202012Validator(fragment, format_checker=FormatChecker())
    return [
        f"{json_path(list(error.absolute_path))}: {error.message}"
        for error in sorted(
            validator.iter_errors(value),
            key=lambda item: json_path(list(item.absolute_path)),
        )
    ]


def _load_drivaerml_release_json(path: Path, *, label: str) -> tuple[Any, str]:
    """Load and hash one immutable local release document from identical bytes."""

    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_submission_duplicate_keys,
            parse_constant=_reject_submission_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, SubmissionJSONError) as error:
        raise SubmissionJSONError(f"cannot load {label}: {error}") from error
    return value, hashlib.sha256(payload).hexdigest()


def _drivaerml_release_path(
    value: Any,
    *,
    base: Path,
    repository_root: Path,
    label: str,
    confined_to_base: bool = False,
) -> Path:
    """Resolve a release path locally, permitting no repository escape."""

    if not isinstance(value, str) or not value:
        raise SubmissionJSONError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise SubmissionJSONError(f"{label} must be relative")
    try:
        resolved_root = repository_root.resolve(strict=True)
        resolved_base = base.resolve(strict=True)
        resolved = (resolved_base / relative).resolve(strict=True)
    except OSError as error:
        raise SubmissionJSONError(f"{label} does not resolve to a retained file") from error
    allowed_root = resolved_base if confined_to_base else resolved_root
    if not resolved.is_relative_to(allowed_root):
        scope = "the DrivAerML benchmark directory" if confined_to_base else "the repository"
        raise SubmissionJSONError(f"{label} must remain inside {scope}")
    if not resolved.is_file():
        raise SubmissionJSONError(f"{label} is not a retained regular file")
    return resolved


def _drivaerml_case_ids(document: Any, *, label: str) -> list[str]:
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        raise SubmissionJSONError(f"{label}.cases must be an array")
    case_ids: list[str] = []
    for position, case in enumerate(document["cases"]):
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise SubmissionJSONError(f"{label}.cases[{position}].case_id is invalid")
        case_ids.append(case["case_id"])
    if len(case_ids) != len(set(case_ids)):
        raise SubmissionJSONError(f"{label} contains duplicate case IDs")
    return case_ids


def _drivaerml_git_commit_is_bound(repository_root: Path, revision: str) -> bool:
    """Return whether ``revision`` is a real commit reachable from local HEAD.

    Activation validation is intentionally offline.  A syntactically plausible
    object ID is not evidence: the exact commit must be available in this clone
    and belong to the checked-out repository history.
    """

    try:
        object_type = subprocess.run(
            ["git", "-C", str(repository_root), "cat-file", "-t", revision],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        if object_type.returncode != 0 or object_type.stdout.strip() != "commit":
            return False
        reachable = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "merge-base",
                "--is-ancestor",
                revision,
                "HEAD",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return reachable.returncode == 0


def _drivaerml_git_commit_is_ancestor(
    repository_root: Path,
    ancestor: str,
    descendant: str,
) -> bool:
    """Return whether two retained commits have the required ancestry."""

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _drivaerml_git_commit_is_on_trusted_dev(
    repository_root: Path,
    revision: str,
) -> bool:
    """Bind owner approval to the fetched upstream development branch.

    Reachability from ``HEAD`` alone is not an approval boundary: a pull-request
    author controls new commits on that history.  The approval record must
    already be present on the locally fetched ``origin/dev`` from the expected
    evaluator repository.  Validation remains offline; callers must fetch the
    full upstream history before checking an activated release.
    """

    try:
        remote = subprocess.run(
            ["git", "-C", str(repository_root), "remote", "get-url", "origin"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if remote.returncode != 0:
        return False
    parsed = urlparse(remote.stdout.strip())
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "github.com"
        or parsed.path.rstrip("/").removesuffix(".git")
        != "/neilashton/fluidsbench-submission"
    ):
        return False
    return _drivaerml_git_commit_is_ancestor(
        repository_root,
        revision,
        DRIVAERML_RELATIVE_APPROVAL_TRUSTED_REF,
    )


def _drivaerml_git_blob_sha256(
    repository_root: Path,
    revision: str,
    path: Path,
) -> str | None:
    """Hash the exact repository-relative file bytes retained by a commit."""

    try:
        relative = path.resolve(strict=True).relative_to(
            repository_root.resolve(strict=True)
        )
    except (OSError, ValueError):
        return None
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "cat-file",
                "blob",
                f"{revision}:{relative.as_posix()}",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _drivaerml_hex_looks_placeholder(value: str) -> bool:
    """Reject low-entropy or short-period hex strings used as fake identities."""

    if len(set(value)) < 8:
        return True
    for period in range(1, min(16, len(value) // 2) + 1):
        if value == (value[:period] * ((len(value) + period - 1) // period))[: len(value)]:
            return True
    return False


def _drivaerml_text_looks_placeholder(value: str) -> bool:
    """Return whether a repository/path identity contains an explicit fake token."""

    placeholder_tokens = {
        "dummy",
        "example",
        "fake",
        "placeholder",
        "synthetic",
        "test",
        "testing",
    }
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", value.lower())
        if token
    }
    return bool(tokens & placeholder_tokens)


def validate_drivaerml_relative_activation_release(
    add: Any,
    dataset_spec: dict[str, Any],
    declaration: dict[str, Any],
    *,
    repository_root: Path | None = None,
    require_active: bool = True,
) -> bool:
    """Verify the complete local release chain authorizing relative profiles.

    The declaration contains only a path and raw-byte digest.  Every referenced
    artifact is loaded from the repository, hashed from the bytes that were
    parsed, and cross-checked before an activation record can authorize the
    format.  No URL is fetched and a pending support-publication record is
    never treated as activation.
    """

    errors: list[str] = []

    def problem(message: str) -> None:
        errors.append(f"DrivAerML relative activation release: {message}")

    root = (ROOT if repository_root is None else repository_root).resolve()
    dataset_directory = root / "benchmark-specs" / "drivaerml"

    pointer = declaration.get("activation_release")
    if not isinstance(pointer, dict) or set(pointer) != {"file", "sha256"}:
        problem("relative_diagnostics.activation_release must contain exactly file and sha256")
        for message in errors:
            add(message)
        return False
    pointer_digest = pointer.get("sha256")
    if (
        not isinstance(pointer_digest, str)
        or LOWER_SHA256.fullmatch(pointer_digest) is None
        or pointer_digest == "0" * 64
    ):
        problem("activation_release.sha256 must be a nonzero lowercase SHA-256")
        for message in errors:
            add(message)
        return False
    try:
        record_path = _drivaerml_release_path(
            pointer.get("file"),
            base=dataset_directory,
            repository_root=root,
            label="activation_release.file",
            confined_to_base=True,
        )
        record, actual_record_digest = _load_drivaerml_release_json(
            record_path,
            label="relative activation release record",
        )
    except SubmissionJSONError as error:
        problem(str(error))
        for message in errors:
            add(message)
        return False
    if actual_record_digest != pointer_digest:
        problem("activation_release.sha256 does not match the retained record bytes")
    if not isinstance(record, dict):
        problem("activation release record must be an object")
        for message in errors:
            add(message)
        return False

    record_fields = {
        "schema",
        "schema_version",
        "release_id",
        "dataset_id",
        "status",
        "profile_format_authorized",
        "submissions_opened_by_this_record",
        "relative_composite_weight",
        "bindings",
        "evaluator",
        "sensitivity_evidence",
        "owner_approval",
        "activation_gates",
    }
    if set(record) != record_fields:
        problem(f"record fields must be exactly {sorted(record_fields)}")
    if record.get("schema") != DRIVAERML_RELATIVE_ACTIVATION_RECORD_SCHEMA:
        problem("record schema is unsupported")
    if record.get("schema_version") != 1:
        problem("record schema_version must equal 1")
    if record.get("dataset_id") != "drivaerml":
        problem("record dataset_id must equal 'drivaerml'")
    release_id = record.get("release_id")
    if not isinstance(release_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,159}", release_id) is None:
        problem("record release_id is invalid")
    if record.get("status") not in {"support_verified_activation_pending", "activated"}:
        problem("record status must be support_verified_activation_pending or activated")
    if not isinstance(record.get("profile_format_authorized"), bool):
        problem("record profile_format_authorized must be boolean")
    if record.get("submissions_opened_by_this_record") is not False:
        problem("record must not open submissions")
    relative_weight = record.get("relative_composite_weight")
    if isinstance(relative_weight, bool) or relative_weight != 0.0:
        problem("record relative_composite_weight must remain 0.0")

    bindings = record.get("bindings")
    binding_names = {
        "contract",
        "profile_chunk_schema",
        "official_case_registry",
        "velocity_placement_manifest",
        "velocity_mapping_manifest",
        "cp_manifest",
        "series_support_index",
        "constant_series_support_index",
    }
    if not isinstance(bindings, dict) or set(bindings) != binding_names:
        problem(f"record bindings must be exactly {sorted(binding_names)}")
        bindings = {}

    loaded: dict[str, tuple[dict[str, Any], Path, str]] = {}
    simple_bindings = {"contract", "profile_chunk_schema"}
    for name in sorted(binding_names):
        binding = bindings.get(name)
        expected_fields = {"file", "sha256"} if name in simple_bindings else {
            "file",
            "sha256",
            "schema",
            "case_count",
        }
        if not isinstance(binding, dict) or set(binding) != expected_fields:
            problem(f"bindings.{name} must contain exactly {sorted(expected_fields)}")
            continue
        digest = binding.get("sha256")
        if (
            not isinstance(digest, str)
            or LOWER_SHA256.fullmatch(digest) is None
            or digest == "0" * 64
        ):
            problem(f"bindings.{name}.sha256 must be a nonzero lowercase SHA-256")
            continue
        try:
            path = _drivaerml_release_path(
                binding.get("file"),
                base=dataset_directory,
                repository_root=root,
                label=f"bindings.{name}.file",
            )
            document, actual_digest = _load_drivaerml_release_json(
                path,
                label=f"bindings.{name}",
            )
        except SubmissionJSONError as error:
            problem(str(error))
            continue
        if actual_digest != digest:
            problem(f"bindings.{name}.sha256 does not match the retained bytes")
        if not isinstance(document, dict):
            problem(f"bindings.{name} must contain a JSON object")
            continue
        loaded[name] = (document, path, actual_digest)

    contract_entry = loaded.get("contract")
    contract = contract_entry[0] if contract_entry is not None else {}
    contract_binding = bindings.get("contract") if isinstance(bindings, dict) else None
    if (
        not isinstance(contract_binding, dict)
        or contract_binding.get("sha256") != RELATIVE_PROFILE_CONTRACT_SHA256
    ):
        problem(
            "record contract digest differs from the independently pinned "
            "candidate-release contract"
        )
    declared_contract = declaration.get("contract")
    if not isinstance(declared_contract, dict):
        problem("relative_diagnostics.contract must be an object")
    elif isinstance(contract_binding, dict):
        if declared_contract.get("sha256") != contract_binding.get("sha256"):
            problem("record contract digest differs from relative_diagnostics.contract")
        try:
            declared_contract_path = _drivaerml_release_path(
                declared_contract.get("file"),
                base=dataset_directory,
                repository_root=root,
                label="relative_diagnostics.contract.file",
                confined_to_base=True,
            )
        except SubmissionJSONError as error:
            problem(str(error))
        else:
            if contract_entry is not None and declared_contract_path != contract_entry[1]:
                problem("record contract file differs from relative_diagnostics.contract")
    if contract:
        if contract.get("id") != RELATIVE_PROFILE_CONTRACT_ID:
            problem("contract ID differs from the relative profile format")
        if contract.get("dataset_id") != "drivaerml":
            problem("contract dataset_id must equal 'drivaerml'")
        if contract.get("activation_by_this_file") is not False:
            problem("the relative contract must not activate itself")
        rollout = contract.get("scoring_and_rollout")
        if (
            not isinstance(rollout, dict)
            or rollout.get("relative_composite_weight") != 0.0
            or rollout.get("submissions_opened_by_this_contract") is not False
        ):
            problem("the relative contract scoring rollout is not fail-closed")

    schema_entry = loaded.get("profile_chunk_schema")
    if schema_entry is not None:
        schema_document, schema_path, _digest = schema_entry
        if bindings.get("profile_chunk_schema", {}).get("sha256") != (
            DRIVAERML_RELATIVE_PROFILE_CHUNK_SCHEMA_SHA256
        ):
            problem(
                "record profile schema digest differs from the independently "
                "pinned candidate-release schema"
            )
        if schema_document.get("$id") != (
            "https://fluidsbench.org/schemas/v1/"
            "drivaerml-relative-profile-chunk.schema.json"
        ):
            problem("profile chunk schema has the wrong $id")
        profile_declaration = declaration.get("profile_chunk")
        if not isinstance(profile_declaration, dict):
            problem("relative_diagnostics.profile_chunk must be an object")
        else:
            if profile_declaration.get("schema_sha256") != bindings.get(
                "profile_chunk_schema", {}
            ).get("sha256"):
                problem("record profile schema digest differs from the benchmark declaration")
            try:
                declared_schema_path = _drivaerml_release_path(
                    profile_declaration.get("schema_file"),
                    base=root,
                    repository_root=root,
                    label="relative_diagnostics.profile_chunk.schema_file",
                )
            except SubmissionJSONError as error:
                problem(str(error))
            else:
                if declared_schema_path != schema_path:
                    problem("record profile schema file differs from the benchmark declaration")
        profile_contract = contract.get("profile_chunk_contract")
        if isinstance(profile_contract, dict):
            try:
                contract_schema_path = _drivaerml_release_path(
                    profile_contract.get("schema_file"),
                    base=root,
                    repository_root=root,
                    label="contract profile_chunk_contract.schema_file",
                )
            except SubmissionJSONError as error:
                problem(str(error))
            else:
                if contract_schema_path != schema_path:
                    problem("record profile schema file differs from the contract")

    registry_entry = loaded.get("official_case_registry")
    official_case_ids: list[str] = []
    source_revision: str | None = None
    if registry_entry is not None:
        registry, _path, _digest = registry_entry
        registry_binding = bindings.get("official_case_registry", {})
        if registry_binding.get("schema") != DRIVAERML_OFFICIAL_CASE_REGISTRY_SCHEMA:
            problem("official_case_registry declares the wrong schema")
        if registry.get("schema") != registry_binding.get("schema"):
            problem("official_case_registry schema does not match its binding")
        if registry_binding.get("case_count") != DRIVAERML_OFFICIAL_CASE_COUNT:
            problem("official_case_registry binding must declare 484 cases")
        try:
            official_case_ids = _drivaerml_case_ids(
                registry,
                label="official_case_registry",
            )
        except SubmissionJSONError as error:
            problem(str(error))
        if len(official_case_ids) != DRIVAERML_OFFICIAL_CASE_COUNT:
            problem("official_case_registry must contain exactly 484 cases")
        case_scope = registry.get("case_scope")
        if not isinstance(case_scope, dict) or case_scope.get("case_count") != 484:
            problem("official_case_registry case_scope must declare 484 cases")
        repository = registry.get("repository")
        if isinstance(repository, dict) and isinstance(repository.get("revision"), str):
            source_revision = repository["revision"]
        else:
            problem("official_case_registry repository revision is missing")

    manifest_digests: dict[str, str] = {}
    for name, expected_schema in DRIVAERML_RELATIVE_MANIFEST_SCHEMAS.items():
        entry = loaded.get(name)
        if entry is None:
            continue
        manifest, _path, actual_digest = entry
        binding = bindings.get(name, {})
        if binding.get("schema") != expected_schema:
            problem(f"bindings.{name}.schema differs from the required producer schema")
        if manifest.get("schema") != expected_schema:
            problem(f"{name} has the wrong producer schema")
        if binding.get("case_count") != DRIVAERML_OFFICIAL_CASE_COUNT:
            problem(f"bindings.{name}.case_count must equal 484")
        if manifest.get("family_id") != DRIVAERML_RELATIVE_MANIFEST_FAMILIES[name]:
            problem(f"{name} has the wrong family_id")
        if source_revision is not None and manifest.get("public_dataset_revision") != source_revision:
            problem(f"{name} targets a different public dataset revision")
        try:
            manifest_case_ids = _drivaerml_case_ids(manifest, label=name)
        except SubmissionJSONError as error:
            problem(str(error))
            manifest_case_ids = []
        if official_case_ids and manifest_case_ids != official_case_ids:
            problem(f"{name} cases must exactly match the ordered official 484-case registry")
        if name in {"velocity_placement_manifest", "cp_manifest"}:
            if manifest.get("case_count") != DRIVAERML_OFFICIAL_CASE_COUNT:
                problem(f"{name}.case_count must equal 484")
        elif (
            manifest.get("official_case_count") != DRIVAERML_OFFICIAL_CASE_COUNT
            or manifest.get("included_case_count") != DRIVAERML_OFFICIAL_CASE_COUNT
            or manifest.get("complete_official_case_coverage") is not True
        ):
            problem("velocity_mapping_manifest does not claim complete 484-case coverage")
        if name == "velocity_placement_manifest" and manifest.get("case_ids") != official_case_ids:
            problem("velocity_placement_manifest.case_ids must match its case records")
        if name == "cp_manifest" and manifest.get(
            "all_official_cases_generated_and_replayed"
        ) is not True:
            problem("cp_manifest does not claim all official cases were generated and replayed")
        manifest_digests[name] = actual_digest

    implementation_bindings = contract.get("support_implementation_bindings")
    if isinstance(implementation_bindings, dict):
        velocity = implementation_bindings.get("relative_velocity_v3")
        cp = implementation_bindings.get("relative_cp_v1")
        expected_contract_digests = {
            "velocity_placement_manifest": (
                velocity.get("placement_all484_manifest", {}).get("sha256")
                if isinstance(velocity, dict)
                else None
            ),
            "velocity_mapping_manifest": (
                velocity.get("mapping_all484_manifest", {}).get("sha256")
                if isinstance(velocity, dict)
                else None
            ),
            "cp_manifest": (
                cp.get("cp_all484_manifest", {}).get("sha256")
                if isinstance(cp, dict)
                else None
            ),
        }
        for name, digest in manifest_digests.items():
            if expected_contract_digests.get(name) != digest:
                problem(f"{name} digest differs from the retained relative contract")
    elif contract:
        problem("contract support_implementation_bindings are missing")

    index_entry = loaded.get("series_support_index")
    if index_entry is not None:
        support_index, _path, _digest = index_entry
        index_binding = bindings.get("series_support_index", {})
        index_fields = {
            "case_count",
            "cases",
            "contract_id",
            "dataset_id",
            "schema",
            "schema_version",
            "scope",
            "series_per_case",
            "source_bindings",
        }
        if set(support_index) != index_fields:
            problem(f"series_support_index fields must be exactly {sorted(index_fields)}")
        if index_binding.get("schema") != DRIVAERML_RELATIVE_SUPPORT_INDEX_SCHEMA:
            problem("series_support_index binding declares the wrong schema")
        if support_index.get("schema") != DRIVAERML_RELATIVE_SUPPORT_INDEX_SCHEMA:
            problem("series_support_index has the wrong schema")
        if support_index.get("schema_version") != 2:
            problem("series_support_index schema_version must equal 2")
        if support_index.get("contract_id") != RELATIVE_PROFILE_CONTRACT_ID:
            problem("series_support_index contract_id differs from the relative contract")
        if support_index.get("scope") != "relative_families_only":
            problem("series_support_index scope must equal relative_families_only")
        if index_binding.get("case_count") != 484 or support_index.get("case_count") != 484:
            problem("series_support_index must declare exactly 484 cases")
        if support_index.get("series_per_case") != 20:
            problem("series_support_index must declare exactly 20 relative series per case")
        if support_index.get("dataset_id") != "drivaerml":
            problem("series_support_index dataset_id must equal 'drivaerml'")

        source_bindings = support_index.get("source_bindings")
        source_binding_fields = {
            "manifests",
            "producer_identity_fields",
            "public_dataset",
        }
        if (
            not isinstance(source_bindings, dict)
            or set(source_bindings) != source_binding_fields
        ):
            problem(
                "series_support_index.source_bindings must contain exactly "
                f"{sorted(source_binding_fields)}"
            )
            source_bindings = {}

        public_dataset = source_bindings.get("public_dataset")
        public_dataset_fields = {
            "native_source_pin_path",
            "native_source_pin_sha256",
            "repository",
            "revision",
        }
        if (
            not isinstance(public_dataset, dict)
            or set(public_dataset) != public_dataset_fields
        ):
            problem(
                "series_support_index.source_bindings.public_dataset must contain "
                f"exactly {sorted(public_dataset_fields)}"
            )
        elif registry_entry is not None:
            registry, registry_path, registry_digest = registry_entry
            registry_repository = registry.get("repository")
            expected_repository = (
                registry_repository.get("repo_id")
                if isinstance(registry_repository, dict)
                else None
            )
            expected_public_dataset = {
                "native_source_pin_path": registry_path.relative_to(root).as_posix(),
                "native_source_pin_sha256": registry_digest,
                "repository": expected_repository,
                "revision": source_revision,
            }
            if public_dataset != expected_public_dataset:
                problem(
                    "series_support_index public-dataset provenance differs from "
                    "the retained official case registry"
                )

        manifest_roles = {
            "velocity_placement_manifest": "velocity_placement",
            "velocity_mapping_manifest": "velocity_mapping",
            "cp_manifest": "cp",
        }
        expected_index_manifests: list[dict[str, Any]] = []
        for manifest_name, role in manifest_roles.items():
            manifest_entry = loaded.get(manifest_name)
            if manifest_entry is None:
                continue
            manifest, manifest_path, manifest_digest = manifest_entry
            expected_index_manifests.append(
                {
                    "path": manifest_path.relative_to(root).as_posix(),
                    "role": role,
                    "schema": manifest.get("schema"),
                    "schema_version": manifest.get("schema_version"),
                    "sha256": manifest_digest,
                }
            )
        index_manifests = source_bindings.get("manifests")
        if index_manifests != expected_index_manifests:
            problem(
                "series_support_index manifest provenance differs from the three "
                "retained release manifest bindings"
            )
        expected_identity_fields = {
            "relative_cp_materialized_coordinate": (
                "support.moving_cuts[].rows[].interval_arc_end_m encoded by "
                "fluidsbench-drivaerml-coordinate-array-v1"
            ),
            "relative_cp_moving_support": (
                "support.moving_cuts[].support_identity_sha256"
            ),
            "relative_cp_placement_receipt": "receipt_identity.sha256",
            "relative_cp_shared_alias_support": (
                "support.centerline_aliases[].canonical_cut_support_sha256"
            ),
            "relative_velocity_placement_receipt": (
                "sha256(exact placement receipt bytes)"
            ),
            "relative_velocity_materialized_coordinate": (
                "placement CSV line_fraction selected in order by valid rows from "
                "the aggregate-bound 10mm mapping, encoded by "
                "fluidsbench-drivaerml-coordinate-array-v1"
            ),
            "relative_velocity_support": (
                "profiles[].coordinates_binary64_be_sha256"
            ),
        }
        if source_bindings.get("producer_identity_fields") != expected_identity_fields:
            problem(
                "series_support_index producer_identity_fields differ from the "
                "frozen producer identity definitions"
            )

        expected_series: dict[tuple[str, str], str] = {}
        raw_families = contract.get("families")
        if isinstance(raw_families, list):
            raw_shared_support_groups = contract.get("shared_support_groups")
            if not isinstance(raw_shared_support_groups, list):
                problem("contract shared_support_groups must be an array")
                raw_shared_support_groups = []
            shared_aliases = {
                str(group.get("alias_member", "")).split(":", 1)[1]
                for group in raw_shared_support_groups
                if isinstance(group, dict)
                and isinstance(group.get("alias_member"), str)
                and ":" in group["alias_member"]
            }
            for family in raw_families:
                if not isinstance(family, dict) or family.get("family_id") not in {
                    "drivaerml-velocity-relative-v3",
                    "drivaerml_cp_relative_v1",
                }:
                    continue
                family_id = family["family_id"]
                station_ids = family.get("station_ids")
                if not isinstance(station_ids, list):
                    continue
                for station_id in station_ids:
                    if not isinstance(station_id, str):
                        continue
                    representation = (
                        "shared_alias"
                        if family_id == "drivaerml_cp_relative_v1"
                        and station_id in shared_aliases
                        else "materialized"
                    )
                    expected_series[(family_id, station_id)] = representation
        if len(expected_series) != 20:
            problem("contract must define exactly 20 relative family/station keys")
        try:
            index_case_ids = _drivaerml_case_ids(
                support_index,
                label="series_support_index",
            )
        except SubmissionJSONError as error:
            problem(str(error))
            index_case_ids = []
        if official_case_ids and index_case_ids != official_case_ids:
            problem("series_support_index cases must exactly match the official registry")
        raw_index_cases = support_index.get("cases")
        if not isinstance(raw_index_cases, list):
            # _drivaerml_case_ids already records the precise schema error.  Keep
            # iterating over an empty fail-closed value instead of raising here.
            raw_index_cases = []
        for case_position, case in enumerate(raw_index_cases):
            if not isinstance(case, dict) or set(case) != {"case_id", "series"}:
                problem(f"series_support_index.cases[{case_position}] fields are invalid")
                continue
            series = case.get("series")
            if not isinstance(series, list) or len(series) != 20:
                problem(f"series_support_index {case.get('case_id')} must contain 20 entries")
                continue
            observed: dict[tuple[str, str], str] = {}
            for series_position, item in enumerate(series):
                common_fields = {
                    "family_id",
                    "station_id",
                    "representation",
                    "support_identity_sha256",
                    "placement_receipt_identity_sha256",
                }
                representation = (
                    item.get("representation") if isinstance(item, dict) else None
                )
                expected_fields = (
                    common_fields
                    | {"coordinate_count", "coordinate_identity_sha256"}
                    if representation == "materialized"
                    else common_fields
                )
                if not isinstance(item, dict) or set(item) != expected_fields:
                    problem(
                        f"series_support_index {case.get('case_id')} series "
                        f"{series_position} fields are invalid"
                    )
                    continue
                key = (item.get("family_id"), item.get("station_id"))
                if key in observed:
                    problem(f"series_support_index {case.get('case_id')} repeats {key}")
                observed[key] = item.get("representation")
                for digest_field in (
                    "support_identity_sha256",
                    "placement_receipt_identity_sha256",
                ):
                    digest = item.get(digest_field)
                    if (
                        not isinstance(digest, str)
                        or LOWER_SHA256.fullmatch(digest) is None
                        or digest == "0" * 64
                    ):
                        problem(
                            f"series_support_index {case.get('case_id')} {key} "
                            f"has an invalid {digest_field}"
                        )
                if representation == "materialized":
                    coordinate_count = item.get("coordinate_count")
                    if (
                        not isinstance(coordinate_count, int)
                        or isinstance(coordinate_count, bool)
                        or coordinate_count < 2
                    ):
                        problem(
                            f"series_support_index {case.get('case_id')} {key} "
                            "has an invalid coordinate_count"
                        )
                    coordinate_digest = item.get("coordinate_identity_sha256")
                    if (
                        not isinstance(coordinate_digest, str)
                        or LOWER_SHA256.fullmatch(coordinate_digest) is None
                        or coordinate_digest == "0" * 64
                    ):
                        problem(
                            f"series_support_index {case.get('case_id')} {key} "
                            "has an invalid coordinate_identity_sha256"
                        )
            if observed != expected_series:
                problem(
                    f"series_support_index {case.get('case_id')} does not exactly "
                    "cover the relative contract namespace"
                )

    constant_index_entry = loaded.get("constant_series_support_index")
    if constant_index_entry is not None:
        constant_index, _constant_path, _constant_digest = constant_index_entry
        constant_binding = bindings.get("constant_series_support_index", {})
        constant_fields = {
            "case_count",
            "cases",
            "contract_id",
            "dataset_id",
            "schema",
            "schema_version",
            "scope",
            "series_per_case",
            "source_bindings",
        }
        if set(constant_index) != constant_fields:
            problem(
                "constant_series_support_index fields must be exactly "
                f"{sorted(constant_fields)}"
            )
        if constant_binding.get("schema") != DRIVAERML_CONSTANT_SUPPORT_INDEX_SCHEMA:
            problem("constant_series_support_index binding declares the wrong schema")
        if constant_index.get("schema") != DRIVAERML_CONSTANT_SUPPORT_INDEX_SCHEMA:
            problem("constant_series_support_index has the wrong schema")
        if constant_index.get("schema_version") != 1:
            problem("constant_series_support_index schema_version must equal 1")
        if constant_index.get("contract_id") != RELATIVE_PROFILE_CONTRACT_ID:
            problem("constant_series_support_index contract_id differs from the contract")
        if constant_index.get("scope") != "constant_families_all_official_cases":
            problem(
                "constant_series_support_index scope must equal "
                "constant_families_all_official_cases"
            )
        if (
            constant_binding.get("case_count") != 484
            or constant_index.get("case_count") != 484
        ):
            problem("constant_series_support_index must declare exactly 484 cases")
        if constant_index.get("series_per_case") != 20:
            problem(
                "constant_series_support_index must declare exactly 20 series per case"
            )
        if constant_index.get("dataset_id") != "drivaerml":
            problem("constant_series_support_index dataset_id must equal 'drivaerml'")

        constant_sources = constant_index.get("source_bindings")
        expected_constant_source_fields = {
            "coordinate_identity_encoding",
            "native_profile_truth",
            "public_dataset",
        }
        if (
            not isinstance(constant_sources, dict)
            or set(constant_sources) != expected_constant_source_fields
        ):
            problem(
                "constant_series_support_index.source_bindings must contain exactly "
                f"{sorted(expected_constant_source_fields)}"
            )
            constant_sources = {}
        expected_native_truth = {
            "master_index_path": (
                "assets/data/profile-ground-truth/datasets/drivaerml/"
                "native-v3/index.json"
            ),
            "master_index_sha256": (
                "e7cf14f161fc7dbf22794e6f66db4e157329be960d2a4368140e08cf0608a5ae"
            ),
            "provenance_sha256": (
                "6030df1dce11c4fbf6028d4e17ab39394cfdd3e7bb4d5a7ff81f33408849f221"
            ),
            "release_receipt_sha256": (
                "ef59ca838c828ac1c5505ef4d35f1b91a55e07ad90df359491599114ec0d878c"
            ),
            "dataset_repository": "neashton/drivaerml",
            "dataset_revision": source_revision,
        }
        if constant_sources.get("native_profile_truth") != expected_native_truth:
            problem(
                "constant_series_support_index native-profile-truth provenance "
                "differs from the published native-v3 release"
            )
        if constant_sources.get("coordinate_identity_encoding") != (
            "fluidsbench-drivaerml-coordinate-array-v1"
        ):
            problem(
                "constant_series_support_index coordinate identity encoding is invalid"
            )
        constant_public_dataset = constant_sources.get("public_dataset")
        if registry_entry is not None:
            registry, registry_path, registry_digest = registry_entry
            registry_repository = registry.get("repository")
            expected_repository = (
                registry_repository.get("repo_id")
                if isinstance(registry_repository, dict)
                else None
            )
            expected_constant_public_dataset = {
                "native_source_pin_path": registry_path.relative_to(root).as_posix(),
                "native_source_pin_sha256": registry_digest,
                "repository": expected_repository,
                "revision": source_revision,
            }
            if constant_public_dataset != expected_constant_public_dataset:
                problem(
                    "constant_series_support_index public-dataset provenance "
                    "differs from the official registry"
                )

        expected_constant_series: set[tuple[str, str]] = set()
        raw_families = contract.get("families")
        if isinstance(raw_families, list):
            for family in raw_families:
                if not isinstance(family, dict) or family.get("family_id") not in {
                    "drivaerml-autocfd5-constant-v1",
                    "drivaerml_cp_constant_v1",
                }:
                    continue
                station_ids = family.get("station_ids")
                if not isinstance(station_ids, list):
                    continue
                expected_constant_series.update(
                    (family["family_id"], station_id)
                    for station_id in station_ids
                    if isinstance(station_id, str)
                )
        if len(expected_constant_series) != 20:
            problem("contract must define exactly 20 constant family/station keys")
        try:
            constant_case_ids = _drivaerml_case_ids(
                constant_index,
                label="constant_series_support_index",
            )
        except SubmissionJSONError as error:
            problem(str(error))
            constant_case_ids = []
        if official_case_ids and constant_case_ids != official_case_ids:
            problem(
                "constant_series_support_index cases must exactly match the "
                "official registry"
            )
        raw_constant_cases = constant_index.get("cases")
        if not isinstance(raw_constant_cases, list):
            raw_constant_cases = []
        constant_item_fields = {
            "family_id",
            "station_id",
            "representation",
            "support_identity_sha256",
            "placement_receipt_identity_sha256",
            "coordinate_count",
            "coordinate_identity_sha256",
        }
        for case_position, case in enumerate(raw_constant_cases):
            if not isinstance(case, dict) or set(case) != {"case_id", "series"}:
                problem(
                    "constant_series_support_index.cases"
                    f"[{case_position}] fields are invalid"
                )
                continue
            series = case.get("series")
            if not isinstance(series, list) or len(series) != 20:
                problem(
                    f"constant_series_support_index {case.get('case_id')} must "
                    "contain 20 entries"
                )
                continue
            observed_constant: set[tuple[str, str]] = set()
            for series_position, item in enumerate(series):
                if not isinstance(item, dict) or set(item) != constant_item_fields:
                    problem(
                        f"constant_series_support_index {case.get('case_id')} "
                        f"series {series_position} fields are invalid"
                    )
                    continue
                key = (item.get("family_id"), item.get("station_id"))
                if key in observed_constant:
                    problem(
                        f"constant_series_support_index {case.get('case_id')} "
                        f"repeats {key}"
                    )
                observed_constant.add(key)
                if item.get("representation") != "materialized":
                    problem(
                        f"constant_series_support_index {case.get('case_id')} "
                        f"{key} must be materialized"
                    )
                for digest_field in (
                    "support_identity_sha256",
                    "placement_receipt_identity_sha256",
                    "coordinate_identity_sha256",
                ):
                    digest = item.get(digest_field)
                    if (
                        not isinstance(digest, str)
                        or LOWER_SHA256.fullmatch(digest) is None
                        or digest == "0" * 64
                    ):
                        problem(
                            f"constant_series_support_index {case.get('case_id')} "
                            f"{key} has an invalid {digest_field}"
                        )
                coordinate_count = item.get("coordinate_count")
                if (
                    not isinstance(coordinate_count, int)
                    or isinstance(coordinate_count, bool)
                    or coordinate_count < 2
                ):
                    problem(
                        f"constant_series_support_index {case.get('case_id')} "
                        f"{key} has an invalid coordinate_count"
                    )
            if observed_constant != expected_constant_series:
                problem(
                    f"constant_series_support_index {case.get('case_id')} does "
                    "not exactly cover the constant contract namespace"
                )

    evaluator_error_count = len(errors)
    evaluator = record.get("evaluator")
    evaluator_fields = {"status", "repository", "git_revision", "reference_version"}
    if not isinstance(evaluator, dict) or set(evaluator) != evaluator_fields:
        problem(f"record evaluator must contain exactly {sorted(evaluator_fields)}")
        evaluator = {}
    evaluator_revision = evaluator.get("git_revision")
    if evaluator.get("status") != "frozen":
        problem("record evaluator status must be frozen")
    if evaluator.get("repository") != DRIVAERML_EVALUATOR_REPOSITORY:
        problem("record evaluator repository is invalid")
    if (
        not isinstance(evaluator_revision, str)
        or LOWER_GIT_SHA1.fullmatch(evaluator_revision) is None
        or evaluator_revision == "0" * 40
    ):
        problem("record evaluator git_revision must be an exact nonzero 40-hex commit")
    elif not _drivaerml_git_commit_is_bound(root, evaluator_revision):
        problem(
            "record evaluator git_revision is not a commit reachable in the "
            "local repository"
        )
    if evaluator.get("reference_version") != dataset_spec.get("evaluation_reference_version"):
        problem("record evaluator reference_version differs from the benchmark specification")
    evaluator_is_valid = len(errors) == evaluator_error_count
    evaluator_binding = dataset_spec.get("scoring_support", {}).get(
        "dataset_evaluator_binding"
    )
    if (require_active or record.get("status") == "activated") and (
        not isinstance(evaluator_binding, dict)
        or evaluator_binding.get("status") != "frozen"
        or evaluator_binding.get("repository_url") != evaluator.get("repository")
        or evaluator_binding.get("evaluator_reference_version")
        != evaluator.get("reference_version")
        or evaluator_binding.get("evaluator_code_revision") != evaluator_revision
    ):
        problem("an active release requires the benchmark frozen evaluator binding")

    sensitivity_error_count = len(errors)
    # An approved release must prove that the owner-reviewed Git commit saw
    # these exact compact evidence bytes.  The large checkpoint/prediction
    # artifacts remain external and are identified by their immutable locator
    # and SHA-256; their retained provenance and manifests are small.
    sensitivity_review_artifacts: list[tuple[str, Path, str]] = []
    sensitivity = record.get("sensitivity_evidence")
    sensitivity_fields = {
        "status",
        "file",
        "sha256",
        "schema",
        "model_checkpoint_count",
        "bootstrap_replicate_count",
    }
    if not isinstance(sensitivity, dict) or set(sensitivity) != sensitivity_fields:
        problem(
            f"record sensitivity_evidence must contain exactly {sorted(sensitivity_fields)}"
        )
        sensitivity = {}
    if sensitivity.get("schema") != DRIVAERML_RELATIVE_SENSITIVITY_SCHEMA:
        problem("record sensitivity_evidence schema is invalid")
    if sensitivity.get("status") not in {"pending", "passed"}:
        problem("record sensitivity_evidence status must be pending or passed")
    if sensitivity.get("status") == "pending":
        if (
            sensitivity.get("file") is not None
            or sensitivity.get("sha256") is not None
            or sensitivity.get("model_checkpoint_count") != 0
            or sensitivity.get("bootstrap_replicate_count") != 0
        ):
            problem("pending sensitivity evidence must use null files/digests and zero counts")
    elif sensitivity.get("status") == "passed":
        digest = sensitivity.get("sha256")
        if (
            not isinstance(digest, str)
            or LOWER_SHA256.fullmatch(digest) is None
            or digest == "0" * 64
        ):
            problem("passed sensitivity evidence requires a nonzero lowercase SHA-256")
        else:
            try:
                evidence_path = _drivaerml_release_path(
                    sensitivity.get("file"),
                    base=dataset_directory,
                    repository_root=root,
                    label="sensitivity_evidence.file",
                )
                evidence, evidence_digest = _load_drivaerml_release_json(
                    evidence_path,
                    label="sensitivity evidence",
                )
            except SubmissionJSONError as error:
                problem(str(error))
            else:
                if evidence_digest != digest:
                    problem("sensitivity evidence SHA-256 does not match its retained bytes")
                sensitivity_review_artifacts.append(
                    ("sensitivity evidence", evidence_path, digest)
                )
                evidence_fields = {
                    "schema",
                    "schema_version",
                    "status",
                    "dataset_revision",
                    "evaluator",
                    "case_scope",
                    "model_checkpoint_count",
                    "checkpoints",
                    "bootstrap_replicate_count",
                    "paired_bootstrap",
                    "conclusion",
                }
                if not isinstance(evidence, dict) or set(evidence) != evidence_fields:
                    problem(
                        "sensitivity evidence must implement the complete v1 schema "
                        f"with exactly {sorted(evidence_fields)}"
                    )
                    evidence = {}
                if evidence.get("schema") != DRIVAERML_RELATIVE_SENSITIVITY_SCHEMA:
                    problem("sensitivity evidence has the wrong schema")
                if evidence.get("schema_version") != 1:
                    problem("sensitivity evidence schema_version must equal 1")
                if evidence.get("status") != "passed":
                    problem("sensitivity evidence status must equal passed")
                if source_revision is None or evidence.get("dataset_revision") != source_revision:
                    problem(
                        "sensitivity evidence dataset_revision must equal the retained "
                        "official dataset revision"
                    )

                evidence_evaluator = evidence.get("evaluator")
                expected_evidence_evaluator = {
                    "repository": evaluator.get("repository"),
                    "git_revision": evaluator_revision,
                    "reference_version": evaluator.get("reference_version"),
                }
                if evidence_evaluator != expected_evidence_evaluator:
                    problem(
                        "sensitivity evidence evaluator must exactly equal the frozen "
                        "release evaluator"
                    )

                case_scope = evidence.get("case_scope")
                case_scope_fields = {
                    "case_count",
                    "case_ids",
                    "case_ids_sha256",
                }
                sensitivity_case_ids: list[str] = []
                if not isinstance(case_scope, dict) or set(case_scope) != case_scope_fields:
                    problem(
                        "sensitivity evidence case_scope must contain exactly "
                        f"{sorted(case_scope_fields)}"
                    )
                    case_scope = {}
                raw_sensitivity_case_ids = case_scope.get("case_ids")
                if not isinstance(raw_sensitivity_case_ids, list):
                    problem("sensitivity evidence case_scope.case_ids must be an array")
                else:
                    for position, case_id in enumerate(raw_sensitivity_case_ids):
                        if not isinstance(case_id, str):
                            problem(
                                "sensitivity evidence case_scope.case_ids contains an "
                                f"invalid value at position {position}"
                            )
                            continue
                        sensitivity_case_ids.append(case_id)
                    if len(sensitivity_case_ids) < 2:
                        problem(
                            "sensitivity evidence must use at least two official cases"
                        )
                    if len(sensitivity_case_ids) != len(set(sensitivity_case_ids)):
                        problem("sensitivity evidence case scope contains duplicate cases")
                    if official_case_ids:
                        sensitivity_case_set = set(sensitivity_case_ids)
                        expected_order = [
                            case_id
                            for case_id in official_case_ids
                            if case_id in sensitivity_case_set
                        ]
                        if sensitivity_case_ids != expected_order:
                            problem(
                                "sensitivity evidence case scope must be an ordered unique "
                                "subset of the official registry"
                            )
                if case_scope.get("case_count") != len(sensitivity_case_ids):
                    problem(
                        "sensitivity evidence case_scope.case_count must equal its "
                        "case_ids length"
                    )
                if case_scope.get("case_ids_sha256") != canonical_json_sha256(
                    sensitivity_case_ids
                ):
                    problem(
                        "sensitivity evidence case_scope.case_ids_sha256 must bind "
                        "the ordered case IDs"
                    )

                checkpoints = evidence.get("checkpoints")
                checkpoint_ids: list[str] = []
                checkpoint_hashes: list[str] = []
                training_run_ids: list[str] = []
                checkpoint_locators: list[tuple[str, str, str]] = []
                checkpoint_provenance_hashes: list[str] = []
                prediction_manifest_hashes: list[str] = []
                prediction_identity_sets: list[tuple[tuple[str, str], ...]] = []
                checkpoint_fields = {
                    "checkpoint_id",
                    "model_family",
                    "training_run_id",
                    "artifact_kind",
                    "checkpoint_repository",
                    "checkpoint_revision",
                    "checkpoint_path",
                    "checkpoint_sha256",
                    "checkpoint_provenance_file",
                    "checkpoint_provenance_sha256",
                    "prediction_manifest_file",
                    "prediction_manifest_sha256",
                }
                if not isinstance(checkpoints, list) or len(checkpoints) < 3:
                    problem(
                        "sensitivity evidence must describe at least three trained checkpoints"
                    )
                    checkpoints = []
                for position, checkpoint in enumerate(checkpoints):
                    label = f"sensitivity evidence checkpoints[{position}]"
                    if not isinstance(checkpoint, dict) or set(checkpoint) != checkpoint_fields:
                        problem(
                            f"{label} must contain exactly {sorted(checkpoint_fields)}"
                        )
                        continue
                    checkpoint_id = checkpoint.get("checkpoint_id")
                    model_family = checkpoint.get("model_family")
                    training_run_id = checkpoint.get("training_run_id")
                    if (
                        not isinstance(checkpoint_id, str)
                        or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,159}", checkpoint_id)
                        is None
                    ):
                        problem(f"{label}.checkpoint_id is invalid")
                    else:
                        checkpoint_ids.append(checkpoint_id)
                    if not isinstance(model_family, str) or not model_family.strip():
                        problem(f"{label}.model_family must be non-empty")
                    if (
                        not isinstance(training_run_id, str)
                        or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/-]{1,199}", training_run_id)
                        is None
                    ):
                        problem(f"{label}.training_run_id is invalid")
                    elif _drivaerml_text_looks_placeholder(training_run_id):
                        problem(f"{label}.training_run_id looks placeholder-shaped")
                    else:
                        training_run_ids.append(training_run_id)
                    if checkpoint.get("artifact_kind") != "trained_model_checkpoint":
                        problem(
                            f"{label}.artifact_kind must equal trained_model_checkpoint"
                        )
                    checkpoint_repository = checkpoint.get("checkpoint_repository")
                    parsed_repository = (
                        urlparse(checkpoint_repository)
                        if isinstance(checkpoint_repository, str)
                        else None
                    )
                    if (
                        parsed_repository is None
                        or parsed_repository.scheme != "https"
                        or not parsed_repository.netloc
                    ):
                        problem(f"{label}.checkpoint_repository must be an HTTPS repository")
                    elif (
                        parsed_repository.hostname is None
                        or parsed_repository.hostname.lower()
                        in {"example.com", "example.net", "example.org", "localhost"}
                        or parsed_repository.hostname.lower().endswith(
                            (".example", ".invalid", ".localhost", ".test")
                        )
                        or _drivaerml_text_looks_placeholder(parsed_repository.path)
                    ):
                        problem(
                            f"{label}.checkpoint_repository looks placeholder-shaped"
                        )
                    checkpoint_revision = checkpoint.get("checkpoint_revision")
                    if (
                        not isinstance(checkpoint_revision, str)
                        or re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", checkpoint_revision)
                        is None
                    ):
                        problem(f"{label}.checkpoint_revision must be immutable")
                    elif _drivaerml_hex_looks_placeholder(checkpoint_revision):
                        problem(f"{label}.checkpoint_revision looks placeholder-shaped")
                    checkpoint_path = checkpoint.get("checkpoint_path")
                    if (
                        not isinstance(checkpoint_path, str)
                        or not checkpoint_path
                        or Path(checkpoint_path).is_absolute()
                        or ".." in Path(checkpoint_path).parts
                    ):
                        problem(f"{label}.checkpoint_path must be a safe relative path")
                    elif _drivaerml_text_looks_placeholder(checkpoint_path):
                        problem(f"{label}.checkpoint_path looks placeholder-shaped")
                    if (
                        isinstance(checkpoint_repository, str)
                        and isinstance(checkpoint_revision, str)
                        and isinstance(checkpoint_path, str)
                    ):
                        checkpoint_locators.append(
                            (
                                checkpoint_repository.rstrip("/"),
                                checkpoint_revision,
                                checkpoint_path,
                            )
                        )
                    for field, collected in (
                        ("checkpoint_sha256", checkpoint_hashes),
                        ("prediction_manifest_sha256", prediction_manifest_hashes),
                    ):
                        value = checkpoint.get(field)
                        if (
                            not isinstance(value, str)
                            or LOWER_SHA256.fullmatch(value) is None
                        ):
                            problem(f"{label}.{field} must be a nonzero SHA-256")
                        elif _drivaerml_hex_looks_placeholder(value):
                            problem(f"{label}.{field} looks placeholder-shaped")
                        else:
                            collected.append(value)

                    provenance_digest = checkpoint.get(
                        "checkpoint_provenance_sha256"
                    )
                    if (
                        not isinstance(provenance_digest, str)
                        or LOWER_SHA256.fullmatch(provenance_digest) is None
                    ):
                        problem(
                            f"{label}.checkpoint_provenance_sha256 must be a "
                            "nonzero SHA-256"
                        )
                    elif _drivaerml_hex_looks_placeholder(provenance_digest):
                        problem(
                            f"{label}.checkpoint_provenance_sha256 looks "
                            "placeholder-shaped"
                        )
                    else:
                        checkpoint_provenance_hashes.append(provenance_digest)
                    try:
                        provenance_path = _drivaerml_release_path(
                            checkpoint.get("checkpoint_provenance_file"),
                            base=dataset_directory,
                            repository_root=root,
                            label=f"{label}.checkpoint_provenance_file",
                            confined_to_base=True,
                        )
                        provenance, actual_provenance_digest = (
                            _load_drivaerml_release_json(
                                provenance_path,
                                label=f"{label} checkpoint provenance",
                            )
                        )
                    except SubmissionJSONError as error:
                        problem(str(error))
                    else:
                        if actual_provenance_digest != provenance_digest:
                            problem(
                                f"{label}.checkpoint_provenance_sha256 does not "
                                "match the retained bytes"
                            )
                        if isinstance(provenance_digest, str):
                            sensitivity_review_artifacts.append(
                                (
                                    f"{label} checkpoint provenance",
                                    provenance_path,
                                    provenance_digest,
                                )
                            )
                        provenance_fields = {
                            "schema",
                            "schema_version",
                            "dataset_id",
                            "dataset_revision",
                            "artifact_kind",
                            "checkpoint_id",
                            "model_family",
                            "training_run_id",
                            "checkpoint_repository",
                            "checkpoint_revision",
                            "checkpoint_path",
                            "checkpoint_sha256",
                            "training_completed",
                            "training_case_count",
                            "optimizer_step_count",
                        }
                        if (
                            not isinstance(provenance, dict)
                            or set(provenance) != provenance_fields
                        ):
                            problem(
                                f"{label} checkpoint provenance must contain exactly "
                                f"{sorted(provenance_fields)}"
                            )
                            provenance = {}
                        expected_provenance = {
                            "schema": DRIVAERML_TRAINED_CHECKPOINT_PROVENANCE_SCHEMA,
                            "schema_version": 1,
                            "dataset_id": "drivaerml",
                            "dataset_revision": source_revision,
                            "artifact_kind": "trained_model_checkpoint",
                            "checkpoint_id": checkpoint_id,
                            "model_family": model_family,
                            "training_run_id": training_run_id,
                            "checkpoint_repository": checkpoint_repository,
                            "checkpoint_revision": checkpoint_revision,
                            "checkpoint_path": checkpoint_path,
                            "checkpoint_sha256": checkpoint.get(
                                "checkpoint_sha256"
                            ),
                        }
                        for field, expected_value in expected_provenance.items():
                            if provenance.get(field) != expected_value:
                                problem(
                                    f"{label} checkpoint provenance {field} differs "
                                    "from the sensitivity binding"
                                )
                        if provenance.get("training_completed") is not True:
                            problem(
                                f"{label} checkpoint provenance must attest completed "
                                "training"
                            )
                        for field in ("training_case_count", "optimizer_step_count"):
                            count = provenance.get(field)
                            if (
                                not isinstance(count, int)
                                or isinstance(count, bool)
                                or count < 1
                            ):
                                problem(
                                    f"{label} checkpoint provenance {field} must be "
                                    "a positive integer"
                                )

                    prediction_digest = checkpoint.get(
                        "prediction_manifest_sha256"
                    )
                    try:
                        prediction_path = _drivaerml_release_path(
                            checkpoint.get("prediction_manifest_file"),
                            base=dataset_directory,
                            repository_root=root,
                            label=f"{label}.prediction_manifest_file",
                            confined_to_base=True,
                        )
                        prediction_manifest, actual_prediction_digest = (
                            _load_drivaerml_release_json(
                                prediction_path,
                                label=f"{label} prediction manifest",
                            )
                        )
                    except SubmissionJSONError as error:
                        problem(str(error))
                    else:
                        if actual_prediction_digest != prediction_digest:
                            problem(
                                f"{label}.prediction_manifest_sha256 does not match "
                                "the retained bytes"
                            )
                        if isinstance(prediction_digest, str):
                            sensitivity_review_artifacts.append(
                                (
                                    f"{label} prediction manifest",
                                    prediction_path,
                                    prediction_digest,
                                )
                            )
                        prediction_fields = {
                            "schema",
                            "schema_version",
                            "dataset_id",
                            "dataset_revision",
                            "evaluator",
                            "checkpoint_id",
                            "checkpoint_sha256",
                            "case_scope",
                            "cases",
                        }
                        if (
                            not isinstance(prediction_manifest, dict)
                            or set(prediction_manifest) != prediction_fields
                        ):
                            problem(
                                f"{label} prediction manifest must contain exactly "
                                f"{sorted(prediction_fields)}"
                            )
                            prediction_manifest = {}
                        expected_prediction_fields = {
                            "schema": (
                                DRIVAERML_SENSITIVITY_PREDICTION_MANIFEST_SCHEMA
                            ),
                            "schema_version": 1,
                            "dataset_id": "drivaerml",
                            "dataset_revision": source_revision,
                            "evaluator": expected_evidence_evaluator,
                            "checkpoint_id": checkpoint_id,
                            "checkpoint_sha256": checkpoint.get(
                                "checkpoint_sha256"
                            ),
                            "case_scope": case_scope,
                        }
                        for field, expected_value in expected_prediction_fields.items():
                            if prediction_manifest.get(field) != expected_value:
                                problem(
                                    f"{label} prediction manifest {field} differs "
                                    "from the sensitivity binding"
                                )
                        prediction_cases = prediction_manifest.get("cases")
                        if not isinstance(prediction_cases, list):
                            problem(f"{label} prediction manifest cases must be an array")
                            prediction_cases = []
                        observed_prediction_cases: list[str] = []
                        observed_prediction_identities: list[tuple[str, str]] = []
                        prediction_case_fields = {
                            "case_id",
                            "surface_prediction_sha256",
                            "volume_prediction_sha256",
                        }
                        for case_position, prediction_case in enumerate(
                            prediction_cases
                        ):
                            case_label = (
                                f"{label} prediction manifest cases[{case_position}]"
                            )
                            if (
                                not isinstance(prediction_case, dict)
                                or set(prediction_case) != prediction_case_fields
                            ):
                                problem(
                                    f"{case_label} must contain exactly "
                                    f"{sorted(prediction_case_fields)}"
                                )
                                continue
                            observed_prediction_cases.append(
                                prediction_case.get("case_id")
                            )
                            case_prediction_identities: list[str] = []
                            for field in (
                                "surface_prediction_sha256",
                                "volume_prediction_sha256",
                            ):
                                value = prediction_case.get(field)
                                if (
                                    not isinstance(value, str)
                                    or LOWER_SHA256.fullmatch(value) is None
                                ):
                                    problem(f"{case_label}.{field} is invalid")
                                elif _drivaerml_hex_looks_placeholder(value):
                                    problem(
                                        f"{case_label}.{field} looks "
                                        "placeholder-shaped"
                                    )
                                else:
                                    case_prediction_identities.append(value)
                            if len(case_prediction_identities) == 2:
                                observed_prediction_identities.append(
                                    (
                                        case_prediction_identities[0],
                                        case_prediction_identities[1],
                                    )
                                )
                        if observed_prediction_cases != sensitivity_case_ids:
                            problem(
                                f"{label} prediction manifest cases must exactly "
                                "match the sensitivity case scope"
                            )
                        elif len(observed_prediction_identities) == len(
                            sensitivity_case_ids
                        ):
                            prediction_identity_sets.append(
                                tuple(observed_prediction_identities)
                            )
                for values, label in (
                    (checkpoint_ids, "checkpoint IDs"),
                    (checkpoint_hashes, "checkpoint hashes"),
                    (training_run_ids, "training run IDs"),
                    (checkpoint_locators, "immutable checkpoint locators"),
                    (checkpoint_provenance_hashes, "checkpoint-provenance hashes"),
                    (prediction_manifest_hashes, "prediction-manifest hashes"),
                    (prediction_identity_sets, "prediction artifact identity sets"),
                ):
                    if len(values) != len(set(values)):
                        problem(f"sensitivity evidence {label} must be distinct")
                if evidence.get("model_checkpoint_count") != len(checkpoints):
                    problem(
                        "sensitivity evidence model_checkpoint_count must equal its "
                        "checkpoint array length"
                    )
                if evidence.get("model_checkpoint_count") != sensitivity.get(
                    "model_checkpoint_count"
                ):
                    problem(
                        "sensitivity evidence model_checkpoint_count differs from "
                        "the release record"
                    )

                bootstrap = evidence.get("paired_bootstrap")
                bootstrap_fields = {
                    "paired",
                    "replicate_count",
                    "random_seed",
                    "resampling_unit",
                    "comparisons",
                }
                if not isinstance(bootstrap, dict) or set(bootstrap) != bootstrap_fields:
                    problem(
                        "sensitivity evidence paired_bootstrap must contain exactly "
                        f"{sorted(bootstrap_fields)}"
                    )
                    bootstrap = {}
                if bootstrap.get("paired") is not True:
                    problem("sensitivity bootstrap must be paired")
                if bootstrap.get("replicate_count") != 10000:
                    problem("sensitivity paired bootstrap must use exactly 10000 replicates")
                random_seed = bootstrap.get("random_seed")
                if (
                    not isinstance(random_seed, int)
                    or isinstance(random_seed, bool)
                    or random_seed < 0
                ):
                    problem("sensitivity paired bootstrap random_seed must be non-negative")
                if bootstrap.get("resampling_unit") != "official_case_id":
                    problem(
                        "sensitivity paired bootstrap resampling_unit must equal "
                        "official_case_id"
                    )
                comparison_fields = {
                    "left_checkpoint_id",
                    "right_checkpoint_id",
                    "metric_id",
                    "observed_delta",
                    "confidence_interval_95",
                    "replicate_count",
                }
                comparisons = bootstrap.get("comparisons")
                compared_checkpoint_ids: set[str] = set()
                if not isinstance(comparisons, list) or not comparisons:
                    problem("sensitivity paired bootstrap comparisons must be non-empty")
                    comparisons = []
                for position, comparison in enumerate(comparisons):
                    label = f"sensitivity paired_bootstrap.comparisons[{position}]"
                    if not isinstance(comparison, dict) or set(comparison) != comparison_fields:
                        problem(f"{label} must contain exactly {sorted(comparison_fields)}")
                        continue
                    left = comparison.get("left_checkpoint_id")
                    right = comparison.get("right_checkpoint_id")
                    if left not in checkpoint_ids or right not in checkpoint_ids or left == right:
                        problem(f"{label} must reference two distinct retained checkpoints")
                    else:
                        compared_checkpoint_ids.update((left, right))
                    if (
                        not isinstance(comparison.get("metric_id"), str)
                        or not comparison["metric_id"].strip()
                    ):
                        problem(f"{label}.metric_id must be non-empty")
                    if not is_number(comparison.get("observed_delta")):
                        problem(f"{label}.observed_delta must be finite")
                    interval = comparison.get("confidence_interval_95")
                    if (
                        not isinstance(interval, list)
                        or len(interval) != 2
                        or not all(is_number(value) for value in interval)
                        or interval[0] > interval[1]
                    ):
                        problem(f"{label}.confidence_interval_95 is invalid")
                    if comparison.get("replicate_count") != 10000:
                        problem(f"{label}.replicate_count must equal 10000")
                if checkpoint_ids and compared_checkpoint_ids != set(checkpoint_ids):
                    problem(
                        "sensitivity paired bootstrap comparisons must cover every checkpoint"
                    )
                if evidence.get("bootstrap_replicate_count") != 10000:
                    problem(
                        "sensitivity evidence bootstrap_replicate_count must equal 10000"
                    )
                if evidence.get("bootstrap_replicate_count") != sensitivity.get(
                    "bootstrap_replicate_count"
                ):
                    problem(
                        "sensitivity evidence bootstrap count differs from the release record"
                    )

                conclusion = evidence.get("conclusion")
                conclusion_fields = {"outcome", "summary"}
                if not isinstance(conclusion, dict) or set(conclusion) != conclusion_fields:
                    problem(
                        "sensitivity evidence conclusion must contain exactly outcome and summary"
                    )
                else:
                    if conclusion.get("outcome") != "passed":
                        problem("sensitivity evidence conclusion outcome must equal passed")
                    if (
                        not isinstance(conclusion.get("summary"), str)
                        or not conclusion["summary"].strip()
                    ):
                        problem("sensitivity evidence conclusion summary must be non-empty")
        if (
            not isinstance(sensitivity.get("model_checkpoint_count"), int)
            or isinstance(sensitivity.get("model_checkpoint_count"), bool)
            or sensitivity["model_checkpoint_count"] < 3
            or not isinstance(sensitivity.get("bootstrap_replicate_count"), int)
            or isinstance(sensitivity.get("bootstrap_replicate_count"), bool)
            or sensitivity["bootstrap_replicate_count"] != 10000
        ):
            problem(
                "passed sensitivity evidence requires >=3 checkpoints and exactly "
                "10000 paired bootstrap replicates"
            )
    sensitivity_is_valid_and_passed = (
        sensitivity.get("status") == "passed"
        and len(errors) == sensitivity_error_count
    )

    approval_error_count = len(errors)
    approval = record.get("owner_approval")
    if not isinstance(approval, dict):
        problem("record owner_approval must be an object")
        approval = {}
    if approval.get("status") not in {"pending", "approved"}:
        problem("record owner_approval status must be pending or approved")
    if approval.get("status") == "pending":
        approval_fields = {"status", "approved_by", "approved_at", "pull_request_url"}
        if set(approval) != approval_fields:
            problem(
                "pending owner_approval must contain exactly "
                f"{sorted(approval_fields)}"
            )
        if any(
            approval.get(field) is not None
            for field in ("approved_by", "approved_at", "pull_request_url")
        ):
            problem("pending owner approval metadata must be null")
    elif approval.get("status") == "approved":
        approval_fields = {"status", "approved_by", "approved_at", "approval_record"}
        if set(approval) != approval_fields:
            problem(
                "approved owner_approval must bind a committed approval record and "
                f"contain exactly {sorted(approval_fields)}"
            )
        for field in ("approved_by", "approved_at"):
            if not isinstance(approval.get(field), str) or not approval[field].strip():
                problem(f"approved owner_approval.{field} must be a non-empty string")
        if approval.get("approved_by") not in DRIVAERML_RELATIVE_OWNER_APPROVERS:
            problem(
                "owner_approval.approved_by is not an independently pinned "
                "benchmark owner"
            )
        try:
            date.fromisoformat(str(approval.get("approved_at")))
        except ValueError:
            problem("owner_approval.approved_at must be an ISO date")
        approval_binding = approval.get("approval_record")
        approval_binding_fields = {"file", "sha256", "schema", "git_revision"}
        if (
            not isinstance(approval_binding, dict)
            or set(approval_binding) != approval_binding_fields
        ):
            problem(
                "owner_approval.approval_record must contain exactly "
                f"{sorted(approval_binding_fields)}"
            )
            approval_binding = {}
        if approval_binding.get("schema") != DRIVAERML_RELATIVE_OWNER_APPROVAL_SCHEMA:
            problem("owner approval record binding has the wrong schema")
        approval_digest = approval_binding.get("sha256")
        if (
            not isinstance(approval_digest, str)
            or LOWER_SHA256.fullmatch(approval_digest) is None
            or approval_digest == "0" * 64
        ):
            problem("owner approval record binding requires a nonzero SHA-256")
        approval_revision = approval_binding.get("git_revision")
        if (
            not isinstance(approval_revision, str)
            or LOWER_GIT_SHA1.fullmatch(approval_revision) is None
            or approval_revision == "0" * 40
            or not _drivaerml_git_commit_is_bound(root, approval_revision)
        ):
            problem(
                "owner approval record git_revision must be a real commit "
                "reachable in the local repository"
            )
        else:
            if not _drivaerml_git_commit_is_on_trusted_dev(
                root,
                approval_revision,
            ):
                problem(
                    "owner approval record git_revision must already be retained "
                    "on fetched origin/dev from the evaluator repository"
                )
            if (
                not isinstance(evaluator_revision, str)
                or LOWER_GIT_SHA1.fullmatch(evaluator_revision) is None
                or evaluator_revision == approval_revision
                or not _drivaerml_git_commit_is_ancestor(
                    root,
                    evaluator_revision,
                    approval_revision,
                )
            ):
                problem(
                    "owner approval record must be committed after the frozen "
                    "evaluator revision"
                )
        try:
            approval_path = _drivaerml_release_path(
                approval_binding.get("file"),
                base=dataset_directory,
                repository_root=root,
                label="owner_approval.approval_record.file",
                confined_to_base=True,
            )
            approval_document, actual_approval_digest = _load_drivaerml_release_json(
                approval_path,
                label="owner approval record",
            )
        except SubmissionJSONError as error:
            problem(str(error))
        else:
            if actual_approval_digest != approval_digest:
                problem("owner approval record SHA-256 does not match its retained bytes")
            if (
                isinstance(approval_revision, str)
                and LOWER_GIT_SHA1.fullmatch(approval_revision) is not None
                and _drivaerml_git_blob_sha256(root, approval_revision, approval_path)
                != actual_approval_digest
            ):
                problem(
                    "owner approval record bytes are not retained by the bound Git commit"
                )
            if (
                isinstance(approval_revision, str)
                and LOWER_GIT_SHA1.fullmatch(approval_revision) is not None
            ):
                for artifact_label, artifact_path, artifact_digest in (
                    sensitivity_review_artifacts
                ):
                    if (
                        _drivaerml_git_blob_sha256(
                            root,
                            approval_revision,
                            artifact_path,
                        )
                        != artifact_digest
                    ):
                        problem(
                            f"{artifact_label} bytes were not retained by the "
                            "owner approval commit"
                        )
            approval_document_fields = {
                "schema",
                "schema_version",
                "activation_release_schema",
                "activation_release_schema_version",
                "dataset_id",
                "release_id",
                "status",
                "approved_by",
                "approved_at",
                "contract_sha256",
                "evaluator_git_revision",
                "release_bindings_sha256",
                "sensitivity_evidence_sha256",
            }
            if (
                not isinstance(approval_document, dict)
                or set(approval_document) != approval_document_fields
            ):
                problem(
                    "owner approval record must implement the complete v1 schema "
                    f"with exactly {sorted(approval_document_fields)}"
                )
                approval_document = {}
            expected_approval_values = {
                "schema": DRIVAERML_RELATIVE_OWNER_APPROVAL_SCHEMA,
                "schema_version": 1,
                "activation_release_schema": (
                    DRIVAERML_RELATIVE_ACTIVATION_RECORD_SCHEMA
                ),
                "activation_release_schema_version": 1,
                "dataset_id": "drivaerml",
                "release_id": release_id,
                "status": "approved",
                "approved_by": approval.get("approved_by"),
                "approved_at": approval.get("approved_at"),
                "contract_sha256": bindings.get("contract", {}).get("sha256"),
                "evaluator_git_revision": evaluator_revision,
                "release_bindings_sha256": canonical_json_sha256(bindings),
                "sensitivity_evidence_sha256": sensitivity.get("sha256"),
            }
            if approval_document != expected_approval_values:
                problem(
                    "owner approval record does not exactly bind this release, "
                    "contract, evaluator, and sensitivity evidence"
                )
    owner_is_valid_and_approved = (
        approval.get("status") == "approved"
        and len(errors) == approval_error_count
    )

    gates = record.get("activation_gates")
    expected_gate_keys = set(DRIVAERML_RELATIVE_ACTIVATION_GATE_IDS)
    if not isinstance(gates, dict) or set(gates) != expected_gate_keys:
        problem(f"record activation_gates must be exactly {sorted(expected_gate_keys)}")
        gates = {}
    elif any(not isinstance(value, bool) for value in gates.values()):
        problem("record activation_gates values must be booleans")

    contract_gates = contract.get("activation_gates")
    if (
        not isinstance(contract_gates, dict)
        or set(contract_gates) != expected_gate_keys
        or any(not isinstance(value, bool) for value in contract_gates.values())
    ):
        problem("the retained contract activation_gates are invalid")
        contract_gates = {}
    if gates and contract_gates and gates != contract_gates:
        problem("record activation_gates must exactly equal the retained contract gates")

    support_gate_names = {
        "all_484_velocity_placement_manifest_bound",
        "all_484_velocity_mapping_manifest_bound",
        "all_484_cp_manifest_bound",
    }
    if gates and any(gates.get(gate) is not True for gate in support_gate_names):
        problem("a support-verified release record must bind all three 484-case manifests")
    expected_state_gates = {
        "immutable_evaluator_revision_bound": evaluator_is_valid,
        "genuine_model_sensitivity_review_complete": sensitivity_is_valid_and_passed,
        "owner_scientific_approval": owner_is_valid_and_approved,
    }
    if gates:
        for gate, expected in expected_state_gates.items():
            if gates.get(gate) is not expected:
                problem(f"activation gate {gate} disagrees with its validated release state")

    record_status = record.get("status")
    if record_status == "support_verified_activation_pending":
        if declaration.get("status") != "support_verified_activation_pending":
            problem(
                "a pending release record requires the benchmark declaration "
                "status support_verified_activation_pending"
            )
        if declaration.get("profile_format_enabled") is not False:
            problem(
                "a pending release record requires profile_format_enabled=false"
            )
        if record.get("profile_format_authorized") is not False:
            problem("a pending release record must not authorize the relative format")
        if sensitivity.get("status") != "pending":
            problem("a pending release record must retain pending sensitivity evidence")
        if approval.get("status") != "pending":
            problem("a pending release record must retain pending owner approval")
    elif record_status == "activated":
        if declaration.get("status") != "activated":
            problem("an active release record requires declaration status activated")
        if declaration.get("profile_format_enabled") is not True:
            problem("an active release record requires profile_format_enabled=true")
        if record.get("profile_format_authorized") is not True:
            problem("record does not authorize the relative profile format")
        if not gates or any(gates.get(gate) is not True for gate in expected_gate_keys):
            problem("every activation gate must be true")
        if not contract_gates or any(
            contract_gates.get(gate) is not True for gate in expected_gate_keys
        ):
            problem("the retained contract does not mark every activation gate complete")
        if not sensitivity_is_valid_and_passed:
            problem("genuine-model sensitivity evidence is not passed")
        if not owner_is_valid_and_approved:
            problem("owner scientific approval is not approved")

    if require_active and record_status != "activated":
        problem("record status is not activated")

    for message in errors:
        add(message)
    return not errors


def safe_submission_path(
    add: Any,
    directory: Path,
    filename: Any,
    *,
    label: str,
) -> Path | None:
    """Resolve a declared package file without allowing absolute paths or traversal."""

    if not isinstance(filename, str) or not filename:
        add(f"{label} must be a non-empty relative file path")
        return None
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        add(f"{label} must stay inside the submission directory")
        return None
    candidate = directory / relative
    try:
        if not candidate.resolve().is_relative_to(directory.resolve()):
            add(f"{label} must stay inside the submission directory")
            return None
    except OSError as error:
        add(f"cannot resolve {label}: {error}")
        return None
    return candidate


def load_json_lines(add: Any, path: Path, *, label: str) -> list[dict[str, Any]]:
    """Load strict UTF-8 JSON Lines while retaining useful source line errors."""

    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    add(f"{label} line {line_number} must not be blank")
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    add(f"{label} line {line_number} is not valid JSON: {error.msg}")
                    continue
                if not isinstance(value, dict):
                    add(f"{label} line {line_number} must be a JSON object")
                    continue
                records.append(value)
    except (OSError, UnicodeError) as error:
        add(f"cannot read {label}: {error}")
    return records


def validate_domain_semantics(add: Any, domain: Any, *, label: str) -> None:
    if not isinstance(domain, dict) or domain.get("kind") != "axis_aligned_box":
        return
    minimum = domain.get("minimum")
    maximum = domain.get("maximum")
    if (
        not isinstance(minimum, list)
        or not isinstance(maximum, list)
        or len(minimum) != len(maximum)
        or not minimum
    ):
        add(f"{label} axis-aligned bounding-box dimensions must match")
        return
    if any(
        not is_number(lower) or not is_number(upper) or lower >= upper
        for lower, upper in zip(minimum, maximum)
    ):
        add(f"{label} axis-aligned bounding-box minimum must be less than maximum")


def submission_files(paths: list[Path] | None = None) -> list[Path]:
    if not paths:
        return sorted(path.resolve() for path in (ROOT / "submissions").glob("*/*/submission.json"))
    result: list[Path] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if path.is_dir():
            candidate = path / "submission.json"
            if candidate.exists():
                result.append(candidate)
            else:
                result.extend(sorted(path.glob("*/submission.json")))
        else:
            result.append(path)
    return sorted(set(result))


def split_entry(dataset: dict[str, Any], split_id: str) -> dict[str, Any] | None:
    return next((split for split in dataset.get("splits", []) if split.get("id") == split_id), None)


def validate_metadata(
    add: Any,
    path: Path,
    submission: dict[str, Any],
    dataset: dict[str, Any],
    split: dict[str, Any],
    *,
    require_registered_path: bool = True,
) -> None:
    expected_parent = ROOT / "submissions" / dataset["slug"] / submission["submission_id"]
    if require_registered_path and path.parent.resolve() != expected_parent.resolve():
        add(f"submission.json must be stored at {expected_parent.relative_to(ROOT)}/submission.json")
    if submission.get("dataset") != dataset["name"]:
        add(f"dataset must be {dataset['name']!r} for dataset_id {dataset['slug']!r}")
    if submission.get("split") != split["name"]:
        add(f"split must be {split['name']!r} for split_id {split['id']!r}")
    if submission.get("model_type") not in submission.get("model_types", []):
        add("model_type must also appear in model_types")

    regime = submission.get("training_regime")
    target = submission.get("target_data_used")
    pretrained = submission.get("external_pretraining")
    pretraining_data = submission.get("pretraining_data", [])
    if regime == "from_scratch":
        if pretrained or pretraining_data:
            add("from_scratch requires external_pretraining=false and an empty pretraining_data list")
        if target != "official_train":
            add("from_scratch requires target_data_used=official_train")
    elif regime == "pretrained_zero_shot":
        if not pretrained or not pretraining_data:
            add("pretrained_zero_shot requires external_pretraining=true and named pretraining_data")
        if target != "none":
            add("pretrained_zero_shot requires target_data_used=none")
    elif regime == "pretrained_official_train":
        if not pretrained or not pretraining_data:
            add("pretrained_official_train requires external_pretraining=true and named pretraining_data")
        if target != "official_train":
            add("pretrained_official_train requires target_data_used=official_train")
    elif regime == "other" and not submission.get("training_regime_explanation", "").strip():
        add("training_regime=other requires training_regime_explanation")

    for key in ("paper_url", "code_url"):
        value = submission.get(key, "")
        if value and (not isinstance(value, str) or urlparse(value).scheme not in {"http", "https"}):
            add(f"{key} must be empty or use http/https")
    try:
        date.fromisoformat(submission.get("submitted_at", ""))
    except (TypeError, ValueError):
        add("submitted_at must be an ISO date in YYYY-MM-DD form")


def validate_metrics(add: Any, submission: dict[str, Any], dataset: dict[str, Any], manifest: dict[str, Any]) -> None:
    values = submission.get("metric_values", {})
    expected_ids = set(dataset.get("metric_ids", []))
    fixed_zero_components: set[str] = set()
    surface_only = (
        submission.get("dataset_id") == "drivaerml"
        and submission.get("prediction_scope") == "surface_only"
    )
    if surface_only:
        policy = dataset.get("overall_score_composite", {}).get("surface_only_policy", {})
        unavailable = policy.get("unavailable_component_metric_ids")
        if (
            not isinstance(unavailable, list)
            or set(unavailable)
            != {
                "volume_velocity_rel_l2",
                "volume_pressure_rel_l2",
                "velocity_profile_r2",
            }
            or policy.get("unavailable_component_score") != 0.0
            or policy.get("component_weight_renormalization") is not False
            or policy.get("maximum_overall_score") != 60.0
        ):
            add("DrivAerML surface-only scoring policy is malformed")
        else:
            fixed_zero_components = set(unavailable)
    actual_ids = set(values)
    missing = sorted((expected_ids - fixed_zero_components) - actual_ids)
    unknown = sorted(actual_ids - expected_ids)
    if missing:
        add(f"metric_values is missing: {', '.join(missing)}")
    if unknown:
        add(f"metric_values contains unknown metrics: {', '.join(unknown)}")
    if surface_only:
        fabricated = sorted(actual_ids & fixed_zero_components)
        if fabricated:
            add(
                "surface_only metric_values must omit unavailable raw metrics: "
                + ", ".join(fabricated)
            )
    definitions = {definition["id"]: definition for definition in manifest["metric_definitions"]}
    composite = dataset.get("overall_score_composite")
    component_groups = dataset.get("component_score_groups")
    negative_score_ids: set[str] = set()
    if isinstance(composite, dict) and composite.get("allow_negative_scores") is True:
        target_metric_id = composite.get("metric_id")
        if isinstance(target_metric_id, str):
            negative_score_ids.add(target_metric_id)
        if isinstance(component_groups, dict):
            negative_score_ids.update(
                group.get("metric_id")
                for group in component_groups.get("groups", [])
                if isinstance(group, dict) and isinstance(group.get("metric_id"), str)
            )
    for metric_id, value in values.items():
        if not is_number(value):
            add(f"metric_values.{metric_id} must be a finite number")
            continue
        definition = definitions.get(metric_id, {})
        if (
            definition.get("kind") in {"error", "score"}
            and value < 0
            and metric_id not in negative_score_ids
        ):
            add(f"metric_values.{metric_id} cannot be negative")
        if definition.get("kind") == "score" and value > 100:
            add(f"metric_values.{metric_id} cannot exceed 100")
        if definition.get("kind") == "r2" and value > 1:
            add(f"metric_values.{metric_id} cannot exceed 1")

    aggregate = dataset.get("metric_aggregate")
    if isinstance(aggregate, dict):
        source_values = [values.get(metric_id) for metric_id in aggregate.get("source_metric_ids", [])]
        target = values.get(aggregate.get("metric_id"))
        if is_number(target) and source_values and all(is_number(value) for value in source_values):
            expected = sum(source_values) / len(source_values)
            if not math.isclose(target, expected, rel_tol=0.0, abs_tol=aggregate.get("tolerance", 1e-6)):
                add(f"metric_values.{aggregate['metric_id']} must equal the declared source-metric mean")

    expected_scores = None
    if isinstance(composite, dict):
        components = composite.get("components")
        component_ids = (
            [component.get("metric_id") for component in components if isinstance(component, dict)]
            if isinstance(components, list)
            else []
        )
        target_metric_id = composite.get("metric_id")
        composite_status = composite.get("status", "active")
        if composite_status not in {"active", "pending_reference_baselines"}:
            add("dataset overall_score_composite has an unsupported status")
        physics_null_components = [
            component
            for component in components or []
            if isinstance(component, dict)
            and component.get("transform") == "physics_null_skill"
        ]
        if physics_null_components and composite.get("allow_negative_scores") is not True:
            add("dataset physics-null composite must allow negative scores")
        for component in physics_null_components:
            if not isinstance(component.get("baseline_id"), str) or not component["baseline_id"]:
                add("dataset physics-null component requires a baseline_id")
            baseline_error = component.get("baseline_error")
            if composite_status == "active" and (
                not is_number(baseline_error) or baseline_error <= 0
            ):
                add("active dataset physics-null component requires a positive baseline_error")
        if (
            composite.get("operation") != "weighted_component_scores"
            or not isinstance(components, list)
            or not components
            or len(component_ids) != len(components)
            or not isinstance(target_metric_id, str)
            or target_metric_id not in expected_ids
            or any(not isinstance(metric_id, str) or metric_id not in expected_ids for metric_id in component_ids)
        ):
            add("dataset overall_score_composite does not reference a valid target and component metric set")
        elif composite_status == "active" and all(
            isinstance(metric_id, str)
            and (
                metric_id in fixed_zero_components
                or is_number(values.get(metric_id))
            )
            for metric_id in component_ids
        ):
            try:
                expected = composite_overall_score(
                    values,
                    composite,
                    fixed_zero_component_ids=tuple(fixed_zero_components),
                )
            except (KeyError, TypeError, ValueError) as exc:
                add(f"dataset overall_score_composite is invalid: {exc}")
            else:
                tolerance = composite.get("tolerance", 1e-6)
                if not is_number(tolerance) or tolerance < 0:
                    add("dataset overall_score_composite requires a metric_id and non-negative tolerance")
                elif not is_number(values.get(target_metric_id)) or not math.isclose(
                    values[target_metric_id], expected, rel_tol=0.0, abs_tol=tolerance
                ):
                    add(f"metric_values.{target_metric_id} does not match its declared composite equation")
    score_tolerance = 1e-6
    if isinstance(component_groups, dict):
        groups = component_groups.get("groups")
        target_ids = (
            [group.get("metric_id") for group in groups if isinstance(group, dict)]
            if isinstance(groups, list)
            else []
        )
        grouped_component_ids = (
            [
                metric_id
                for group in groups
                if isinstance(group, dict) and isinstance(group.get("component_metric_ids"), list)
                for metric_id in group["component_metric_ids"]
            ]
            if isinstance(groups, list)
            else []
        )
        score_tolerance = component_groups.get("tolerance", 1e-6)
        if (
            component_groups.get("operation") != "normalized_weighted_component_scores"
            or not isinstance(groups, list)
            or not groups
            or len(target_ids) != len(groups)
            or any(not isinstance(metric_id, str) or metric_id not in expected_ids for metric_id in target_ids)
            or any(
                not isinstance(metric_id, str) or metric_id not in expected_ids
                for metric_id in grouped_component_ids
            )
        ):
            add("dataset component_score_groups does not reference a valid target and component metric set")
        elif not isinstance(composite, dict):
            add("dataset component_score_groups requires overall_score_composite")
        elif not is_number(score_tolerance) or score_tolerance < 0:
            add("dataset component_score_groups requires a non-negative tolerance")
        elif (
            isinstance(composite, dict)
            and composite.get("status", "active") == "active"
            and all(
                metric_id in fixed_zero_components
                or is_number(values.get(metric_id))
                for metric_id in grouped_component_ids
            )
        ):
            try:
                expected_scores = composite_component_group_scores(
                    values,
                    composite,
                    component_groups,
                    fixed_zero_component_ids=tuple(fixed_zero_components),
                )
            except (KeyError, TypeError, ValueError) as exc:
                add(f"dataset component_score_groups is invalid: {exc}")
    elif dataset.get("submission_format") == "legacy_external_aero" and all(
        is_number(values.get(metric_id))
        for metric_id in {"cd_r2", "cl_r2", "velocity_profile_r2", "cp_cut_r2"}
    ):
        try:
            expected_scores = legacy_aero_scores(values)
        except (KeyError, TypeError, ValueError):
            pass
    if expected_scores is not None:
        for metric_id, expected in expected_scores.items():
            if not is_number(values.get(metric_id)) or not math.isclose(
                values[metric_id], expected, rel_tol=0.0, abs_tol=score_tolerance
            ):
                add(f"metric_values.{metric_id} does not match its declared score equation")


def scoring_support_manifest_path(
    add: Any,
    dataset_spec: dict[str, Any],
    submission: dict[str, Any],
    *,
    binding: dict[str, Any] | None = None,
) -> Path | None:
    """Locate the repository copy of the benchmark-owned scoring-support manifest."""

    if binding is None:
        binding = dataset_spec.get("scoring_support")
    if not isinstance(binding, dict):
        add("schema v3 requires scoring_support in the benchmark specification")
        return None
    manifest_file = binding.get("manifest_file")
    dataset_directory = ROOT / "benchmark-specs" / submission["dataset_id"]
    if isinstance(manifest_file, str) and manifest_file:
        relative = Path(manifest_file)
        if relative.is_absolute() or ".." in relative.parts:
            add("benchmark scoring_support.manifest_file must remain inside its dataset specification")
            return None
        return dataset_directory / relative

    release_id = binding.get("release_id")
    matches = []
    for candidate in dataset_directory.glob("scoring-support/**/manifest.json"):
        try:
            value = load_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("release_id") == release_id:
            matches.append(candidate)
    if len(matches) != 1:
        add(
            "benchmark scoring_support must provide manifest_file or identify exactly one local "
            f"manifest for release_id={release_id!r}"
        )
        return None
    return matches[0]


def validate_v3_scoring_support(
    add: Any,
    submission: dict[str, Any],
    dataset_spec: dict[str, Any],
    split_spec_entry: dict[str, Any],
    *,
    candidate_dry_run: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate and load the fixed spatial scoring support used by schema v3.

    Normal validation accepts only an open, owner-approved official release.
    ``candidate_dry_run`` is a separate non-approving path which accepts only a
    closed owner-review candidate and its explicitly nested candidate manifest.
    It never falls back from one lifecycle state to the other.
    """

    declared = submission["scoring_support"]
    owner_binding = dataset_spec.get("scoring_support", {})
    owner_status = owner_binding.get("status")
    submissions_open = owner_binding.get("submissions_open")
    if candidate_dry_run:
        if owner_status not in {"candidate", "owner_review_required"}:
            add(
                "candidate dry-run requires benchmark scoring_support.status "
                "to be 'candidate' or 'owner_review_required', never official"
            )
        if submissions_open is not False:
            add("candidate dry-run requires scoring_support.submissions_open=false")
        if declared.get("status") != "candidate":
            add("candidate dry-run requires submission scoring_support.status='candidate'")
        candidate_binding = owner_binding.get("candidate_manifest")
        if not isinstance(candidate_binding, dict):
            add(
                "candidate dry-run requires benchmark "
                "scoring_support.candidate_manifest"
            )
            candidate_binding = {}
        if candidate_binding.get("status") != "candidate":
            add(
                "benchmark scoring_support.candidate_manifest.status must be "
                "'candidate'"
            )
        for forbidden in ("owner_approval", "publication_validation"):
            if forbidden in owner_binding or forbidden in candidate_binding:
                add(
                    "candidate dry-run benchmark metadata must not contain "
                    f"{forbidden}"
                )
        active_binding = candidate_binding
        expected_manifest_status = "candidate"
    else:
        if owner_status != "official" or submissions_open is not True:
            reason = owner_binding.get("closed_reason") or "dataset-owner approval is incomplete"
            add(
                "schema v3 submissions are closed for this dataset's scoring support: "
                f"status={owner_status!r}, submissions_open={submissions_open!r}; {reason}"
            )
        active_binding = owner_binding
        expected_manifest_status = "official"
    required_owner_fields = (
        "release_id",
        "manifest_file",
        "manifest_url",
        "manifest_sha256",
    )
    missing_owner_fields = [
        key
        for key in required_owner_fields
        if not isinstance(active_binding.get(key), str) or not active_binding[key].strip()
    ]
    if missing_owner_fields:
        prefix = (
            "benchmark candidate scoring_support is incomplete; missing: "
            if candidate_dry_run
            else "benchmark scoring_support is incomplete; missing: "
        )
        add(
            f"{prefix}{', '.join(missing_owner_fields)}"
        )
    owner_approval = owner_binding.get("owner_approval")
    if owner_status == "official" and (
        not isinstance(owner_approval, dict)
        or any(
            not isinstance(owner_approval.get(key), str) or not owner_approval[key].strip()
            for key in ("approved_by", "approved_at", "pull_request_url")
        )
    ):
        add("official benchmark scoring_support requires complete dataset-owner approval metadata")
    for key in ("release_id", "manifest_url", "manifest_sha256"):
        if declared.get(key) != active_binding.get(key):
            qualifier = " candidate" if candidate_dry_run else ""
            add(
                f"scoring_support.{key} must match the benchmark{qualifier} "
                "manifest binding"
            )
    if not candidate_dry_run and declared.get("status") != owner_status:
        add("scoring_support.status must match the benchmark specification")

    manifest_path = scoring_support_manifest_path(
        add,
        dataset_spec,
        submission,
        binding=active_binding,
    )
    if manifest_path is None:
        return None, None
    if not manifest_path.is_file():
        add(f"missing scoring-support manifest: {manifest_path.relative_to(ROOT)}")
        return None, None
    if sha256_file(manifest_path) != declared["manifest_sha256"]:
        add("scoring-support manifest does not match scoring_support.manifest_sha256")
    try:
        support_manifest = load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as error:
        add(f"cannot read scoring-support manifest: {error}")
        return None, None
    for error in schema_errors(
        support_manifest,
        "manifest.schema.json",
        schema_version="scoring-support/v1",
    ):
        add(f"scoring-support manifest {error}")

    manifest_identities = {
        "release_id": declared["release_id"],
        "dataset_id": submission["dataset_id"],
        "dataset_version": submission["dataset_version"],
        "evaluation_reference_version": submission["evaluation"]["reference_version"],
    }
    for key, expected in manifest_identities.items():
        if support_manifest.get(key) != expected:
            add(f"scoring-support manifest {key} must equal {expected!r}")
    if support_manifest.get("status") != expected_manifest_status:
        add(
            "candidate dry-run requires a candidate scoring-support manifest"
            if candidate_dry_run
            else "schema v3 submissions require an official scoring-support manifest"
        )
    if candidate_dry_run and "owner_approval" in support_manifest:
        add("candidate scoring-support manifest must not contain owner_approval")
    if owner_status == "official" and support_manifest.get("owner_approval") != owner_approval:
        add("scoring-support manifest owner_approval must match the benchmark specification")

    dataset_metrics = {
        metric.get("id"): metric
        for metric in dataset_spec.get("metrics", [])
        if isinstance(metric, dict) and isinstance(metric.get("id"), str)
    }
    support_aggregations = {
        "per_geometry_then_macro_average",
        "flatten_all_aligned_field_values",
        "all_test_cases",
        "benchmark_field_rrmse_across_cases",
        "benchmark_scalar_rrmse_across_cases",
    }
    required_bound_metric_ids = {
        metric_id
        for metric_id, metric in dataset_metrics.items()
        if metric.get("aggregation") in support_aggregations
    }

    def expected_reduction(metric_id: str, metric: dict[str, Any]) -> str | None:
        aggregation = metric.get("aggregation")
        if aggregation in {
            "benchmark_field_rrmse_across_cases",
            "benchmark_scalar_rrmse_across_cases",
        }:
            return "dataset_reference"
        if metric_id.endswith("_rel_l1"):
            return "relative_l1_percent"
        if metric_id.endswith("_rel_l2"):
            return "relative_l2_percent"
        if metric_id.endswith("_rrmse"):
            return "dataset_reference"
        if metric_id.endswith("_rmse"):
            return "rmse"
        if metric_id.endswith("_mse"):
            return "mse"
        if metric_id.endswith("_mae"):
            return "mae"
        if metric_id.endswith("_r2"):
            return "r2"
        return None

    bound_metric_ids: list[str] = []
    required_artifact_roles: dict[str, set[str]] = {}
    for support in support_manifest.get("supports", []):
        if not isinstance(support, dict):
            continue
        support_id = support.get("id")
        quantities = [
            quantity for quantity in support.get("quantities", []) if isinstance(quantity, dict)
        ]
        quantity_ids = [quantity.get("id") for quantity in quantities]
        if len(quantity_ids) != len(set(quantity_ids)):
            add(f"scoring support {support_id!r} quantity IDs must be unique")
        for quantity in quantities:
            component_ids = [
                component.get("id")
                for component in quantity.get("components", [])
                if isinstance(component, dict)
            ]
            if len(component_ids) != len(set(component_ids)):
                add(
                    f"scoring support {support_id!r}/{quantity.get('id')!r} component IDs "
                    "must be unique"
                )
        for binding in support.get("metric_bindings", []):
            if not isinstance(binding, dict):
                continue
            metric_id = binding.get("metric_id")
            bound_metric_ids.append(metric_id)
            metric = dataset_metrics.get(metric_id)
            if metric is None:
                add(f"scoring support {support_id!r} binds unknown metric_id {metric_id!r}")
                continue
            if binding.get("quantity_id") not in quantity_ids:
                add(
                    f"scoring support {support_id!r} metric {metric_id!r} references "
                    "an unknown quantity"
                )
            if binding.get("aggregation") != metric.get("aggregation"):
                add(
                    f"scoring support {support_id!r} metric {metric_id!r} aggregation "
                    "must match the dataset specification"
                )
            if binding.get("dataset_weighting") != metric.get("weighting"):
                add(
                    f"scoring support {support_id!r} metric {metric_id!r} "
                    "dataset_weighting must match the dataset specification"
                )
            expected = expected_reduction(metric_id, metric)
            if expected is None:
                add(
                    f"dataset metric {metric_id!r} requires an explicit supported "
                    "scoring reduction"
                )
            elif binding.get("reduction") != expected:
                add(
                    f"scoring support {support_id!r} metric {metric_id!r} reduction "
                    f"must be {expected!r}"
                )
            expected_weighting = evaluator_weighting(metric.get("weighting"))
            if binding.get("weighting") != expected_weighting:
                add(
                    f"scoring support {support_id!r} metric {metric_id!r} weighting "
                    f"must be {expected_weighting!r}"
                )
            expected_case_evidence = (
                "aggregate_only"
                if expected in {"r2", "dataset_reference"}
                else "metric_value"
            )
            if binding.get("case_evidence") != expected_case_evidence:
                add(
                    f"scoring support {support_id!r} metric {metric_id!r} "
                    f"case_evidence must be {expected_case_evidence!r}"
                )
            if expected == "dataset_reference":
                reference_rule = binding.get("reference_rule", {})
                if (
                    reference_rule.get("id") != metric.get("aggregation")
                    or reference_rule.get("version")
                    != dataset_spec.get("evaluation_reference_version")
                ):
                    add(
                        f"scoring support {support_id!r} metric {metric_id!r} "
                        "must pin the dataset aggregation rule and evaluator version"
                    )
            scalar_aggregation = metric.get("aggregation") in {
                "all_test_cases",
                "benchmark_scalar_rrmse_across_cases",
            }
            if scalar_aggregation and support.get("domain") != "scalar_case":
                add(
                    f"scoring support {support_id!r} metric {metric_id!r} "
                    "requires a scalar_case support"
                )
            if not scalar_aggregation and support.get("domain") == "scalar_case":
                add(
                    f"scoring support {support_id!r} metric {metric_id!r} "
                    "cannot use a scalar_case support"
                )
        location = support.get("location_definition", {})
        roles = {
            value
            for key, value in location.items()
            if key in {"artifact_role", "target_artifact_role"}
            and isinstance(value, str)
        }
        weight_rule = location.get("weight_rule", {})
        if isinstance(weight_rule, dict) and isinstance(weight_rule.get("artifact_role"), str):
            roles.add(weight_rule["artifact_role"])
        required_artifact_roles[support_id] = roles
    if len(bound_metric_ids) != len(set(bound_metric_ids)):
        add("each scoring-support metric may be bound only once")
    bound_metric_id_set = {
        metric_id for metric_id in bound_metric_ids if isinstance(metric_id, str)
    }
    if bound_metric_id_set != required_bound_metric_ids:
        add(
            "scoring-support metric bindings must exactly cover dataset field and "
            "case-scalar metrics; "
            f"missing={sorted(required_bound_metric_ids - bound_metric_id_set)}, "
            f"unexpected={sorted(bound_metric_id_set - required_bound_metric_ids)}"
        )

    case_set_entry = next(
        (
            entry
            for entry in support_manifest.get("case_sets", [])
            if isinstance(entry, dict) and entry.get("id") == submission["case_set_id"]
        ),
        None,
    )
    if case_set_entry is None:
        add(f"scoring-support manifest does not define case_set_id {submission['case_set_id']!r}")
        return support_manifest, None
    index_file = case_set_entry.get("index_file")
    if not isinstance(index_file, str) or not index_file:
        add("scoring-support case set must declare index_file")
        return support_manifest, None
    relative = Path(index_file)
    if relative.is_absolute() or ".." in relative.parts:
        add("scoring-support case-set index_file must remain beside the manifest")
        return support_manifest, None
    case_index_path = manifest_path.parent / relative
    if not case_index_path.is_file():
        add(f"missing scoring-support case index: {case_index_path.relative_to(ROOT)}")
        return support_manifest, None
    if sha256_file(case_index_path) != case_set_entry.get("index_sha256"):
        add("scoring-support case index does not match its manifest digest")
    try:
        case_index = load_json(case_index_path)
    except (OSError, json.JSONDecodeError) as error:
        add(f"cannot read scoring-support case index: {error}")
        return support_manifest, None
    for error in schema_errors(
        case_index,
        "case-index.schema.json",
        schema_version="scoring-support/v1",
    ):
        add(f"scoring-support case index {error}")
    case_index_identities = {
        "release_id": declared["release_id"],
        "dataset_id": submission["dataset_id"],
        "case_set_id": submission["case_set_id"],
    }
    for key, expected in case_index_identities.items():
        if case_index.get(key) != expected:
            add(f"scoring-support case index {key} must equal {expected!r}")
    if case_index.get("case_count") != split_spec_entry.get("case_count"):
        add("scoring-support case index case_count must match the benchmark split")

    support_ids = [
        support.get("id")
        for support in support_manifest.get("supports", [])
        if isinstance(support, dict) and isinstance(support.get("id"), str)
    ]
    if len(support_ids) != len(set(support_ids)):
        add("scoring-support manifest support IDs must be unique")
    loaded_cases: list[dict[str, Any]] = []
    indexed_case_ids: list[str] = []
    referenced_files: set[str] = set()
    for chunk_entry in case_index.get("chunks", []):
        if not isinstance(chunk_entry, dict):
            continue
        filename = chunk_entry.get("file")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
        ):
            add(f"invalid scoring-support chunk filename: {filename!r}")
            continue
        if filename in referenced_files:
            add(f"scoring-support case index references {filename!r} more than once")
            continue
        referenced_files.add(filename)
        chunk_path = case_index_path.parent / filename
        if not chunk_path.is_file():
            add(f"missing scoring-support case chunk: {chunk_path.relative_to(ROOT)}")
            continue
        if sha256_file(chunk_path) != chunk_entry.get("sha256"):
            add(f"scoring-support case chunk {filename} does not match its index digest")
        try:
            chunk = load_json(chunk_path)
        except (OSError, json.JSONDecodeError) as error:
            add(f"cannot read scoring-support case chunk {filename}: {error}")
            continue
        for error in schema_errors(
            chunk,
            "case-chunk.schema.json",
            schema_version="scoring-support/v1",
        ):
            add(f"scoring-support case chunk {filename} {error}")
        for key, expected in case_index_identities.items():
            if chunk.get(key) != expected:
                add(f"scoring-support case chunk {filename} {key} must equal {expected!r}")
        chunk_cases = [case for case in chunk.get("cases", []) if isinstance(case, dict)]
        chunk_case_ids = [case.get("case_id") for case in chunk_cases]
        if chunk_case_ids != chunk_entry.get("case_ids"):
            add(f"scoring-support case chunk {filename} order differs from its index entry")
        if len(chunk_cases) != chunk_entry.get("case_count"):
            add(f"scoring-support case chunk {filename} count differs from its index entry")
        indexed_case_ids.extend(chunk_entry.get("case_ids", []))
        loaded_cases.extend(chunk_cases)
        for case in chunk_cases:
            observed_supports: set[str] = set()
            for instance in case.get("support_instances", []):
                if not isinstance(instance, dict):
                    continue
                support_id = instance.get("support_id")
                if support_id in observed_supports:
                    add(
                        f"scoring-support case {case.get('case_id')} contains duplicate "
                        f"support_id {support_id!r}"
                    )
                observed_supports.add(support_id)
                if support_id not in support_ids:
                    add(
                        f"scoring-support case {case.get('case_id')} references unknown "
                        f"support_id {support_id!r}"
                    )
                roles = [
                    artifact.get("role")
                    for artifact in instance.get("artifacts", [])
                    if isinstance(artifact, dict)
                ]
                if len(roles) != len(set(roles)):
                    add(
                        f"scoring-support case {case.get('case_id')}/{support_id} "
                        "artifact roles must be unique"
                    )
                missing_roles = sorted(required_artifact_roles.get(support_id, set()) - set(roles))
                if missing_roles:
                    add(
                        f"scoring-support case {case.get('case_id')}/{support_id} is missing "
                        f"required artifact roles: {missing_roles}"
                    )
            missing_supports = sorted(set(support_ids) - observed_supports)
            if missing_supports:
                add(
                    f"scoring-support case {case.get('case_id')} is missing support "
                    f"instances: {missing_supports}"
                )
    if len(loaded_cases) != case_index.get("case_count"):
        add("scoring-support loaded case count does not match the case index")
    if len(indexed_case_ids) != len(set(indexed_case_ids)):
        add("scoring-support case IDs must be unique across chunks")
    split_path = ROOT / "benchmark-specs" / submission["dataset_id"] / split_spec_entry["index_file"]
    if split_path.is_file():
        expected_case_ids = load_json(split_path).get("case_ids", [])
        if indexed_case_ids != expected_case_ids:
            missing = sorted(set(expected_case_ids) - set(indexed_case_ids))
            unexpected = sorted(set(indexed_case_ids) - set(expected_case_ids))
            add(
                "scoring-support case coverage differs from the benchmark split; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
    actual_chunk_files = {
        path.relative_to(case_index_path.parent).as_posix()
        for path in case_index_path.parent.rglob("chunk-*.json")
    }
    if actual_chunk_files != referenced_files:
        add(
            "scoring-support case-index directory contains unindexed chunks: "
            f"{sorted(actual_chunk_files - referenced_files)}"
        )
    case_index["_loaded_cases"] = loaded_cases
    return support_manifest, case_index


def _hilift_force_r2(
    truth: list[float], prediction: list[float]
) -> float | None:
    """Replay the dataset-declared equal-case R2 reduction."""

    if len(truth) != len(prediction) or len(truth) < 2:
        return None
    truth_mean = math.fsum(truth) / len(truth)
    denominator = math.fsum((value - truth_mean) ** 2 for value in truth)
    if denominator <= 0.0:
        return None
    numerator = math.fsum(
        (predicted - target) ** 2
        for target, predicted in zip(truth, prediction, strict=True)
    )
    return 1.0 - numerator / denominator


def _validate_hiliftaeroml_force_metric_bindings(
    add: Any,
    *,
    submission: dict[str, Any],
    case_metrics: dict[str, Any],
) -> None:
    """Replay every HiLift force metric from per-case coefficient evidence."""

    cases = case_metrics.get("cases")
    if not isinstance(cases, list) or not cases:
        add("HiLiftAeroML force recomputation requires a non-empty cases array")
        return
    submitted_aggregates = submission.get("metric_values")
    if not isinstance(submitted_aggregates, dict):
        add("HiLiftAeroML force recomputation requires submission metric_values")
        return

    coefficient_inputs: dict[str, list[float]] = {
        field: [] for field in HILIFT_FORCE_COEFFICIENT_FIELDS
    }
    per_case_errors: dict[str, list[float]] = {
        metric_id: [] for metric_id in HILIFT_FORCE_MAE_INPUTS
    }
    complete_coefficient_cases = 0
    expected_coefficient_fields = set(HILIFT_FORCE_COEFFICIENT_FIELDS)
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = case.get("case_id")
        coefficients = case.get("force_coefficients")
        if not isinstance(coefficients, dict):
            add(f"case-metrics {case_id} has no force_coefficients evidence")
            continue
        observed_fields = set(coefficients)
        if observed_fields != expected_coefficient_fields:
            add(
                f"case-metrics {case_id} force_coefficients must contain exactly "
                f"{sorted(expected_coefficient_fields)}; "
                f"missing={sorted(expected_coefficient_fields-observed_fields)}, "
                f"unexpected={sorted(observed_fields-expected_coefficient_fields)}"
            )
            continue
        invalid_fields = [
            field
            for field in HILIFT_FORCE_COEFFICIENT_FIELDS
            if not is_number(coefficients.get(field))
        ]
        if invalid_fields:
            add(
                f"case-metrics {case_id} force_coefficients must be finite numbers; "
                f"invalid={invalid_fields}"
            )
            continue

        numeric_coefficients = {
            field: float(coefficients[field])
            for field in HILIFT_FORCE_COEFFICIENT_FIELDS
        }
        complete_coefficient_cases += 1
        for field, value in numeric_coefficients.items():
            coefficient_inputs[field].append(value)

        expected_errors = {
            metric_id: abs(
                numeric_coefficients[prediction_field]
                - numeric_coefficients[truth_field]
            )
            for metric_id, (truth_field, prediction_field) in (
                HILIFT_FORCE_MAE_INPUTS.items()
            )
        }
        nonspatial = case.get("nonspatial_metric_values")
        if not isinstance(nonspatial, dict):
            add(f"case-metrics {case_id} has no nonspatial_metric_values")
            nonspatial = {}

        force_supports = [
            support
            for support in case.get("supports", [])
            if isinstance(support, dict)
            and support.get("support_id") == HILIFT_FORCE_SUPPORT_ID
        ]
        if len(force_supports) != 1:
            add(
                f"case-metrics {case_id} requires exactly one "
                f"{HILIFT_FORCE_SUPPORT_ID!r} support for force recomputation"
            )
            support_metric_values: dict[str, Any] = {}
        else:
            candidate_values = force_supports[0].get("metric_values")
            if not isinstance(candidate_values, dict):
                add(
                    f"case-metrics {case_id}/{HILIFT_FORCE_SUPPORT_ID} has no "
                    "metric_values"
                )
                support_metric_values = {}
            else:
                support_metric_values = candidate_values

        for metric_id, expected in expected_errors.items():
            per_case_errors[metric_id].append(expected)
            truth_field, prediction_field = HILIFT_FORCE_MAE_INPUTS[metric_id]
            for container_name, container in (
                ("nonspatial_metric_values", nonspatial),
                (
                    f"supports[{HILIFT_FORCE_SUPPORT_ID}].metric_values",
                    support_metric_values,
                ),
            ):
                submitted = container.get(metric_id)
                if not is_number(submitted) or not math.isclose(
                    float(submitted), expected, rel_tol=1e-12, abs_tol=1e-12
                ):
                    add(
                        f"case-metrics {case_id} {container_name}.{metric_id} must "
                        f"equal abs(force_coefficients.{prediction_field} - "
                        f"force_coefficients.{truth_field})"
                    )

    if complete_coefficient_cases != len(cases):
        add(
            "HiLiftAeroML force recomputation requires complete finite coefficient "
            "evidence for every test case"
        )
        return

    expected_aggregates = {
        metric_id: math.fsum(values) / len(values)
        for metric_id, values in per_case_errors.items()
    }
    for metric_id, (truth_field, prediction_field) in HILIFT_FORCE_R2_INPUTS.items():
        recomputed = _hilift_force_r2(
            coefficient_inputs[truth_field],
            coefficient_inputs[prediction_field],
        )
        if recomputed is None:
            add(
                f"metric_values.{metric_id} cannot be recomputed because the "
                "complete truth coefficient array has zero variance or fewer than two cases"
            )
        else:
            expected_aggregates[metric_id] = recomputed

    for metric_id, expected in expected_aggregates.items():
        submitted = submitted_aggregates.get(metric_id)
        if not is_number(submitted) or not math.isclose(
            float(submitted), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            add(
                f"metric_values.{metric_id} must equal the independent HiLiftAeroML "
                "force-coefficient recomputation"
            )


def validate_v3_case_metrics(
    add: Any,
    directory: Path,
    submission: dict[str, Any],
    split_case_ids: list[str],
    support_manifest: dict[str, Any] | None,
    support_case_index: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the per-case metric evidence and its aggregate scalar binding."""

    surface_only = (
        submission.get("dataset_id") == "drivaerml"
        and submission.get("prediction_scope") == "surface_only"
    )
    unavailable_metric_ids = (
        {"volume_velocity_rel_l2", "volume_pressure_rel_l2", "velocity_profile_r2"}
        if surface_only
        else set()
    )

    declaration = submission["case_metrics"]
    path = safe_submission_path(
        add,
        directory,
        declaration["file"],
        label="case_metrics.file",
    )
    if path is None:
        return None
    if not path.is_file():
        add(f"missing case metrics: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        return None
    if sha256_file(path) != declaration["sha256"]:
        add("case-metrics file does not match case_metrics.sha256")
    try:
        case_metrics = load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        add(f"cannot read case metrics: {error}")
        return None
    for error in schema_errors(case_metrics, "case-metrics.schema.json", schema_version="v3"):
        add(f"{declaration['file']} {error}")

    identities = {
        "submission_id": submission["submission_id"],
        "dataset_id": submission["dataset_id"],
        "split_id": submission["split_id"],
        "case_set_id": submission["case_set_id"],
        "scoring_support_release_id": submission["scoring_support"]["release_id"],
        "scoring_support_manifest_sha256": submission["scoring_support"]["manifest_sha256"],
    }
    for key, expected in identities.items():
        if case_metrics.get(key) != expected:
            add(f"{declaration['file']} {key} must equal {expected!r}")
    if declaration.get("case_count") != len(split_case_ids):
        add(f"case_metrics.case_count must be {len(split_case_ids)} for this split")
    if case_metrics.get("case_count") != declaration.get("case_count"):
        add("case-metrics case_count must equal submission.json case_metrics.case_count")
    if case_metrics.get("metric_values") != submission.get("metric_values"):
        add("case-metrics metric_values must exactly match submission.json")

    cases = case_metrics.get("cases", [])
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if case_ids != split_case_ids:
        missing = sorted(set(split_case_ids) - set(case_ids))
        unexpected = sorted(set(case_ids) - set(split_case_ids))
        add(
            "case-metrics case order differs from the benchmark split; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    expected_instances: dict[str, dict[str, int]] = {}
    if isinstance(support_case_index, dict):
        for case in support_case_index.get("_loaded_cases", support_case_index.get("cases", [])):
            if not isinstance(case, dict):
                continue
            expected_instances[case.get("case_id")] = {
                instance.get("support_id"): instance.get("entity_count")
                for instance in case.get("support_instances", [])
                if isinstance(instance, dict)
                and isinstance(instance.get("support_id"), str)
                and isinstance(instance.get("entity_count"), int)
            }

    bound_metrics: dict[str, str] = {}
    bound_metric_bindings: dict[str, dict[str, Any]] = {}
    extrapolation_policies: dict[str, str] = {}
    if isinstance(support_manifest, dict):
        for support in support_manifest.get("supports", []):
            if not isinstance(support, dict):
                continue
            support_id = support.get("id")
            if isinstance(support_id, str):
                extrapolation_policies[support_id] = support.get("extrapolation_policy")
            for binding in support.get("metric_bindings", []):
                if isinstance(binding, str):
                    bound_metrics[binding] = support_id
                elif isinstance(binding, dict):
                    metric_id = binding.get("metric_id") or binding.get("id")
                    if isinstance(metric_id, str):
                        bound_metrics[metric_id] = support_id
                        bound_metric_bindings[metric_id] = binding

    bound_metrics_by_support: dict[str, set[str]] = {}
    for metric_id, support_id in bound_metrics.items():
        if bound_metric_bindings.get(metric_id, {}).get("case_evidence") == "metric_value":
            bound_metrics_by_support.setdefault(support_id, set()).add(metric_id)
    per_case_values: dict[str, list[float]] = {
        metric_id: []
        for metric_id, binding in bound_metric_bindings.items()
        if (
            binding.get("case_evidence") == "metric_value"
            and metric_id not in unavailable_metric_ids
        )
    }
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = case.get("case_id")
        expected_for_case = expected_instances.get(case_id, {})
        if surface_only:
            expected_for_case = {
                support_id: count
                for support_id, count in expected_for_case.items()
                if support_id != "volume_native_cells"
            }
        observed_support_ids: set[str] = set()
        for support in case.get("supports", []):
            if not isinstance(support, dict):
                continue
            support_id = support.get("support_id")
            if not isinstance(support_id, str):
                continue
            if support_id in observed_support_ids:
                add(f"{declaration['file']} {case_id} contains duplicate support_id {support_id!r}")
            observed_support_ids.add(support_id)
            if support_id not in expected_for_case:
                add(
                    f"{declaration['file']} {case_id} contains unexpected "
                    f"support_id {support_id!r}"
                )
            support_count = support.get("support_count")
            scored_count = support.get("scored_count")
            coverage = support.get("coverage_fraction")
            expected_count = expected_for_case.get(support_id)
            if expected_count is not None and support_count != expected_count:
                add(
                    f"{declaration['file']} {case_id}/{support_id} support_count must equal "
                    "the fixed scoring-support entity_count"
                )
            if isinstance(support_count, int) and isinstance(scored_count, int):
                if scored_count != support_count:
                    add(f"{declaration['file']} {case_id}/{support_id} must score every support entity")
                calculated = scored_count / support_count if support_count else 0.0
                if is_number(coverage) and not math.isclose(
                    coverage,
                    calculated,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    add(
                        f"{declaration['file']} {case_id}/{support_id} coverage_fraction "
                        "does not match scored_count/support_count"
                    )
            if not is_number(coverage) or not math.isclose(coverage, 1.0, rel_tol=0.0, abs_tol=1e-12):
                add(f"{declaration['file']} {case_id}/{support_id} requires coverage_fraction=1.0")
            if support.get("unmapped_count") != 0:
                add(f"{declaration['file']} {case_id}/{support_id} requires unmapped_count=0")
            if (
                extrapolation_policies.get(support_id) == "forbidden"
                and support.get("extrapolated_count") != 0
            ):
                add(
                    f"{declaration['file']} {case_id}/{support_id} forbids extrapolated scoring values"
                )
            weight_coverage = support.get("weight_coverage_fraction")
            if not is_number(weight_coverage) or not math.isclose(
                weight_coverage,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                add(f"{declaration['file']} {case_id}/{support_id} requires weight_coverage_fraction=1.0")
            metric_values = support.get("metric_values", {})
            observed_metric_ids = set(metric_values) if isinstance(metric_values, dict) else set()
            expected_metric_ids = (
                bound_metrics_by_support.get(support_id, set())
                - unavailable_metric_ids
            )
            if observed_metric_ids != expected_metric_ids:
                missing_metrics = sorted(expected_metric_ids - observed_metric_ids)
                unexpected_metrics = sorted(observed_metric_ids - expected_metric_ids)
                add(
                    f"{declaration['file']} {case_id}/{support_id} metric IDs must "
                    "exactly match the official support bindings; "
                    f"missing={missing_metrics}, unexpected={unexpected_metrics}"
                )
            expected_statistics_ids = {
                metric_id
                for metric_id in expected_metric_ids
                if bound_metric_bindings.get(metric_id, {}).get("reduction")
                == "relative_l2_percent"
            }
            metric_statistics = support.get("metric_sufficient_statistics", {})
            observed_statistics_ids = (
                set(metric_statistics) if isinstance(metric_statistics, dict) else set()
            )
            if observed_statistics_ids != expected_statistics_ids:
                missing_statistics = sorted(
                    expected_statistics_ids - observed_statistics_ids
                )
                unexpected_statistics = sorted(
                    observed_statistics_ids - expected_statistics_ids
                )
                add(
                    f"{declaration['file']} {case_id}/{support_id} relative-L2 "
                    "sufficient-statistic IDs must exactly match the official "
                    f"support bindings; missing={missing_statistics}, "
                    f"unexpected={unexpected_statistics}"
                )
            for metric_id, statistics in (
                metric_statistics.items() if isinstance(metric_statistics, dict) else []
            ):
                if not isinstance(statistics, dict):
                    continue
                binding = bound_metric_bindings.get(metric_id, {})
                expected_identity = {
                    "reduction": "relative_l2_percent",
                    "weighting": binding.get("weighting"),
                    "dataset_weighting": binding.get("dataset_weighting"),
                }
                for key, expected in expected_identity.items():
                    if statistics.get(key) != expected:
                        add(
                            f"{declaration['file']} {case_id}/{support_id} "
                            f"metric_sufficient_statistics.{metric_id}.{key} "
                            f"must equal {expected!r}"
                        )
                entity_count = statistics.get("entity_count")
                if isinstance(support_count, int) and entity_count != support_count:
                    add(
                        f"{declaration['file']} {case_id}/{support_id} "
                        f"metric_sufficient_statistics.{metric_id}.entity_count "
                        "must equal support_count"
                    )
                total_weight = statistics.get("total_weight")
                if (
                    binding.get("weighting") == "uniform"
                    and isinstance(entity_count, int)
                    and is_number(total_weight)
                    and not math.isclose(
                        float(total_weight),
                        float(entity_count),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    add(
                        f"{declaration['file']} {case_id}/{support_id} "
                        f"metric_sufficient_statistics.{metric_id}.total_weight "
                        "must equal entity_count for uniform weighting"
                    )
                numerator = statistics.get("numerator")
                denominator = statistics.get("denominator")
                metric_value = (
                    metric_values.get(metric_id)
                    if isinstance(metric_values, dict)
                    else None
                )
                if (
                    is_number(numerator)
                    and float(numerator) >= 0
                    and is_number(denominator)
                    and float(denominator) > 0
                    and is_number(metric_value)
                ):
                    calculated_value = 100.0 * math.sqrt(
                        float(numerator) / float(denominator)
                    )
                    if not math.isclose(
                        float(metric_value),
                        calculated_value,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        add(
                            f"{declaration['file']} {case_id}/{support_id} "
                            f"metric_values.{metric_id} must equal "
                            "100*sqrt(numerator/denominator) from its sufficient "
                            "statistics"
                        )
            for metric_id, value in metric_values.items():
                expected_support_id = bound_metrics.get(metric_id)
                if expected_support_id is None:
                    add(
                        f"{declaration['file']} {case_id} contains unknown "
                        f"metric {metric_id!r}"
                    )
                elif expected_support_id != support_id:
                    add(
                        f"{declaration['file']} {case_id} metric {metric_id!r} belongs to "
                        f"support {expected_support_id!r}, not {support_id!r}"
                    )
                if metric_id in per_case_values and is_number(value):
                    per_case_values[metric_id].append(float(value))
        missing_supports = sorted(set(expected_for_case) - observed_support_ids)
        if missing_supports:
            add(f"{declaration['file']} {case_id} is missing scoring supports: {missing_supports}")

    for metric_id, values in per_case_values.items():
        if len(values) != len(split_case_ids):
            add(f"{declaration['file']} metric {metric_id!r} must have one value per test case")
            continue
        binding = bound_metric_bindings.get(metric_id, {})
        aggregation = binding.get("aggregation")
        reduction = binding.get("reduction")
        if aggregation == "per_geometry_then_macro_average":
            expected_value = math.fsum(values) / len(values)
            reduction_label = "macro-average"
        elif aggregation == "all_test_cases" and reduction == "mae":
            expected_value = math.fsum(values) / len(values)
            reduction_label = "equal-case MAE reduction"
        elif aggregation == "all_test_cases" and reduction == "rmse":
            expected_value = math.sqrt(
                math.fsum(value * value for value in values) / len(values)
            )
            reduction_label = "equal-case RMSE reduction"
        else:
            continue
        submitted_value = submission.get("metric_values", {}).get(metric_id)
        if is_number(submitted_value) and not math.isclose(
            submitted_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            add(
                f"metric_values.{metric_id} must equal the {reduction_label} of its "
                "submitted per-case values"
            )
    if submission.get("dataset_id") == "hiliftaeroml":
        _validate_hiliftaeroml_force_metric_bindings(
            add,
            submission=submission,
            case_metrics=case_metrics,
        )
    if submission.get("dataset_id") == "drivaerml" and not surface_only:
        try:
            validate_schema_v3_candidate_nonspatial_metrics(case_metrics)
        except DrivAerDatasetScorerError as error:
            add(
                f"{declaration['file']} DrivAerML nonspatial validation failed: "
                f"{error}"
            )
    return case_metrics


def validate_v3_discretization(
    add: Any,
    directory: Path,
    submission: dict[str, Any],
    split_case_ids: list[str],
    support_manifest: dict[str, Any] | None,
    support_case_index: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Validate the spatial summary and ordered per-evaluation-case JSONL records."""

    surface_only = (
        submission.get("dataset_id") == "drivaerml"
        and submission.get("prediction_scope") == "surface_only"
    )

    declaration = submission["spatial_discretization"]
    path = safe_submission_path(
        add,
        directory,
        declaration["file"],
        label="spatial_discretization.file",
    )
    if path is None:
        return None, []
    if not path.is_file():
        add(
            f"missing discretization summary: "
            f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"
        )
        return None, []
    if sha256_file(path) != declaration["sha256"]:
        add("discretization.json does not match spatial_discretization.sha256")
    try:
        summary = load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        add(f"cannot read discretization summary: {error}")
        return None, []
    for error in schema_errors(summary, "discretization.schema.json", schema_version="v3"):
        add(f"{declaration['file']} {error}")
    identities = {
        "submission_id": submission["submission_id"],
        "dataset_id": submission["dataset_id"],
        "split_id": submission["split_id"],
        "scoring_support_release_id": submission["scoring_support"]["release_id"],
        "scoring_support_manifest_sha256": submission["scoring_support"]["manifest_sha256"],
    }
    for key, expected in identities.items():
        if summary.get(key) != expected:
            add(f"{declaration['file']} {key} must equal {expected!r}")

    def validate_representation(label: str, representation: Any) -> None:
        if not isinstance(representation, dict) or representation.get("used") is not True:
            return
        entity_counts = representation.get("entity_counts", [])
        entities = [
            entry.get("entity")
            for entry in entity_counts
            if isinstance(entry, dict)
        ]
        if len(entities) != len(set(entities)):
            add(f"{declaration['file']} {label} entity count names must be unique")
        for entry in entity_counts:
            if not isinstance(entry, dict):
                continue
            count = entry.get("count", {})
            if count.get("kind") == "per_case":
                ordered_values = (
                    count.get("minimum"),
                    count.get("median"),
                    count.get("maximum"),
                )
                if all(is_number(value) for value in ordered_values) and not (
                    ordered_values[0] <= ordered_values[1] <= ordered_values[2]
                ):
                    add(
                        f"{declaration['file']} {label}/{entry.get('entity')} requires "
                        "minimum <= median <= maximum"
                    )
        native = representation.get("native_comparison", {})
        if native.get("status") == "reported":
            native_entities = [
                entry.get("entity")
                for entry in native.get("native_entity_counts", [])
                if isinstance(entry, dict)
            ]
            fraction_entities = [
                entry.get("entity")
                for entry in native.get("fractions", [])
                if isinstance(entry, dict)
            ]
            if len(native_entities) != len(set(native_entities)):
                add(f"{declaration['file']} {label} native entity names must be unique")
            if len(fraction_entities) != len(set(fraction_entities)):
                add(f"{declaration['file']} {label} native fraction entity names must be unique")
            if set(native_entities) != set(entities) or set(fraction_entities) != set(entities):
                add(
                    f"{declaration['file']} {label} native counts and fractions "
                    "must cover exactly the reported entity count names"
                )
            for fraction_entry in native.get("fractions", []):
                if not isinstance(fraction_entry, dict):
                    continue
                fraction = fraction_entry.get("fraction", {})
                ordered_values = (
                    fraction.get("minimum"),
                    fraction.get("median"),
                    fraction.get("maximum"),
                )
                if (
                    fraction.get("kind") == "per_case"
                    and all(is_number(value) for value in ordered_values)
                    and not (ordered_values[0] <= ordered_values[1] <= ordered_values[2])
                ):
                    add(
                        f"{declaration['file']} {label}/{fraction_entry.get('entity')} native "
                        "fraction requires minimum <= median <= maximum"
                    )
            counts_by_entity = {
                entry.get("entity"): entry.get("count", {})
                for entry in entity_counts
                if isinstance(entry, dict)
            }
            native_counts_by_entity = {
                entry.get("entity"): entry.get("count", {})
                for entry in native.get("native_entity_counts", [])
                if isinstance(entry, dict)
            }
            fractions_by_entity = {
                entry.get("entity"): entry.get("fraction", {})
                for entry in native.get("fractions", [])
                if isinstance(entry, dict)
            }
            for entity in set(entities):
                count = counts_by_entity.get(entity, {})
                native_count = native_counts_by_entity.get(entity, {})
                fraction = fractions_by_entity.get(entity, {})
                if (
                    count.get("kind") == "fixed"
                    and native_count.get("kind") == "fixed"
                    and fraction.get("kind") == "fixed"
                    and native_count.get("value")
                ):
                    expected_fraction = count["value"] / native_count["value"]
                    if not math.isclose(
                        fraction.get("value", math.nan),
                        expected_fraction,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        add(
                            f"{declaration['file']} {label}/{entity} native fraction "
                            "must equal count/native_count"
                        )
        validate_domain_semantics(
            add,
            representation.get("domain"),
            label=f"{declaration['file']} {label}",
        )

    for name, representation in summary.get("training", {}).items():
        validate_representation(f"training.{name}", representation)
    inference_summary = summary.get("inference", {})
    validate_representation("inference.surface_input", inference_summary.get("surface_input"))
    validate_representation("inference.volume_input", inference_summary.get("volume_input"))
    summary_inputs: dict[str, dict[str, Any]] = {}
    for input_name in ("surface_input", "volume_input"):
        representation = inference_summary.get(input_name, {})
        if not isinstance(representation, dict) or representation.get("used") is not True:
            continue
        case_record_id = representation.get("case_record_id")
        if not isinstance(case_record_id, str) or not case_record_id:
            add(
                f"{declaration['file']} inference.{input_name}.case_record_id "
                "is required when the input is used"
            )
            continue
        if case_record_id in summary_inputs:
            add(f"{declaration['file']} inference case-record input IDs must be unique")
        summary_inputs[case_record_id] = representation
    summary_direct_outputs = inference_summary.get("direct_outputs", [])
    summary_output_ids = [
        output.get("id") for output in summary_direct_outputs if isinstance(output, dict)
    ]
    if len(summary_output_ids) != len(set(summary_output_ids)):
        add(f"{declaration['file']} inference direct-output IDs must be unique")
    for output in summary_direct_outputs:
        if isinstance(output, dict):
            validate_representation(
                f"inference.direct_outputs[{output.get('id')}]",
                output.get("representation"),
            )
    summary_mappings = inference_summary.get("mappings", [])
    summary_mapping_supports = [
        mapping.get("support_id") for mapping in summary_mappings if isinstance(mapping, dict)
    ]
    if len(summary_mapping_supports) != len(set(summary_mapping_supports)):
        add(f"{declaration['file']} inference mapping support IDs must be unique")
    official_supports = {
        support.get("id"): support
        for support in (support_manifest or {}).get("supports", [])
        if isinstance(support, dict) and isinstance(support.get("id"), str)
    }
    if surface_only:
        official_supports.pop("volume_native_cells", None)
    for mapping in summary_mappings:
        if not isinstance(mapping, dict):
            continue
        support_id = mapping.get("support_id")
        source_output_id = mapping.get("source_output_id")
        official = official_supports.get(support_id)
        if official is None:
            add(f"{declaration['file']} mapping references unknown support_id {support_id!r}")
        elif mapping.get("extrapolation_policy") != official.get("extrapolation_policy"):
            add(
                f"{declaration['file']} mapping {support_id!r} extrapolation policy must "
                "match the official support"
            )
        if source_output_id not in summary_output_ids:
            add(
                f"{declaration['file']} mapping {support_id!r} source_output_id is not a "
                "declared direct output"
            )
        if mapping.get("unmapped_fraction") != 0:
            add(f"{declaration['file']} mapping {support_id!r} requires unmapped_fraction=0")
        if (
            mapping.get("extrapolation_policy") == "forbidden"
            and mapping.get("extrapolated_fraction") != 0
        ):
            add(f"{declaration['file']} mapping {support_id!r} forbids extrapolated values")
    missing_summary_mappings = sorted(set(official_supports) - set(summary_mapping_supports))
    if missing_summary_mappings:
        add(
            f"{declaration['file']} inference is missing mappings for official supports: "
            f"{missing_summary_mappings}"
        )

    case_manifest = summary.get("case_manifest", {})
    case_path = safe_submission_path(
        add,
        directory,
        case_manifest.get("file"),
        label="discretization case_manifest.file",
    )
    if case_path is None:
        return summary, []
    if not case_path.is_file():
        add(
            f"missing discretization case records: "
            f"{case_path.relative_to(ROOT) if case_path.is_relative_to(ROOT) else case_path}"
        )
        return summary, []
    if sha256_file(case_path) != case_manifest.get("sha256"):
        add("discretization case records do not match case_manifest.sha256")
    records = load_json_lines(add, case_path, label=case_manifest.get("file", "discretization cases"))
    official_support_ids = {
        support.get("id")
        for support in (support_manifest or {}).get("supports", [])
        if isinstance(support, dict) and isinstance(support.get("id"), str)
    }
    if surface_only:
        official_support_ids.discard("volume_native_cells")
    extrapolation_policies = {
        support.get("id"): support.get("extrapolation_policy")
        for support in (support_manifest or {}).get("supports", [])
        if isinstance(support, dict) and isinstance(support.get("id"), str)
    }
    expected_support_counts = {
        case.get("case_id"): {
            instance.get("support_id"): instance.get("entity_count")
            for instance in case.get("support_instances", [])
            if isinstance(instance, dict)
        }
        for case in (support_case_index or {}).get("_loaded_cases", [])
        if isinstance(case, dict)
    }
    if surface_only:
        expected_support_counts = {
            case_id: {
                support_id: count
                for support_id, count in counts.items()
                if support_id != "volume_native_cells"
            }
            for case_id, counts in expected_support_counts.items()
        }
    for line_number, record in enumerate(records, start=1):
        for error in schema_errors(record, "discretization-case.schema.json", schema_version="v3"):
            add(f"{case_manifest.get('file')} line {line_number} {error}")
        record_identities = {
            "submission_id": submission["submission_id"],
            "dataset_id": submission["dataset_id"],
            "split_id": submission["split_id"],
        }
        for key, expected in record_identities.items():
            if record.get(key) != expected:
                add(f"{case_manifest.get('file')} line {line_number} {key} must equal {expected!r}")
        inference = record.get("inference", {})
        inputs = inference.get("inputs", [])
        direct_outputs = inference.get("direct_outputs", [])
        input_ids = [item.get("id") for item in inputs if isinstance(item, dict)]
        output_ids = [item.get("id") for item in direct_outputs if isinstance(item, dict)]
        if len(input_ids) != len(set(input_ids)):
            add(f"{case_manifest.get('file')} {record.get('case_id')} inference input IDs must be unique")
        if len(output_ids) != len(set(output_ids)):
            add(
                f"{case_manifest.get('file')} {record.get('case_id')} direct-output IDs must be unique"
            )
        if input_ids != list(summary_inputs):
            add(
                f"{case_manifest.get('file')} {record.get('case_id')} inference input IDs "
                "must exactly match the used summary inputs in declared order"
            )
        if output_ids != summary_output_ids:
            add(
                f"{case_manifest.get('file')} {record.get('case_id')} direct-output IDs "
                "must exactly match the summary outputs in declared order"
            )
        for item in [*inputs, *direct_outputs]:
            if not isinstance(item, dict):
                continue
            validate_domain_semantics(
                add,
                item.get("domain"),
                label=f"{case_manifest.get('file')} {record.get('case_id')}/{item.get('id')}",
            )
        case_mappings = inference.get("mappings", [])
        case_mapping_supports = [
            mapping.get("support_id")
            for mapping in case_mappings
            if isinstance(mapping, dict)
        ]
        if len(case_mapping_supports) != len(set(case_mapping_supports)):
            add(
                f"{case_manifest.get('file')} {record.get('case_id')} mapping "
                "support IDs must be unique"
            )
        if set(case_mapping_supports) != official_support_ids:
            add(
                f"{case_manifest.get('file')} {record.get('case_id')} mappings "
                "must cover exactly the official supports"
            )
        for mapping in case_mappings:
            if not isinstance(mapping, dict):
                continue
            label = f"{record.get('case_id')}/{mapping.get('support_id')}"
            support_id = mapping.get("support_id")
            if support_id not in official_support_ids:
                add(f"{case_manifest.get('file')} {label} references an unknown official support")
            if mapping.get("source_output_id") not in output_ids:
                add(f"{case_manifest.get('file')} {label} source_output_id is not a direct output")
            support_count = mapping.get("support_count")
            scored = mapping.get("scored_count")
            unmapped = mapping.get("unmapped_count")
            extrapolated = mapping.get("extrapolated_count")
            coverage = mapping.get("final_coverage_fraction")
            expected_count = expected_support_counts.get(record.get("case_id"), {}).get(support_id)
            if expected_count is not None and support_count != expected_count:
                add(f"{case_manifest.get('file')} {label} support_count must match the official case support")
            if scored != support_count:
                add(f"{case_manifest.get('file')} {label} scored_count must equal support_count")
            if unmapped != 0:
                add(f"{case_manifest.get('file')} {label} requires unmapped_count=0")
            if not is_number(coverage) or not math.isclose(coverage, 1.0, rel_tol=0.0, abs_tol=1e-12):
                add(f"{case_manifest.get('file')} {label} requires final_coverage_fraction=1.0")
            if isinstance(scored, int) and isinstance(unmapped, int) and scored < 0:
                add(f"{case_manifest.get('file')} {label} scored_count cannot be negative")
            if isinstance(extrapolated, int) and extrapolated < 0:
                add(f"{case_manifest.get('file')} {label} extrapolated_count cannot be negative")
            if extrapolation_policies.get(support_id) == "forbidden" and extrapolated != 0:
                add(f"{case_manifest.get('file')} {label} forbids extrapolated scoring values")

    case_ids = [record.get("case_id") for record in records]
    if case_manifest.get("case_count") != len(split_case_ids):
        add(f"discretization case_manifest.case_count must be {len(split_case_ids)} for this split")
    if len(records) != case_manifest.get("case_count"):
        add("discretization case record count does not match case_manifest.case_count")
    if case_ids != split_case_ids:
        missing = sorted(set(split_case_ids) - set(case_ids))
        unexpected = sorted(set(case_ids) - set(split_case_ids))
        add(
            "discretization case order differs from the benchmark split; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    def validate_case_representation_counts(
        representation_id: str,
        representation: dict[str, Any],
        *,
        collection: str,
        description: str,
    ) -> None:
        expected_entities = [
            entry.get("entity")
            for entry in representation.get("entity_counts", [])
            if isinstance(entry, dict)
        ]
        values_by_entity: dict[str, list[int]] = {
            entity: [] for entity in expected_entities if isinstance(entity, str)
        }
        native = representation.get("native_comparison", {})
        native_reported = native.get("status") == "reported"
        native_values_by_entity: dict[str, list[int]] = {
            entity: [] for entity in expected_entities if isinstance(entity, str)
        }
        fraction_values_by_entity: dict[str, list[float]] = {
            entity: [] for entity in expected_entities if isinstance(entity, str)
        }
        matched_cases = 0
        for record in records:
            matching = next(
                (
                    item
                    for item in record.get("inference", {}).get(collection, [])
                    if isinstance(item, dict) and item.get("id") == representation_id
                ),
                None,
            )
            if matching is None:
                continue
            matched_cases += 1
            observed_entities = [
                entry.get("entity")
                for entry in matching.get("entity_counts", [])
                if isinstance(entry, dict)
            ]
            if observed_entities != expected_entities:
                add(
                    f"{case_manifest.get('file')} {record.get('case_id')} "
                    f"{description} {representation_id!r} entity IDs must exactly "
                    "match the summary in declared order"
                )
            for entity in values_by_entity:
                matching_entity = next(
                    (
                        entry
                        for entry in matching.get("entity_counts", [])
                        if isinstance(entry, dict) and entry.get("entity") == entity
                    ),
                    None,
                )
                if isinstance(matching_entity, dict) and isinstance(
                    matching_entity.get("count"), int
                ):
                    values_by_entity[entity].append(matching_entity["count"])
            observed_native_entities = [
                entry.get("entity")
                for entry in matching.get("native_entity_counts", [])
                if isinstance(entry, dict)
            ]
            observed_fraction_entities = [
                entry.get("entity")
                for entry in matching.get("native_fractions", [])
                if isinstance(entry, dict)
            ]
            if not native_reported:
                if observed_native_entities or observed_fraction_entities:
                    add(
                        f"{case_manifest.get('file')} {record.get('case_id')} "
                        f"{description} {representation_id!r} native counts and fractions "
                        "must be omitted unless the summary reports a native comparison"
                    )
                continue
            if (
                observed_native_entities != expected_entities
                or observed_fraction_entities != expected_entities
            ):
                add(
                    f"{case_manifest.get('file')} {record.get('case_id')} "
                    f"{description} {representation_id!r} native entity and fraction IDs "
                    "must exactly match the summary in declared order"
                )
            for entity in native_values_by_entity:
                native_entry = next(
                    (
                        entry
                        for entry in matching.get("native_entity_counts", [])
                        if isinstance(entry, dict) and entry.get("entity") == entity
                    ),
                    None,
                )
                fraction_entry = next(
                    (
                        entry
                        for entry in matching.get("native_fractions", [])
                        if isinstance(entry, dict) and entry.get("entity") == entity
                    ),
                    None,
                )
                count_entry = next(
                    (
                        entry
                        for entry in matching.get("entity_counts", [])
                        if isinstance(entry, dict) and entry.get("entity") == entity
                    ),
                    None,
                )
                native_count = (
                    native_entry.get("count")
                    if isinstance(native_entry, dict)
                    else None
                )
                fraction = (
                    fraction_entry.get("fraction")
                    if isinstance(fraction_entry, dict)
                    else None
                )
                reported_count = (
                    count_entry.get("count")
                    if isinstance(count_entry, dict)
                    else None
                )
                if isinstance(native_count, int):
                    native_values_by_entity[entity].append(native_count)
                if is_number(fraction):
                    fraction_values_by_entity[entity].append(float(fraction))
                if (
                    isinstance(reported_count, int)
                    and isinstance(native_count, int)
                    and native_count > 0
                    and is_number(fraction)
                    and not math.isclose(
                        float(fraction),
                        reported_count / native_count,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    add(
                        f"{case_manifest.get('file')} {record.get('case_id')} "
                        f"{description} {representation_id!r}/{entity!r} native fraction "
                        "must equal count/native_count"
                    )
        if matched_cases != len(records):
            add(
                f"{case_manifest.get('file')} {description} {representation_id!r} "
                "must be reported for every case"
            )

        def validate_summaries(
            entries: Any,
            *,
            value_key: str,
            observed: dict[str, list[int]] | dict[str, list[float]],
            value_label: str,
        ) -> None:
            for entity_entry in entries:
                if not isinstance(entity_entry, dict):
                    continue
                entity = entity_entry.get("entity")
                values = observed.get(entity, [])
                if len(values) != len(records):
                    add(
                        f"{case_manifest.get('file')} {description} "
                        f"{representation_id!r}/{entity!r} {value_label} "
                        "must be reported for every case"
                    )
                    continue
                value_summary = entity_entry.get(value_key, {})
                if value_summary.get("kind") == "fixed":
                    if any(
                        not math.isclose(
                            float(value),
                            float(value_summary.get("value", math.nan)),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        for value in values
                    ):
                        add(
                            f"{declaration['file']} {description} "
                            f"{representation_id!r}/{entity!r} fixed {value_label} "
                            "does not match case records"
                        )
                elif value_summary.get("kind") == "per_case":
                    calculated = {
                        "minimum": min(values),
                        "median": median(values),
                        "maximum": max(values),
                    }
                    if any(
                        not is_number(value_summary.get(key))
                        or not math.isclose(
                            float(value_summary[key]),
                            float(value),
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        for key, value in calculated.items()
                    ):
                        add(
                            f"{declaration['file']} {description} "
                            f"{representation_id!r}/{entity!r} per-case {value_label} "
                            "summary does not match case records"
                        )

        validate_summaries(
            representation.get("entity_counts", []),
            value_key="count",
            observed=values_by_entity,
            value_label="count",
        )
        if native_reported:
            validate_summaries(
                native.get("native_entity_counts", []),
                value_key="count",
                observed=native_values_by_entity,
                value_label="native count",
            )
            validate_summaries(
                native.get("fractions", []),
                value_key="fraction",
                observed=fraction_values_by_entity,
                value_label="native fraction",
            )

    for input_id, representation in summary_inputs.items():
        validate_case_representation_counts(
            input_id,
            representation,
            collection="inputs",
            description="inference input",
        )
    for output in summary_direct_outputs:
        if not isinstance(output, dict):
            continue
        validate_case_representation_counts(
            output.get("id"),
            output.get("representation", {}),
            collection="direct_outputs",
            description="direct output",
        )
    return summary, records


def validate_v3_prediction_metadata(
    add: Any,
    directory: Path,
    submission: dict[str, Any],
    split_case_ids: list[str],
    *,
    contributor_stage: bool,
    candidate_dry_run: bool = False,
    case_metrics: dict[str, Any] | None = None,
    dataset_spec: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate optional prediction declarations/checks without downloading artifacts."""

    artifacts = submission.get("prediction_artifacts", [])
    artifact_ids = [artifact.get("artifact_id") for artifact in artifacts if isinstance(artifact, dict)]
    if len(artifact_ids) != len(set(artifact_ids)):
        add("prediction_artifacts artifact_id values must be unique")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        label = f"prediction_artifacts[{artifact.get('artifact_id')!r}]"
        if artifact.get("support_release_id") != submission["scoring_support"]["release_id"]:
            add(f"{label}.support_release_id must match scoring_support.release_id")
        if artifact.get("support_manifest_sha256") != submission["scoring_support"]["manifest_sha256"]:
            add(f"{label}.support_manifest_sha256 must match scoring_support.manifest_sha256")
        if artifact.get("split_id") != submission["split_id"]:
            add(f"{label}.split_id must match submission split_id")
        coverage = artifact.get("coverage", {})
        if coverage.get("expected_case_count") != len(split_case_ids):
            add(f"{label}.coverage.expected_case_count must match the benchmark split")
        case_ids = coverage.get("case_ids")
        if case_ids is not None:
            expected_subset_order = [case_id for case_id in split_case_ids if case_id in set(case_ids)]
            if case_ids != expected_subset_order:
                add(f"{label}.coverage.case_ids must be a unique subset in benchmark split order")
            if coverage.get("case_count") != len(case_ids):
                add(f"{label}.coverage.case_count must equal the number of listed case_ids")
        if coverage.get("kind") == "complete_split":
            if coverage.get("case_count") != len(split_case_ids):
                add(f"{label} complete_split coverage must include every benchmark case")
            if case_ids is not None and case_ids != split_case_ids:
                add(f"{label} complete_split case_ids must match the exact benchmark split")

    checks_path = directory / "prediction-artifact-checks.json"
    if not checks_path.exists():
        return None
    if contributor_stage or candidate_dry_run:
        add(
            "candidate dry-run packages must not contain "
            "prediction-artifact-checks.json"
            if candidate_dry_run
            else "contributors must not add prediction-artifact-checks.json"
        )
        return None
    try:
        checks = load_json(checks_path)
    except (OSError, json.JSONDecodeError) as error:
        add(f"cannot read prediction-artifact-checks.json: {error}")
        return None
    for error in schema_errors(
        checks,
        "prediction-artifact-checks.schema.json",
        schema_version="v3",
    ):
        add(f"prediction-artifact-checks.json {error}")
    if checks.get("submission_id") != submission["submission_id"]:
        add("prediction-artifact-checks.json submission_id must match submission.json")
    declared_by_id = {
        artifact["artifact_id"]: artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and isinstance(artifact.get("artifact_id"), str)
    }
    checked_ids: set[str] = set()
    for check in checks.get("checks", []):
        if not isinstance(check, dict):
            continue
        artifact_id = check.get("artifact_id")
        if artifact_id in checked_ids:
            add(f"prediction-artifact-checks.json contains duplicate artifact_id {artifact_id!r}")
        checked_ids.add(artifact_id)
        declared = declared_by_id.get(artifact_id)
        if declared is None:
            add(f"prediction-artifact-checks.json references undeclared artifact_id {artifact_id!r}")
            continue
        if check.get("repository_revision") != declared.get("revision"):
            add(f"prediction check {artifact_id!r} repository_revision must match the declaration")
        if check.get("manifest_sha256") != declared.get("manifest_sha256"):
            add(f"prediction check {artifact_id!r} manifest_sha256 must match the declaration")
        if check.get("expected_case_count") != len(split_case_ids):
            add(f"prediction check {artifact_id!r} expected_case_count must match the benchmark split")
        checked_case_count = check.get("checked_case_count")
        if isinstance(checked_case_count, int) and checked_case_count > len(split_case_ids):
            add(f"prediction check {artifact_id!r} checked_case_count exceeds the benchmark split")
        recomputed_case_count = check.get("recomputed_case_count")
        if (
            isinstance(recomputed_case_count, int)
            and isinstance(checked_case_count, int)
            and recomputed_case_count > checked_case_count
        ):
            add(
                f"prediction check {artifact_id!r} recomputed_case_count "
                "cannot exceed checked_case_count"
            )
        recomputation = check.get("metric_recomputation")
        status = check.get("status")
        if recomputation == "not_performed":
            if recomputed_case_count != 0:
                add(
                    f"prediction check {artifact_id!r} not_performed requires "
                    "recomputed_case_count=0"
                )
            if status == "metrics_recomputed":
                add(
                    f"prediction check {artifact_id!r} status metrics_recomputed "
                    "requires partial or performed metric_recomputation"
                )
        elif recomputation == "partial":
            if (
                not isinstance(recomputed_case_count, int)
                or not 0 < recomputed_case_count < len(split_case_ids)
            ):
                add(
                    f"prediction check {artifact_id!r} partial metric recomputation "
                    "requires 1..expected_case_count-1 recomputed cases"
                )
            if status != "metrics_recomputed":
                add(
                    f"prediction check {artifact_id!r} partial metric recomputation "
                    "requires status metrics_recomputed"
                )
        elif recomputation == "performed":
            if recomputed_case_count != len(split_case_ids):
                add(
                    f"prediction check {artifact_id!r} performed metric recomputation "
                    "requires the complete benchmark split"
                )
            if status != "metrics_recomputed":
                add(
                    f"prediction check {artifact_id!r} performed metric recomputation "
                    "requires status metrics_recomputed"
                )
            if declared.get("coverage", {}).get("kind") != "complete_split":
                add(
                    f"prediction check {artifact_id!r} performed metric recomputation "
                    "requires a complete_split prediction artifact"
                )
    if (
        submission.get("dataset_id") == "drivaerml"
        and "dataset_evaluator_recomputation" in checks
    ):
        _validate_drivaerml_native_evaluator_recomputation(
            add,
            submission=submission,
            split_case_ids=split_case_ids,
            checks=checks,
            case_metrics=case_metrics,
            dataset_spec=dataset_spec,
        )
    return checks


def _validate_drivaerml_native_evaluator_recomputation(
    add: Any,
    *,
    submission: dict[str, Any],
    split_case_ids: list[str],
    checks: dict[str, Any],
    case_metrics: dict[str, Any] | None,
    dataset_spec: dict[str, Any] | None,
) -> None:
    """Bind DrivAerML nonspatial values to a maintainer evaluator replay."""

    receipt = checks.get("dataset_evaluator_recomputation")
    if not isinstance(receipt, dict):
        add(
            "prediction-artifact-checks.json requires "
            "dataset_evaluator_recomputation for DrivAerML"
        )
        return
    if case_metrics is None:
        add(
            "DrivAerML dataset_evaluator_recomputation requires valid "
            "case-metrics evidence"
        )
        return
    try:
        nonspatial = validate_schema_v3_candidate_nonspatial_metrics(case_metrics)
    except DrivAerDatasetScorerError as error:
        add(
            "DrivAerML dataset_evaluator_recomputation cannot bind invalid "
            f"nonspatial values: {error}"
        )
        return

    expected_identity = {
        "dataset_id": "drivaerml",
        "status": "complete_native_evaluator_recomputation",
        "evaluator_reference_version": submission.get("evaluation", {}).get(
            "reference_version"
        ),
        "case_count": len(split_case_ids),
        "case_metrics_sha256": submission.get("case_metrics", {}).get("sha256"),
        "nonspatial_metric_ids": nonspatial["metric_ids"],
        "nonspatial_values_sha256": nonspatial["nonspatial_values_sha256"],
    }
    for key, expected in expected_identity.items():
        if receipt.get(key) != expected:
            add(
                "prediction-artifact-checks.json "
                f"dataset_evaluator_recomputation.{key} must equal {expected!r}"
            )

    scoring_support = (
        dataset_spec.get("scoring_support")
        if isinstance(dataset_spec, dict)
        else None
    )
    evaluator_binding = (
        scoring_support.get("dataset_evaluator_binding")
        if isinstance(scoring_support, dict)
        else None
    )
    if not isinstance(evaluator_binding, dict):
        add(
            "DrivAerML final validation requires a benchmark-owned "
            "scoring_support.dataset_evaluator_binding"
        )
    elif evaluator_binding.get("status") != "frozen":
        add(
            "DrivAerML final validation requires "
            "scoring_support.dataset_evaluator_binding.status='frozen'; "
            "the candidate evaluator revision is not frozen"
        )
    else:
        frozen_reference_version = evaluator_binding.get(
            "evaluator_reference_version"
        )
        frozen_code_revision = evaluator_binding.get("evaluator_code_revision")
        valid_code_revision = (
            isinstance(frozen_code_revision, str)
            and len(frozen_code_revision) in {40, 64}
            and all(
                character in "0123456789abcdef"
                for character in frozen_code_revision
            )
        )
        if (
            not isinstance(frozen_reference_version, str)
            or not frozen_reference_version
            or not valid_code_revision
        ):
            add(
                "DrivAerML frozen dataset_evaluator_binding requires a non-empty "
                "evaluator_reference_version and an immutable 40- or 64-character "
                "lowercase hexadecimal evaluator_code_revision"
            )
        else:
            if frozen_reference_version != submission.get("evaluation", {}).get(
                "reference_version"
            ):
                add(
                    "DrivAerML frozen dataset_evaluator_binding reference version "
                    "must match submission evaluation.reference_version"
                )
            if receipt.get("evaluator_reference_version") != frozen_reference_version:
                add(
                    "prediction-artifact-checks.json "
                    "dataset_evaluator_recomputation.evaluator_reference_version "
                    "must match the benchmark-owned frozen evaluator binding"
                )
            if receipt.get("evaluator_code_revision") != frozen_code_revision:
                add(
                    "prediction-artifact-checks.json "
                    "dataset_evaluator_recomputation.evaluator_code_revision must "
                    "match the benchmark-owned frozen evaluator binding"
                )

    declared = {
        artifact.get("artifact_id"): artifact
        for artifact in submission.get("prediction_artifacts", [])
        if isinstance(artifact, dict)
        and isinstance(artifact.get("artifact_id"), str)
    }
    expected_artifact_ids = sorted(
        artifact_id
        for artifact_id, artifact in declared.items()
        if artifact.get("kind") == "scored_predictions"
        and artifact.get("coverage", {}).get("kind") == "complete_split"
    )
    if len(expected_artifact_ids) != 1:
        add(
            "DrivAerML native-evaluator recomputation requires exactly one "
            "complete_split scored_predictions artifact"
        )
    if receipt.get("prediction_artifact_ids") != expected_artifact_ids:
        add(
            "prediction-artifact-checks.json "
            "dataset_evaluator_recomputation.prediction_artifact_ids must list "
            "every complete_split scored_predictions artifact exactly once in "
            "artifact_id order"
        )

    checks_by_id = {
        check.get("artifact_id"): check
        for check in checks.get("checks", [])
        if isinstance(check, dict) and isinstance(check.get("artifact_id"), str)
    }
    for artifact_id in expected_artifact_ids:
        check = checks_by_id.get(artifact_id)
        if not isinstance(check, dict):
            add(
                f"DrivAerML native-evaluator receipt is missing prediction check "
                f"{artifact_id!r}"
            )
            continue
        required = {
            "status": "metrics_recomputed",
            "metric_recomputation": "performed",
            "checked_case_count": len(split_case_ids),
            "recomputed_case_count": len(split_case_ids),
            "expected_case_count": len(split_case_ids),
        }
        for key, expected in required.items():
            if check.get(key) != expected:
                add(
                    f"DrivAerML prediction check {artifact_id!r} {key} must "
                    f"equal {expected!r}"
                )


def validate_drivaerml_maintainer_receipt_hash(
    add: Any,
    directory: Path,
    validation: dict[str, Any],
) -> None:
    """Bind optional DrivAerML prediction checks when maintainers add them."""

    checks_path = directory / "prediction-artifact-checks.json"
    declared = validation.get("prediction_artifact_checks_sha256")
    if not checks_path.is_file():
        if declared is not None:
            add(
                "maintainer-validation.json prediction_artifact_checks_sha256 "
                "must be absent when prediction-artifact-checks.json is absent"
            )
        return
    expected = sha256_file(checks_path)
    if not isinstance(declared, str) or declared != expected:
        add(
            "maintainer-validation.json prediction_artifact_checks_sha256 "
            "must bind the optional DrivAerML prediction-artifact checks"
        )


def _validate_hiliftaeroml_dataset_evaluator_binding(
    add: Any,
    *,
    submission: dict[str, Any],
    evidence: dict[str, Any],
    dataset_spec: dict[str, Any] | None,
) -> None:
    """Bind HiLift evidence to dataset-owned evaluator code, not participant code."""

    profile_data = submission.get("profile_data")
    compact_profile = (
        isinstance(profile_data, dict)
        and profile_data.get("format") == HILIFT_COMPACT_PROFILE_FORMAT
    )
    submitted_compact_binding = (
        profile_data.get("compact_profile_implementation_binding")
        if isinstance(profile_data, dict)
        else None
    )
    evidence_compact_binding = evidence.get(
        "compact_profile_implementation_binding"
    )
    if compact_profile:
        if submitted_compact_binding != HILIFT_COMPACT_PROFILE_IMPLEMENTATION_BINDING:
            add(
                "submission.json compact profile_data requires the exact unbound "
                "compact_profile_implementation_binding"
            )
        if evidence_compact_binding != HILIFT_COMPACT_PROFILE_IMPLEMENTATION_BINDING:
            add(
                "evaluation-evidence.json compact profile evaluation requires the "
                "exact unbound compact_profile_implementation_binding"
            )
        if submitted_compact_binding != evidence_compact_binding:
            add(
                "evaluation-evidence.json compact_profile_implementation_binding "
                "must exactly match submission.json profile_data"
            )
    elif (
        submitted_compact_binding is not None
        or evidence_compact_binding is not None
    ):
        add(
            "compact_profile_implementation_binding is permitted only for the "
            "HiLiftAeroML compact-v2 profile format"
        )

    declared = evidence.get("dataset_evaluator_binding")
    if not isinstance(declared, dict):
        add(
            "evaluation-evidence.json HiLiftAeroML requires a structured "
            "dataset_evaluator_binding"
        )
        return
    expected_fields = {"status", "reference_version", "code_revision"}
    if set(declared) != expected_fields:
        add(
            "evaluation-evidence.json dataset_evaluator_binding must contain exactly "
            f"{sorted(expected_fields)}"
        )

    support = (
        dataset_spec.get("scoring_support")
        if isinstance(dataset_spec, dict)
        else None
    )
    owner_binding = (
        support.get("dataset_evaluator_binding")
        if isinstance(support, dict)
        else None
    )
    if not isinstance(owner_binding, dict):
        add("HiLiftAeroML specification has no dataset_evaluator_binding")
        return
    if owner_binding.get("status") != "frozen":
        add(
            "HiLiftAeroML specification dataset_evaluator_binding must be frozen "
            "before candidate package validation"
        )
        return
    expected = {
        "status": "frozen",
        "reference_version": owner_binding.get("evaluator_reference_version"),
        "code_revision": owner_binding.get("evaluator_code_revision"),
    }
    for key, value in expected.items():
        if declared.get(key) != value:
            add(
                "evaluation-evidence.json dataset_evaluator_binding."
                f"{key} must equal the frozen HiLiftAeroML specification value "
                f"{value!r}"
            )
    if declared.get("reference_version") != submission.get("evaluation", {}).get(
        "reference_version"
    ):
        add(
            "evaluation-evidence.json dataset_evaluator_binding.reference_version "
            "must equal submission evaluation.reference_version"
        )


def validate_evaluation_evidence(
    add: Any,
    directory: Path,
    submission: dict[str, Any],
    dataset_spec: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    evaluation = submission["evaluation"]
    filename = evaluation["evidence_file"]
    if Path(filename).name != filename:
        add("evaluation.evidence_file must be a filename in the submission directory")
        return None
    evidence_path = directory / filename
    if not evidence_path.is_file():
        display_path = evidence_path.relative_to(ROOT) if evidence_path.is_relative_to(ROOT) else evidence_path
        add(f"missing evaluation evidence: {display_path}")
        return None
    if sha256_file(evidence_path) != evaluation["evidence_sha256"]:
        add("evaluation-evidence.json does not match evaluation.evidence_sha256")
    try:
        evidence = load_json(evidence_path)
    except (OSError, json.JSONDecodeError) as error:
        add(f"cannot read evaluation evidence: {error}")
        return None
    evidence_schema_directory = {
        "1.0": "v1",
        "2.0": "v2",
        "3.0": "v3",
    }[submission["schema_version"]]
    for error in schema_errors(
        evidence,
        "evaluation-evidence.schema.json",
        schema_version=evidence_schema_directory,
    ):
        add(f"evaluation-evidence.json {error}")

    identities = {
        "submission_id": submission["submission_id"],
        "dataset_id": submission["dataset_id"],
        "split_id": submission["split_id"],
        "reference_version": evaluation["reference_version"],
        "command": evaluation["command"],
    }
    for key, expected in identities.items():
        if evidence.get(key) != expected:
            add(f"evaluation-evidence.json {key} must equal {expected!r}")
    if submission.get("dataset_id") == "drivaerml":
        expected_scope = submission.get("prediction_scope", "surface_and_volume")
        if evidence.get("prediction_scope", "surface_and_volume") != expected_scope:
            add("evaluation-evidence.json prediction_scope must match submission.json")
    if submission.get("schema_version") == "3.0":
        reproducibility = submission.get("reproducibility", {})
        structured_code = (
            reproducibility.get("code")
            if isinstance(reproducibility, dict)
            else None
        )
        if isinstance(structured_code, dict):
            expected_revision = structured_code.get("commit")
            if evidence.get("code_revision") != expected_revision:
                add(
                    "evaluation-evidence.json code_revision must equal the full "
                    "reproducibility.code.commit"
                )
        elif "code_revision" in evidence:
            add(
                "evaluation-evidence.json code_revision requires "
                "reproducibility.code"
            )
    elif evidence.get("code_revision") != evaluation["code_revision"]:
        add(
            "evaluation-evidence.json code_revision must equal "
            f"{evaluation['code_revision']!r}"
        )
    if submission.get("schema_version") in {"2.0", "3.0"}:
        open_track_identities = {
            "dataset_version": submission["dataset_version"],
            "split_sha256": submission["split_sha256"],
            "case_set_id": submission["case_set_id"],
            "profile_ground_truth_release_id": submission["profile_data"]["profile_ground_truth_release_id"],
            "profile_ground_truth_manifest_sha256": submission["profile_data"][
                "profile_ground_truth_manifest_sha256"
            ],
        }
        for key, expected in open_track_identities.items():
            if evidence.get(key) != expected:
                add(f"evaluation-evidence.json {key} must equal {expected!r}")
    if submission.get("schema_version") == "3.0":
        v3_bindings = {
            "scoring_support_release_id": submission["scoring_support"]["release_id"],
            "scoring_support_manifest_sha256": submission["scoring_support"]["manifest_sha256"],
            "discretization_sha256": submission["spatial_discretization"]["sha256"],
            "case_metrics_sha256": submission["case_metrics"]["sha256"],
        }
        for key, expected in v3_bindings.items():
            if evidence.get(key) != expected:
                add(f"evaluation-evidence.json {key} must equal {expected!r}")
        if submission.get("dataset_id") == "hiliftaeroml":
            _validate_hiliftaeroml_dataset_evaluator_binding(
                add,
                submission=submission,
                evidence=evidence,
                dataset_spec=dataset_spec,
            )
    if evidence.get("metric_values") != submission.get("metric_values"):
        add("evaluation-evidence.json metric_values must exactly match submission.json")

    profile_index_path = directory / submission["profile_data"]["index_file"]
    if profile_index_path.is_file() and evidence.get("profile_index_sha256") != sha256_file(profile_index_path):
        add("evaluation-evidence.json profile_index_sha256 does not match profiles/index.json")
    return evidence


def validate_regional_diagnostics(
    add: Any,
    directory: Path,
    submission: dict[str, Any],
    dataset_spec: dict[str, Any],
    split_case_ids: list[str],
    evidence: dict[str, Any] | None,
) -> None:
    """Validate a dataset-owned zero-weight regional report."""

    contract = dataset_spec.get("regional_diagnostics")
    required = (
        isinstance(contract, dict)
        and contract.get("required_for_new_submissions") is True
    )
    declaration = submission.get("regional_diagnostics")
    if not isinstance(declaration, dict):
        if required:
            add(
                f"{submission.get('dataset_id')} schema-v3 submissions require "
                "regional_diagnostics metadata"
            )
        return
    if not isinstance(contract, dict):
        add("regional_diagnostics is not enabled by the benchmark specification")
        return
    expected_contract = {
        "format": contract.get("format"),
        "contract_sha256": contract.get("contract_sha256"),
        "role": "report_only",
        "weight": 0.0,
        "official_score_changed": False,
    }
    if isinstance(contract.get("definition_id"), str):
        expected_contract["definition_id"] = contract["definition_id"]
    for key, expected in expected_contract.items():
        if declaration.get(key) != expected:
            add(f"regional_diagnostics.{key} must equal {expected!r}")
    if declaration.get("case_count") != len(split_case_ids):
        add("regional_diagnostics.case_count must match the selected split")
    filename = declaration.get("file")
    if filename != "regional-diagnostics.json":
        add("regional_diagnostics.file must be regional-diagnostics.json")
        return
    report_path = directory / filename
    if not report_path.is_file():
        add("missing regional-diagnostics.json")
        return
    digest = sha256_file(report_path)
    if declaration.get("sha256") != digest:
        add("regional-diagnostics.json does not match regional_diagnostics.sha256")
    if evidence is not None and evidence.get("regional_diagnostics_sha256") != digest:
        add(
            "evaluation-evidence.json regional_diagnostics_sha256 must match "
            "regional-diagnostics.json"
        )
    try:
        report = load_json(report_path)
        regional_format = contract.get("format")
        if regional_format == AGGREGATE_REGIONAL_REPORT_SCHEMA:
            if contract.get("contract_sha256") != REGIONAL_DIAGNOSTICS_CONTRACT_SHA256:
                raise RegionalAggregateError(
                    "DrivAerML regional contract SHA-256 differs"
                )
            validate_aggregate_regional_diagnostics(
                report,
                expected_case_ids=split_case_ids,
            )
        elif regional_format == HILIFT_AGGREGATE_REGIONAL_REPORT_SCHEMA:
            if (
                contract.get("contract_sha256")
                != HILIFT_REGIONAL_DIAGNOSTICS_CONTRACT_SHA256
                or contract.get("definition_id") != HILIFT_REGIONAL_DEFINITION_ID
            ):
                raise HiLiftRegionalAggregateError(
                    "HiLiftAeroML regional contract identity differs"
                )
            validate_hilift_aggregate_regional_diagnostics(
                report,
                expected_case_ids=split_case_ids,
                expected_split_id=submission.get("split_id"),
            )
        else:
            raise HiLiftRegionalAggregateError(
                f"unsupported dataset regional format {regional_format!r}"
            )
    except (
        OSError,
        json.JSONDecodeError,
        RegionalAggregateError,
        HiLiftRegionalAggregateError,
    ) as error:
        add(f"regional-diagnostics.json is invalid: {error}")


def validate_open_reproducibility(
    add: Any,
    directory: Path,
    submission: dict[str, Any],
    evidence: dict[str, Any] | None,
    manifest: dict[str, Any],
    dataset_spec: dict[str, Any],
    split_spec_entry: dict[str, Any],
    *,
    contributor_stage: bool,
    candidate_dry_run: bool = False,
) -> None:
    """Validate declared open artifacts and submitted data without executing a model."""

    if evidence is None:
        return
    evidence_status = evidence.get("status")
    approval = submission.get("approval")
    approval_status = approval.get("status") if isinstance(approval, dict) else None
    reproducibility = submission.get("reproducibility")
    submission_schema_version = submission.get("schema_version")
    methodology = submission.get("methodology")
    methodology_kind = (
        methodology.get("record_kind") if isinstance(methodology, dict) else None
    )
    if submission_schema_version in {"2.0", "3.0"} and (directory / "maintainer-replay.json").exists():
        add("maintainer-replay.json is not part of the open-reproducibility contract")

    if contributor_stage:
        if evidence_status != "submitted_evaluation":
            add("contributor-stage packages require evaluation-evidence.status=submitted_evaluation")
        if submission_schema_version != "3.0":
            add(
                "new contributor-stage submitted_evaluation packages require submission "
                "schema_version=3.0"
            )
        if approval is not None:
            add("contributors must leave approval absent; maintainers add it only after submitted-data validation")
        if (directory / "maintainer-validation.json").exists():
            add("contributors must not add maintainer-validation.json")

    if candidate_dry_run:
        if evidence_status != "submitted_evaluation":
            add(
                "candidate dry-run packages require "
                "evaluation-evidence.status=submitted_evaluation"
            )
        if submission_schema_version != "3.0":
            add("candidate dry-run packages require submission schema_version=3.0")
        if approval is not None:
            add("candidate dry-run packages must not contain approval metadata")
        if (directory / "maintainer-validation.json").exists():
            add("candidate dry-run packages must not contain maintainer-validation.json")

    if evidence_status == "prototype_dummy_data":
        if approval_status != "prototype":
            add("prototype_dummy_data evidence requires approval.status=prototype")
        if reproducibility is not None:
            add("prototype dummy data must not claim the open reproducibility contract")
        if submission.get("schema_version") != "1.0":
            add("prototype dummy data must use the historical submission schema_version=1.0")
        if not isinstance(submission.get("methodology"), dict):
            add("prototype dummy data must include a structured methodology record")
        elif methodology_kind != "prototype_fixture":
            add(
                "prototype dummy data must use "
                "methodology.record_kind='prototype_fixture'"
            )
        return

    if evidence_status == "pre_release_reference":
        if approval_status != "prototype":
            add("pre_release_reference evidence requires approval.status=prototype")
        if reproducibility is not None:
            add("pre-release reference data must not claim the open reproducibility contract")
        if submission_schema_version != "1.0":
            add("pre-release reference data must use the historical submission schema_version=1.0")
        return

    if evidence_status != "submitted_evaluation":
        return

    if approval_status == "prototype":
        add("submitted_evaluation evidence cannot use approval.status=prototype")
    if submission_schema_version not in {"2.0", "3.0"}:
        add("real submitted data requires submission schema_version=2.0 or 3.0")
    if (
        submission_schema_version == "3.0"
        and methodology_kind != "submitter_reported"
    ):
        add(
            "real schema-v3 submitted data must use "
            "methodology.record_kind='submitter_reported'"
        )
    if approval_status not in {None, "approved"}:
        add("a real contributor submission must remain unapproved until maintainer validation")
    if contributor_stage and approval_status == "approved":
        add("contributor-stage submissions cannot set approval.status=approved; maintainers add approval after validation")

    expected_contract = OPEN_REPRODUCIBILITY_CONTRACTS.get(submission_schema_version)
    if not isinstance(reproducibility, dict):
        add(
            f"submitted_evaluation evidence requires reproducibility.contract_version="
            f"{expected_contract!r}"
        )
        return

    if reproducibility.get("contract_version") != expected_contract:
        add(f"reproducibility.contract_version must be {expected_contract!r}")
    code = reproducibility.get("code")
    evaluation = submission.get("evaluation", {})
    if submission_schema_version == "3.0":
        if isinstance(code, dict):
            if submission.get("code_url") != code.get("repository_url"):
                add("code_url must equal reproducibility.code.repository_url")
            if evaluation.get("code_revision") != code.get("commit"):
                add(
                    "evaluation.code_revision must equal the full "
                    "reproducibility.code.commit"
                )
        else:
            if "code_url" in submission:
                add("code_url requires reproducibility.code")
            if "code_revision" in evaluation:
                add("evaluation.code_revision requires reproducibility.code")
    else:
        code_repository_url = (
            code.get("repository_url")
            if isinstance(code, dict)
            else None
        )
        code_commit = code.get("commit") if isinstance(code, dict) else None
        if submission.get("code_url") != code_repository_url:
            add("code_url must equal reproducibility.code.repository_url")
        if evaluation.get("code_revision") != code_commit:
            add(
                "evaluation.code_revision must equal the full "
                "reproducibility.code.commit"
            )
    if dataset_spec.get("status") != "official" and not candidate_dry_run:
        add("submitted_evaluation evidence requires an official dataset specification")

    ground_truth_source = "the leaderboard manifest"
    if candidate_dry_run and submission.get("dataset_id") == "hiliftaeroml":
        profile_definition = dataset_spec.get("profile_definition")
        ground_truth = (
            profile_definition.get("candidate_dry_run_profile_ground_truth", {})
            if isinstance(profile_definition, dict)
            else {}
        )
        ground_truth_source = "the HiLiftAeroML candidate dry-run declaration"
        if not isinstance(ground_truth, dict):
            ground_truth = {}
            add(
                "HiLiftAeroML candidate dry-run profile truth has no benchmark "
                "declaration"
            )
    else:
        ground_truth = manifest.get("data_release", {}).get(
            "profile_ground_truth", {}
        )
    submitted_profile_ground_truth = submission.get("profile_data", {})
    if submitted_profile_ground_truth.get("profile_ground_truth_release_id") != ground_truth.get("release_id"):
        add(
            "profile_data.profile_ground_truth_release_id must match "
            f"{ground_truth_source}"
        )
    if submitted_profile_ground_truth.get("profile_ground_truth_manifest_sha256") != ground_truth.get("manifest_sha256"):
        add(
            "profile_data.profile_ground_truth_manifest_sha256 must match "
            f"{ground_truth_source}"
        )

    split_path = ROOT / "benchmark-specs" / submission["dataset_id"] / split_spec_entry["index_file"]
    if split_path.is_file():
        split_index = load_json(split_path)
        if split_index.get("case_id_status") != "official":
            add("submitted_evaluation evidence requires an official public split index")

    if approval_status != "approved":
        if (directory / "maintainer-validation.json").exists():
            add("maintainer-validation.json is only allowed with maintainer approval")
        return

    validation_metadata = approval.get("validation") if isinstance(approval, dict) else None
    if not isinstance(validation_metadata, dict):
        add("approval.status=approved requires approval.validation")
        return
    validation_filename = validation_metadata.get("evidence_file")
    if validation_filename != "maintainer-validation.json":
        add("approval.validation.evidence_file must be maintainer-validation.json")
        return
    validation_path = directory / validation_filename
    if not validation_path.is_file():
        add("approval.status=approved requires maintainer-validation.json")
        return
    if sha256_file(validation_path) != validation_metadata.get("evidence_sha256"):
        add("maintainer-validation.json does not match approval.validation.evidence_sha256")
    try:
        validation = load_json(validation_path)
    except (OSError, json.JSONDecodeError) as error:
        add(f"cannot read maintainer validation evidence: {error}")
        return
    for error in schema_errors(
        validation,
        "maintainer-validation.schema.json",
        schema_version="v3" if submission_schema_version == "3.0" else "v2",
    ):
        add(f"maintainer-validation.json {error}")

    for key in ("submission_id", "dataset_id", "split_id", "case_set_id"):
        if validation.get(key) != submission.get(key):
            add(f"maintainer-validation.json {key} must equal {submission.get(key)!r}")
    if validation.get("reference_version") != submission.get("evaluation", {}).get("reference_version"):
        add("maintainer-validation.json reference_version must match evaluation.reference_version")
    if validation.get("contract_version") != reproducibility.get("contract_version"):
        add("maintainer-validation.json contract_version must match reproducibility.contract_version")

    validation_ground_truth_fields = {
        "profile_ground_truth_release_id": "release_id",
        "profile_ground_truth_manifest_sha256": "manifest_sha256",
    }
    for validation_key, manifest_key in validation_ground_truth_fields.items():
        submitted_value = submitted_profile_ground_truth.get(validation_key)
        if validation.get(validation_key) != submitted_value:
            add(f"maintainer-validation.json {validation_key} must match submission profile_data")
        if validation.get(validation_key) != ground_truth.get(manifest_key):
            add(f"maintainer-validation.json {validation_key} must match the leaderboard manifest")

    try:
        submitted_date = date.fromisoformat(submission["submitted_at"])
        validation_date = datetime.fromisoformat(str(validation["validated_at"]).replace("Z", "+00:00")).date()
        approved_date = date.fromisoformat(approval["approved_at"])
        if not submitted_date <= validation_date <= approved_date:
            add("submission, maintainer validation, and approval dates must be chronological")
    except (KeyError, TypeError, ValueError):
        pass  # JSON Schema reports malformed or missing date values.

    reviewed_submission = dict(submission)
    reviewed_submission.pop("approval", None)
    if validation.get("reviewed_submission_sha256") != canonical_json_sha256(reviewed_submission):
        add("maintainer-validation.json reviewed_submission_sha256 does not match submission metadata")

    if validation.get("evaluation_evidence_sha256") != submission.get("evaluation", {}).get("evidence_sha256"):
        add("maintainer-validation.json evaluation_evidence_sha256 does not match submission metadata")

    profile_index_path = directory / submission["profile_data"]["index_file"]
    if profile_index_path.is_file():
        if validation.get("profile_index_sha256") != sha256_file(profile_index_path):
            add("maintainer-validation.json profile_index_sha256 does not match the submitted profile index")
    if submission_schema_version == "3.0":
        validation_v3_bindings = {
            "scoring_support_release_id": submission["scoring_support"]["release_id"],
            "scoring_support_manifest_sha256": submission["scoring_support"]["manifest_sha256"],
            "discretization_sha256": submission["spatial_discretization"]["sha256"],
            "case_metrics_sha256": submission["case_metrics"]["sha256"],
        }
        for key, expected in validation_v3_bindings.items():
            if validation.get(key) != expected:
                add(f"maintainer-validation.json {key} must match submission metadata")
        regional = submission.get("regional_diagnostics")
        if isinstance(regional, dict):
            if (
                validation.get("regional_diagnostics_sha256")
                != regional.get("sha256")
            ):
                add(
                    "maintainer-validation.json regional_diagnostics_sha256 "
                    "must match submission metadata"
                )
        elif "regional_diagnostics_sha256" in validation:
            add(
                "maintainer-validation.json regional_diagnostics_sha256 must "
                "be absent when the submission has no regional report"
            )
        if submission.get("dataset_id") == "drivaerml":
            validate_drivaerml_maintainer_receipt_hash(
                add, directory, validation
            )


def _validate_hilift_native_profile_score_bindings(
    add: Any,
    *,
    submission: dict[str, Any],
    case_metrics: dict[str, Any] | None,
    recomputed_scores: dict[str, dict[str, float]],
) -> None:
    """Bind submitted HiLift profile scores to hidden-truth recomputation."""

    metric_ids = ("cp_cut_r2", "velocity_profile_r2")
    if case_metrics is None:
        add(
            "HiLiftAeroML hidden profile-truth recomputation requires valid "
            "case-metrics evidence"
        )
        return
    cases = case_metrics.get("cases")
    if not isinstance(cases, list):
        add("HiLiftAeroML case-metrics evidence has no cases array")
        return
    case_records = {
        case.get("case_id"): case
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }
    aggregate_inputs = {metric_id: [] for metric_id in metric_ids}
    for case_id, scores in recomputed_scores.items():
        record = case_records.get(case_id)
        if not isinstance(record, dict):
            add(f"hidden profile-truth recomputation has no case-metrics record for {case_id}")
            continue
        submitted_scores = record.get("nonspatial_metric_values")
        if not isinstance(submitted_scores, dict):
            add(f"case-metrics {case_id} has no nonspatial_metric_values")
            continue
        for metric_id in metric_ids:
            recomputed = scores.get(metric_id)
            submitted = submitted_scores.get(metric_id)
            if not is_number(recomputed):
                add(
                    f"hidden profile-truth recomputation did not produce a finite "
                    f"{metric_id} for {case_id}"
                )
                continue
            aggregate_inputs[metric_id].append(float(recomputed))
            if not is_number(submitted) or not math.isclose(
                float(submitted),
                float(recomputed),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                add(
                    f"case-metrics {case_id} nonspatial_metric_values.{metric_id} "
                    "must equal the benchmark hidden-truth recomputation"
                )

    submitted_aggregates = submission.get("metric_values")
    if not isinstance(submitted_aggregates, dict):
        add("submission metric_values are absent for hidden profile-truth recomputation")
        return
    for metric_id, values in aggregate_inputs.items():
        if len(values) != len(recomputed_scores) or not values:
            continue
        expected = math.fsum(values) / len(values)
        submitted = submitted_aggregates.get(metric_id)
        if not is_number(submitted) or not math.isclose(
            float(submitted),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            add(
                f"metric_values.{metric_id} must equal the macro-average of the "
                "benchmark hidden-truth per-case recomputation"
            )


def validate_profiles(
    add: Any,
    directory: Path,
    submission: dict[str, Any],
    dataset_spec: dict[str, Any],
    split_spec_entry: dict[str, Any],
    *,
    candidate_dry_run: bool = False,
    registered_preview: bool = False,
    candidate_profile_truth_release: Path | None = None,
    candidate_compact_profile_support_release: Path | None = None,
    opened_candidate_compact_profile_support_release: HiLiftCompactSupportRelease
    | None = None,
    case_metrics: dict[str, Any] | None = None,
) -> dict[str, int]:
    candidate_contract_validation = candidate_dry_run or registered_preview
    profile_format = submission["profile_data"].get(
        "format", "fluidsbench-profile-chunks-v1"
    )
    relative_profile = profile_format == RELATIVE_PROFILE_FORMAT
    # A retained real-inference reference may expose fully validated relative
    # profiles before the closed candidate release is activated.  This is not
    # an intake path: the v1 schema independently restricts it to an explicit
    # pre-release reference with prototype approval, while every series still
    # passes the immutable v3 support/coordinate validator below.
    pre_release_relative_reference = (
        relative_profile
        and submission.get("record_type") == "pre_release_reference"
        and submission.get("approval", {}).get("status") == "prototype"
    )
    hilift_native_profile = profile_format == HILIFT_PROFILE_FORMAT
    hilift_compact_profile = profile_format == HILIFT_COMPACT_PROFILE_FORMAT
    physical_coordinate_profile = (
        profile_format == "fluidsbench-drivaerml-physical-profile-chunks-v1"
    )
    if profile_format not in {
        "fluidsbench-profile-chunks-v1",
        "fluidsbench-drivaerml-physical-profile-chunks-v1",
        RELATIVE_PROFILE_FORMAT,
        HILIFT_PROFILE_FORMAT,
        HILIFT_COMPACT_PROFILE_FORMAT,
    }:
        add(f"unsupported profile_data.format {profile_format!r}")
        return {"cases": 0, "series": 0}
    relative_contract_sha256 = RELATIVE_PROFILE_CONTRACT_SHA256
    declaration = dataset_spec.get("relative_diagnostics")
    release_valid = False
    release_was_checked = False
    if (
        submission.get("dataset_id") == "drivaerml"
        and isinstance(declaration, dict)
        and "activation_release" in declaration
        and not pre_release_relative_reference
    ):
        declaration_claims_activation = (
            declaration.get("status") == "activated"
            or declaration.get("profile_format_enabled") is True
        )
        release_valid = validate_drivaerml_relative_activation_release(
            add,
            dataset_spec,
            declaration,
            require_active=(
                declaration_claims_activation
                or (relative_profile and not candidate_dry_run)
            ),
        )
        release_was_checked = True
    if relative_profile:
        if submission.get("dataset_id") != "drivaerml":
            add("the relative profile format is available only for DrivAerML")
        if not isinstance(declaration, dict):
            add("DrivAerML relative profile format has no benchmark contract declaration")
        else:
            contract = declaration.get("contract")
            if isinstance(contract, dict) and isinstance(contract.get("sha256"), str):
                relative_contract_sha256 = contract["sha256"]
            if not release_was_checked and not pre_release_relative_reference:
                release_valid = validate_drivaerml_relative_activation_release(
                    add,
                    dataset_spec,
                    declaration,
                    require_active=not candidate_dry_run,
                )
            if not candidate_dry_run and not pre_release_relative_reference and (
                declaration.get("status") != "activated"
                or declaration.get("profile_format_enabled") is not True
                or not release_valid
            ):
                add(
                    "DrivAerML relative profile format is closed until all benchmark "
                    "activation gates are complete"
                )
    if (
        (hilift_native_profile or hilift_compact_profile)
        and submission.get("dataset_id") != "hiliftaeroml"
    ):
        add("HiLiftAeroML profile formats are available only for HiLiftAeroML")
    if candidate_profile_truth_release is not None and not candidate_dry_run:
        add(
            "candidate_profile_truth_release is permitted only with "
            "candidate dry-run validation"
        )
    if (
        candidate_compact_profile_support_release is not None
        and not candidate_dry_run
    ):
        add(
            "candidate_compact_profile_support_release is permitted only with "
            "candidate dry-run validation"
        )
    if (
        candidate_contract_validation
        and submission.get("dataset_id") == "hiliftaeroml"
    ):
        if not (hilift_native_profile or hilift_compact_profile):
            add(
                "HiLiftAeroML candidate dry-run requires a retained prediction-only "
                "profile format"
            )
        if (
            hilift_native_profile
            and candidate_profile_truth_release is None
            and not registered_preview
        ):
            add(
                "HiLiftAeroML candidate dry-run native profiles require an "
                "explicit local candidate profile-truth release"
            )
        if hilift_native_profile and candidate_compact_profile_support_release is not None:
            add(
                "HiLiftAeroML native-v1 profiles do not accept the compact "
                "evaluator-support release argument"
            )
        if (
            hilift_compact_profile
            and candidate_compact_profile_support_release is None
            and not registered_preview
        ):
            add(
                "HiLiftAeroML candidate dry-run compact profiles require an "
                "explicit local candidate compact evaluator-support release"
            )
        if hilift_compact_profile and candidate_profile_truth_release is not None:
            add(
                "HiLiftAeroML compact-v2 profiles obtain hidden truth only through "
                "their bound evaluator-support release"
            )
    profile_definition = dataset_spec.get("profile_definition")
    candidate_profile_truth: dict[str, Any] | None = None
    candidate_compact_support: dict[str, Any] | None = None
    if hilift_native_profile and submission.get("dataset_id") == "hiliftaeroml":
        if not isinstance(profile_definition, dict):
            add("HiLiftAeroML native profiles have no benchmark profile definition")
        elif candidate_contract_validation:
            declared_candidate = profile_definition.get(
                "candidate_dry_run_profile_ground_truth"
            )
            if not isinstance(declared_candidate, dict):
                add(
                    "HiLiftAeroML candidate dry-run profile truth has no benchmark "
                    "declaration"
                )
            else:
                candidate_profile_truth = declared_candidate
                submitted_truth = submission.get("profile_data", {})
                if (
                    submitted_truth.get("profile_ground_truth_release_id")
                    != declared_candidate.get("release_id")
                ):
                    add(
                        "profile_data.profile_ground_truth_release_id must match the "
                        "HiLiftAeroML candidate dry-run declaration"
                    )
                if (
                    submitted_truth.get("profile_ground_truth_manifest_sha256")
                    != declared_candidate.get("manifest_sha256")
                ):
                    add(
                        "profile_data.profile_ground_truth_manifest_sha256 must match "
                        "the HiLiftAeroML candidate dry-run declaration"
                    )
        else:
            public_truth = profile_definition.get("profile_ground_truth")
            if (
                not isinstance(public_truth, dict)
                or public_truth.get("status") != "published"
            ):
                add(
                    "HiLiftAeroML native profile intake is closed until profile "
                    "ground truth is published"
                )
    if hilift_compact_profile and submission.get("dataset_id") == "hiliftaeroml":
        compact_definition = dataset_spec.get("compact_profile_definition")
        if not isinstance(compact_definition, dict):
            add("HiLiftAeroML compact profiles have no benchmark contract definition")
        else:
            expected_contract = {
                "contract_id": HILIFT_COMPACT_PROFILE_CONTRACT_ID,
                "format": HILIFT_COMPACT_PROFILE_FORMAT,
                "sha256": HILIFT_COMPACT_PROFILE_CONTRACT_SHA256,
            }
            for key, expected in expected_contract.items():
                if compact_definition.get(key) != expected:
                    add(
                        f"HiLiftAeroML compact profile definition {key} must "
                        f"equal {expected!r}"
                    )
            public_support = compact_definition.get("evaluator_support")
            if public_support != {
                "status": "not_published",
                "release_id": None,
                "manifest_sha256": None,
            }:
                add("HiLiftAeroML public compact evaluator-support boundary differs")
            if candidate_contract_validation:
                declared_support = compact_definition.get(
                    "candidate_dry_run_evaluator_support"
                )
                if not isinstance(declared_support, dict):
                    add(
                        "HiLiftAeroML candidate compact evaluator support has no "
                        "benchmark declaration"
                    )
                else:
                    candidate_compact_support = declared_support
                    submitted_profile = submission.get("profile_data", {})
                    for submitted_key, declaration_key in (
                        ("evaluator_support_release_id", "release_id"),
                        ("evaluator_support_manifest_sha256", "manifest_sha256"),
                    ):
                        if submitted_profile.get(submitted_key) != declared_support.get(
                            declaration_key
                        ):
                            add(
                                f"profile_data.{submitted_key} must match the "
                                "HiLiftAeroML candidate compact evaluator-support "
                                "declaration"
                            )
                    candidate_truth = (
                        profile_definition.get(
                            "candidate_dry_run_profile_ground_truth"
                        )
                        if isinstance(profile_definition, dict)
                        else None
                    )
                    if not isinstance(candidate_truth, dict):
                        add(
                            "HiLiftAeroML compact evaluator support has no bound "
                            "candidate ground-truth declaration"
                        )
                    else:
                        submitted_profile = submission.get("profile_data", {})
                        if submitted_profile.get(
                            "profile_ground_truth_release_id"
                        ) != candidate_truth.get("release_id"):
                            add(
                                "profile_data.profile_ground_truth_release_id must "
                                "match the compact support's source truth declaration"
                            )
                        if submitted_profile.get(
                            "profile_ground_truth_manifest_sha256"
                        ) != candidate_truth.get("manifest_sha256"):
                            add(
                                "profile_data.profile_ground_truth_manifest_sha256 "
                                "must match the compact support's source truth declaration"
                            )
            else:
                add(
                    "HiLiftAeroML compact profile intake is closed until the compact "
                    "contract and evaluator support are explicitly activated"
                )
    index_path = directory / submission["profile_data"]["index_file"]
    if not index_path.is_file():
        add(f"missing profile index: {index_path.relative_to(ROOT)}")
        return {"cases": 0, "series": 0}
    try:
        index = load_json(index_path)
    except (OSError, json.JSONDecodeError) as error:
        add(f"cannot read profile index: {error}")
        return {"cases": 0, "series": 0}
    for error in schema_errors(index, "profile-index.schema.json"):
        add(f"profiles/index.json {error}")

    identities = {
        "submission_id": submission["submission_id"],
        "dataset_id": submission["dataset_id"],
        "split_id": submission["split_id"],
        "case_set_id": submission["case_set_id"],
        "case_count": submission["profile_data"]["case_count"],
    }
    for key, expected in identities.items():
        if index.get(key) != expected:
            add(f"profiles/index.json {key} must equal {expected!r}")
    if relative_profile:
        if index.get("format") != RELATIVE_PROFILE_FORMAT:
            add("profiles/index.json format must match profile_data.format")
        if index.get("contract_id") != RELATIVE_PROFILE_CONTRACT_ID:
            add("profiles/index.json contract_id is not the retained relative-v3 contract")
        if index.get("contract_sha256") != relative_contract_sha256:
            add("profiles/index.json contract_sha256 does not match the benchmark contract")
    elif physical_coordinate_profile:
        if index.get("format") != "fluidsbench-drivaerml-physical-profile-chunks-v1":
            add("profiles/index.json format must match profile_data.format")
    elif hilift_native_profile:
        if index.get("format") != HILIFT_PROFILE_FORMAT:
            add("profiles/index.json format must match profile_data.format")
        if index.get("contract_id") != HILIFT_PROFILE_CONTRACT_ID:
            add("profiles/index.json contract_id is not the retained HiLift native-profile contract")
        if index.get("contract_sha256") != HILIFT_PROFILE_CONTRACT_SHA256:
            add("profiles/index.json contract_sha256 does not match the HiLift native-profile contract")
    elif hilift_compact_profile:
        if index.get("format") != HILIFT_COMPACT_PROFILE_FORMAT:
            add("profiles/index.json format must match profile_data.format")
        if index.get("contract_id") != HILIFT_COMPACT_PROFILE_CONTRACT_ID:
            add(
                "profiles/index.json contract_id is not the retained HiLift "
                "compact-profile contract"
            )
        if index.get("contract_sha256") != HILIFT_COMPACT_PROFILE_CONTRACT_SHA256:
            add(
                "profiles/index.json contract_sha256 does not match the HiLift "
                "compact-profile contract"
            )
        for key in (
            "evaluator_support_release_id",
            "evaluator_support_manifest_sha256",
        ):
            if index.get(key) != submission["profile_data"].get(key):
                add(f"profiles/index.json {key} must match profile_data.{key}")
    elif index.get("format") not in {None, "fluidsbench-profile-chunks-v1"}:
        add("profiles/index.json format must match profile_data.format")

    split_path = ROOT / "benchmark-specs" / submission["dataset_id"] / split_spec_entry["index_file"]
    if not split_path.is_file():
        add(f"missing benchmark split index: {split_path.relative_to(ROOT)}")
        return {"cases": 0, "series": 0}
    split_index = load_json(split_path)
    if sha256_file(split_path) != split_spec_entry["sha256"]:
        add("benchmark split index does not match submission-spec.json sha256")
    if split_index.get("dataset_id") != submission["dataset_id"]:
        add("benchmark split index dataset_id does not match submission.json")
    if split_index.get("split_id") != submission["split_id"]:
        add("benchmark split index split_id does not match submission.json")
    if split_index.get("case_set_id") != split_spec_entry.get("case_set_id"):
        add("benchmark split index case_set_id does not match submission-spec.json")
    if split_index.get("case_id_status") != split_spec_entry.get("case_id_status"):
        add("benchmark split index case_id_status does not match submission-spec.json")
    if split_index.get("case_count") != split_spec_entry.get("case_count"):
        add("benchmark split index case_count does not match submission-spec.json")
    if submission.get("split_sha256") != split_spec_entry["sha256"]:
        add("split_sha256 does not match the benchmark split index")
    expected_case_ids = split_index["case_ids"]
    if submission["profile_data"]["case_count"] != len(expected_case_ids):
        add(f"profile_data.case_count must be {len(expected_case_ids)} for this split")
    if submission["profile_data"].get("case_set_id") != split_index.get("case_set_id"):
        add("profile_data.case_set_id does not match the benchmark split index")
    if submission.get("case_set_id") != split_index.get("case_set_id"):
        add("case_set_id does not match the benchmark split index")
    if submission["profile_data"].get("case_set_id") != submission.get("case_set_id"):
        add("profile_data.case_set_id must equal submission case_set_id")

    panels = {panel["id"]: panel for panel in dataset_spec["profile_panels"]}
    prototype_fixture = submission.get("approval", {}).get("status") == "prototype"
    indexed_case_ids: list[str] = []
    loaded_case_ids: list[str] = []
    series_count = 0
    referenced_files: set[str] = set()
    referenced_profile_artifacts: set[str] = set()
    for chunk_entry in index.get("chunks", []):
        filename = chunk_entry.get("file", "")
        if not filename or Path(filename).name != filename:
            add(f"invalid profile chunk filename: {filename!r}")
            continue
        if filename in referenced_files:
            add(f"profile index references {filename} more than once")
            continue
        referenced_files.add(filename)
        indexed_case_ids.extend(chunk_entry.get("case_ids", []))
        chunk_path = index_path.parent / filename
        if not chunk_path.is_file():
            add(f"missing profile chunk: {chunk_path.relative_to(ROOT)}")
            continue
        if sha256_file(chunk_path) != chunk_entry.get("sha256"):
            add(f"{filename} does not match its index sha256")
        try:
            chunk = load_json(chunk_path)
        except (OSError, json.JSONDecodeError) as error:
            add(f"cannot read {filename}: {error}")
            continue
        chunk_schema = (
            "drivaerml-relative-profile-chunk.schema.json"
            if relative_profile
            else (
                "hiliftaeroml-compact-profile-chunk.schema.json"
                if hilift_compact_profile
                else (
                    "hiliftaeroml-native-profile-chunk.schema.json"
                    if hilift_native_profile
                    else "profile-chunk.schema.json"
                )
            )
        )
        for error in schema_errors(chunk, chunk_schema):
            add(f"profiles/{filename} {error}")
        normalized_relative_chunk: dict[str, Any] | None = None
        if relative_profile:
            try:
                normalized_relative_chunk = (
                    validate_schema_v3_relative_profile_chunk_candidate(
                        chunk,
                        expected_contract_sha256=relative_contract_sha256,
                    )
                )
            except DrivAerDatasetScorerError as error:
                add(f"profiles/{filename} {error}")
        chunk_case_ids = [case.get("case_id") for case in chunk.get("cases", []) if isinstance(case, dict)]
        if chunk_case_ids != chunk_entry.get("case_ids"):
            add(f"{filename} case order does not match profiles/index.json")
        loaded_case_ids.extend(chunk_case_ids)

        if hilift_native_profile:
            chunk_identity = {
                "format": HILIFT_PROFILE_FORMAT,
                "contract_id": HILIFT_PROFILE_CONTRACT_ID,
                "contract_sha256": HILIFT_PROFILE_CONTRACT_SHA256,
                "submission_id": submission["submission_id"],
                "dataset_id": "hiliftaeroml",
                "split_id": submission["split_id"],
                "case_set_id": submission["case_set_id"],
            }
            for key, expected in chunk_identity.items():
                if chunk.get(key) != expected:
                    add(f"profiles/{filename} {key} must equal {expected!r}")
            for case in chunk.get("cases", []):
                if not isinstance(case, dict):
                    continue
                case_id = case.get("case_id")
                for domain_key, expected_name, expected_arrays in (
                    (
                        "surface_cp",
                        "surface-cp-predictions.npz",
                        HILIFT_CP_PROFILE_ARRAYS,
                    ),
                    (
                        "volume_velocity",
                        "volume-velocity-predictions.npz",
                        HILIFT_VELOCITY_PROFILE_ARRAYS,
                    ),
                ):
                    domain = case.get(domain_key)
                    artifact = domain.get("artifact") if isinstance(domain, dict) else None
                    if not isinstance(artifact, dict):
                        add(f"profiles/{filename} {case_id}/{domain_key} artifact is absent")
                        continue
                    relative_artifact = artifact.get("file")
                    expected_relative = f"artifacts/{case_id}/{expected_name}"
                    if relative_artifact != expected_relative:
                        add(
                            f"profiles/{filename} {case_id}/{domain_key} artifact.file "
                            f"must equal {expected_relative!r}"
                        )
                        continue
                    if relative_artifact in referenced_profile_artifacts:
                        add(f"profile artifact {relative_artifact!r} is referenced more than once")
                        continue
                    referenced_profile_artifacts.add(relative_artifact)
                    artifact_path = index_path.parent / relative_artifact
                    try:
                        resolved_artifact = artifact_path.resolve()
                        resolved_artifact.relative_to(index_path.parent.resolve())
                    except (OSError, ValueError):
                        add(f"profile artifact escapes profiles/: {relative_artifact!r}")
                        continue
                    if not artifact_path.is_file() or artifact_path.is_symlink():
                        add(f"missing regular profile artifact: {relative_artifact}")
                        continue
                    if artifact_path.stat().st_size != artifact.get("byte_size"):
                        add(f"profile artifact {relative_artifact} byte_size differs")
                    if artifact.get("source_native_npz_sha256") != domain.get(
                        "source_npz_sha256"
                    ):
                        add(
                            f"profile artifact {relative_artifact} source native SHA-256 differs"
                        )
                    try:
                        validate_hilift_prediction_npz(
                            artifact_path,
                            expected_arrays=expected_arrays,
                            expected_sha256=artifact.get("sha256"),
                            metadata=domain,
                        )
                    except HiLiftNativeProfileError as error:
                        add(f"profile artifact {relative_artifact} is invalid: {error}")
            series_count += 15 * len(chunk_case_ids)
            continue

        if hilift_compact_profile:
            chunk_identity = {
                "format": HILIFT_COMPACT_PROFILE_FORMAT,
                "contract_id": HILIFT_COMPACT_PROFILE_CONTRACT_ID,
                "contract_sha256": HILIFT_COMPACT_PROFILE_CONTRACT_SHA256,
                "submission_id": submission["submission_id"],
                "dataset_id": "hiliftaeroml",
                "split_id": submission["split_id"],
                "case_set_id": submission["case_set_id"],
            }
            for key, expected in chunk_identity.items():
                if chunk.get(key) != expected:
                    add(f"profiles/{filename} {key} must equal {expected!r}")
            for case in chunk.get("cases", []):
                if not isinstance(case, dict):
                    continue
                case_id = case.get("case_id")
                artifact = case.get("artifact")
                if not isinstance(artifact, dict):
                    add(f"profiles/{filename} {case_id} compact artifact is absent")
                    continue
                relative_artifact = artifact.get("file")
                expected_relative = (
                    f"artifacts/{case_id}/compact-profile-predictions.npz"
                )
                if relative_artifact != expected_relative:
                    add(
                        f"profiles/{filename} {case_id} artifact.file must equal "
                        f"{expected_relative!r}"
                    )
                    continue
                if relative_artifact in referenced_profile_artifacts:
                    add(
                        f"profile artifact {relative_artifact!r} is referenced "
                        "more than once"
                    )
                    continue
                referenced_profile_artifacts.add(relative_artifact)
                artifact_path = index_path.parent / relative_artifact
                try:
                    artifact_path.resolve().relative_to(index_path.parent.resolve())
                except (OSError, ValueError):
                    add(f"profile artifact escapes profiles/: {relative_artifact!r}")
                    continue
                if not artifact_path.is_file() or artifact_path.is_symlink():
                    add(f"missing regular profile artifact: {relative_artifact}")
                    continue
                if artifact_path.stat().st_size != artifact.get("byte_size"):
                    add(f"profile artifact {relative_artifact} byte_size differs")
                if sha256_file(artifact_path) != artifact.get("sha256"):
                    add(f"profile artifact {relative_artifact} SHA-256 differs")
            series_count += 15 * len(chunk_case_ids)
            continue

        if relative_profile:
            if normalized_relative_chunk is not None:
                series_count += sum(
                    len(case["series"])
                    for case in normalized_relative_chunk["cases"]
                )
            continue

        for case in chunk.get("cases", []):
            if not isinstance(case, dict):
                continue
            provided: list[tuple[str, str, str]] = []
            for series in case.get("series", []):
                if not isinstance(series, dict):
                    continue
                panel_id = series.get("panel_id")
                panel = panels.get(panel_id)
                station_id = series.get("station_id")
                quantity_id = series.get("quantity_id")
                key = (panel_id, station_id, quantity_id)
                if key in provided:
                    add(f"{case.get('case_id')} contains duplicate series {key}")
                provided.append(key)
                if panel is None:
                    add(f"{case.get('case_id')} has unknown panel_id {panel_id!r}")
                    continue
                if station_id not in panel["station_ids"]:
                    add(f"{case.get('case_id')}/{panel_id} has unknown station_id {station_id!r}")
                if quantity_id not in panel["quantity_ids"]:
                    add(f"{case.get('case_id')}/{panel_id} has unknown quantity_id {quantity_id!r}")
                coordinates = series.get("coordinate", [])
                predictions = series.get("prediction", [])
                if not isinstance(coordinates, list) or not isinstance(predictions, list):
                    add(f"{case.get('case_id')}/{panel_id}/{station_id}/{quantity_id} arrays must be JSON lists")
                    continue
                if len(coordinates) != len(predictions):
                    add(f"{case.get('case_id')}/{panel_id}/{station_id}/{quantity_id} coordinate and prediction lengths differ")
                if len(coordinates) < panel.get("minimum_points", 2):
                    add(f"{case.get('case_id')}/{panel_id}/{station_id}/{quantity_id} has too few points")
                station_sample_counts = panel.get("station_sample_counts", {})
                expected_sample_count = (
                    station_sample_counts.get(station_id)
                    if isinstance(station_sample_counts, dict)
                    else None
                )
                if expected_sample_count is None:
                    expected_sample_count = panel.get("sample_count")
                if (
                    not prototype_fixture
                    and isinstance(expected_sample_count, int)
                    and not isinstance(expected_sample_count, bool)
                    and len(coordinates) != expected_sample_count
                ):
                    add(
                        f"{case.get('case_id')}/{panel_id}/{station_id}/{quantity_id} "
                        f"must contain exactly {expected_sample_count} points"
                    )
                if any(not is_number(value) for value in coordinates):
                    add(f"{case.get('case_id')}/{panel_id}/{station_id}/{quantity_id} coordinates must be finite numbers")
                elif any(right <= left for left, right in zip(coordinates, coordinates[1:])):
                    add(f"{case.get('case_id')}/{panel_id}/{station_id}/{quantity_id} coordinates must be strictly increasing")
                else:
                    station_intervals = panel.get("station_coordinate_intervals", {})
                    interval = (
                        station_intervals.get(station_id)
                        if isinstance(station_intervals, dict)
                        else None
                    )
                    if interval is None:
                        interval = panel.get("coordinate_interval")
                    if (
                        not physical_coordinate_profile
                        and isinstance(interval, list)
                        and len(interval) == 2
                        and all(is_number(value) for value in interval)
                        and len(coordinates) >= 2
                    ):
                        start, end = interval
                        if not math.isclose(coordinates[0], start, rel_tol=0.0, abs_tol=1e-12):
                            add(
                                f"{case.get('case_id')}/{panel_id}/{station_id}/{quantity_id} "
                                f"coordinate must start at {start}"
                            )
                        if not math.isclose(coordinates[-1], end, rel_tol=0.0, abs_tol=1e-12):
                            add(
                                f"{case.get('case_id')}/{panel_id}/{station_id}/{quantity_id} "
                                f"coordinate must end at {end}"
                            )
                        station_spacings = panel.get("station_coordinate_spacings", {})
                        spacing = (
                            station_spacings.get(station_id)
                            if isinstance(station_spacings, dict)
                            else None
                        )
                        if spacing is None:
                            spacing = panel.get("coordinate_spacing")
                        if spacing == "uniform":
                            denominator = len(coordinates) - 1
                            if any(
                                not math.isclose(
                                    value,
                                    start + (end - start) * index / denominator,
                                    rel_tol=0.0,
                                    abs_tol=1e-12,
                                )
                                for index, value in enumerate(coordinates)
                            ):
                                add(
                                    f"{case.get('case_id')}/{panel_id}/{station_id}/{quantity_id} "
                                    "coordinates must be uniformly spaced over the declared interval"
                                )
                if any(not is_number(value) for value in predictions):
                    add(f"{case.get('case_id')}/{panel_id}/{station_id}/{quantity_id} predictions must be finite numbers")
                series_count += 1
            provided_set = set(provided)
            for panel_id, panel in panels.items():
                if (
                    submission.get("dataset_id") == "drivaerml"
                    and submission.get("prediction_scope") == "surface_only"
                    and panel_id == "velocity_profiles"
                ):
                    continue
                if not panel.get("required", True):
                    continue
                expected = {
                    (panel_id, station_id, quantity_id)
                    for station_id in panel["station_ids"]
                    for quantity_id in panel["quantity_ids"]
                }
                missing = sorted(expected - provided_set)
                if missing:
                    add(f"{case.get('case_id')} is missing required profile series: {missing}")

    if indexed_case_ids != expected_case_ids:
        missing = sorted(set(expected_case_ids) - set(indexed_case_ids))
        unexpected = sorted(set(indexed_case_ids) - set(expected_case_ids))
        add(f"profile index case coverage differs from the benchmark split; missing={missing[:5]}, unexpected={unexpected[:5]}")
    if loaded_case_ids != indexed_case_ids:
        add("loaded profile chunk cases do not match the profile index")
    actual_chunk_files = {path.name for path in index_path.parent.glob("chunk-*.json")}
    if actual_chunk_files != referenced_files:
        add(f"profile directory contains unindexed chunks: {sorted(actual_chunk_files - referenced_files)}")
    if hilift_native_profile or hilift_compact_profile:
        actual_profile_artifacts = {
            path.relative_to(index_path.parent).as_posix()
            for path in (index_path.parent / "artifacts").glob("**/*.npz")
            if path.is_file()
        }
        if actual_profile_artifacts != referenced_profile_artifacts:
            add(
                "profile directory artifact inventory differs; "
                f"missing={sorted(referenced_profile_artifacts - actual_profile_artifacts)[:5]}, "
                f"unindexed={sorted(actual_profile_artifacts - referenced_profile_artifacts)[:5]}"
            )
        if (
            hilift_native_profile
            and candidate_dry_run
            and candidate_profile_truth_release is not None
            and candidate_profile_truth is not None
        ):
            try:
                recomputed_scores = score_hilift_native_profile_directory(
                    profiles_root=index_path.parent,
                    release_root=candidate_profile_truth_release,
                    candidate_declaration=candidate_profile_truth,
                    submission_id=submission["submission_id"],
                    split_id=submission["split_id"],
                    case_set_id=submission["case_set_id"],
                    expected_case_ids=expected_case_ids,
                )
            except HiLiftNativeProfileEvaluationError as error:
                add(f"HiLiftAeroML hidden profile-truth scoring failed: {error}")
            else:
                _validate_hilift_native_profile_score_bindings(
                    add,
                    submission=submission,
                    case_metrics=case_metrics,
                    recomputed_scores=recomputed_scores,
                )
        if (
            hilift_compact_profile
            and candidate_dry_run
            and candidate_compact_profile_support_release is not None
            and candidate_compact_support is not None
        ):
            try:
                recomputed_scores = score_hilift_compact_profile_directory(
                    profiles_root=index_path.parent,
                    support_release_root=(
                        candidate_compact_profile_support_release
                    ),
                    support_manifest_sha256=candidate_compact_support[
                        "manifest_sha256"
                    ],
                    submission_id=submission["submission_id"],
                    split_id=submission["split_id"],
                    case_set_id=submission["case_set_id"],
                    expected_case_ids=expected_case_ids,
                    opened_support_release=(
                        opened_candidate_compact_profile_support_release
                    ),
                )
            except HiLiftCompactProfileEvaluationError as error:
                add(f"HiLiftAeroML compact profile scoring failed: {error}")
            else:
                _validate_hilift_native_profile_score_bindings(
                    add,
                    submission=submission,
                    case_metrics=case_metrics,
                    recomputed_scores=recomputed_scores,
                )
    return {"cases": len(loaded_case_ids), "series": series_count}


def validate_submission_file(
    path: Path,
    manifest: dict[str, Any] | None = None,
    *,
    contributor_stage: bool = False,
    candidate_dry_run: bool = False,
    candidate_profile_truth_release: Path | None = None,
    candidate_compact_profile_support_release: Path | None = None,
    opened_candidate_compact_profile_support_release: HiLiftCompactSupportRelease
    | None = None,
) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    stats = {"cases": 0, "series": 0}
    prefix = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    def add(message: str) -> None:
        errors.append(f"{prefix}: {message}")

    if contributor_stage and candidate_dry_run:
        add("--contributor-stage and --candidate-dry-run are mutually exclusive")
        return errors, stats
    if candidate_profile_truth_release is not None and not candidate_dry_run:
        add(
            "--candidate-profile-truth-release requires --candidate-dry-run"
        )
        return errors, stats
    if (
        candidate_compact_profile_support_release is not None
        and not candidate_dry_run
    ):
        add(
            "--candidate-compact-profile-support-release requires "
            "--candidate-dry-run"
        )
        return errors, stats

    try:
        submission = load_submission_json(path)
    except (OSError, json.JSONDecodeError, SubmissionJSONError) as error:
        add(f"cannot read submission JSON: {error}")
        return errors, stats
    if (
        (
            candidate_profile_truth_release is not None
            or candidate_compact_profile_support_release is not None
        )
        and submission.get("dataset_id") != "hiliftaeroml"
    ):
        add(
            "candidate HiLiftAeroML profile release arguments are available "
            "only for HiLiftAeroML candidate validation"
        )
        return errors, stats
    submission_schema_version = submission.get("schema_version")
    if candidate_dry_run and submission_schema_version != "3.0":
        add("candidate dry-run validation accepts only schema_version='3.0'")
        return errors, stats
    schema_directory = {"1.0": "v1", "2.0": "v2", "3.0": "v3"}.get(submission_schema_version)
    if schema_directory is None:
        add("schema_version must be '1.0', '2.0', or '3.0'")
        return errors, stats
    for error in schema_errors(
        submission,
        "submission.schema.json",
        schema_version=schema_directory,
    ):
        add(error)
    if errors:
        return errors, stats

    if candidate_dry_run:
        if "approval" in submission:
            add("candidate dry-run packages must not contain approval metadata")
        for filename in (
            "maintainer-validation.json",
            "prediction-artifact-checks.json",
            "maintainer-replay.json",
        ):
            if (path.parent / filename).exists():
                add(f"candidate dry-run packages must not contain {filename}")

    if manifest is None:
        manifest = manifest_with_benchmark_contract(load_json(MANIFEST_PATH))
    configured_preview_path = is_registered_hiliftaeroml_preview_path(path)
    preview_binding = registered_hiliftaeroml_preview(path, submission, manifest)
    if configured_preview_path and preview_binding is None and not candidate_dry_run:
        add(
            "registered HiLiftAeroML preview does not match its maintainer "
            "lifecycle, submission, evidence, profile-index, or archive binding"
        )
    if contributor_stage and preview_binding is not None:
        add(
            "the maintainer-registered HiLiftAeroML preview is unavailable in "
            "contributor-stage validation"
        )
    registered_preview = (
        preview_binding is not None
        and not contributor_stage
        and not candidate_dry_run
    )
    candidate_contract_validation = candidate_dry_run or registered_preview
    if registered_preview:
        if "approval" in submission:
            add("registered HiLiftAeroML previews must not contain approval metadata")
        for filename in (
            "maintainer-validation.json",
            "prediction-artifact-checks.json",
            "maintainer-replay.json",
        ):
            if (path.parent / filename).exists():
                add(f"registered HiLiftAeroML previews must not contain {filename}")
    dataset = next((item for item in manifest["datasets"] if item["slug"] == submission["dataset_id"]), None)
    if dataset is None:
        add(f"unknown dataset_id {submission['dataset_id']!r}")
        return errors, stats
    split = split_entry(dataset, submission["split_id"])
    if split is None:
        add(f"unknown split_id {submission['split_id']!r} for {dataset['name']}")
        return errors, stats
    spec_path = ROOT / "benchmark-specs" / dataset["slug"] / "submission-spec.json"
    if not spec_path.is_file():
        add(f"missing benchmark specification: {spec_path.relative_to(ROOT)}")
        return errors, stats
    dataset_spec = load_json(spec_path)
    if dataset_spec.get("dataset_id") != submission["dataset_id"]:
        add("benchmark specification dataset_id does not match submission.json")
        return errors, stats
    methodology_contract_path = spec_path.parent / "methodology-contract.json"
    methodology_contract: dict[str, Any] | None = None
    if not methodology_contract_path.is_file():
        add(
            "missing dataset methodology contract: "
            f"{methodology_contract_path.relative_to(ROOT)}"
        )
    else:
        try:
            loaded_methodology_contract = load_json(methodology_contract_path)
        except (OSError, json.JSONDecodeError) as error:
            add(f"cannot read dataset methodology contract: {error}")
        else:
            if isinstance(loaded_methodology_contract, dict):
                methodology_contract = loaded_methodology_contract
            else:
                add("dataset methodology contract must be a JSON object")
    if candidate_contract_validation and dataset_spec.get("status") not in {
        "candidate",
        "candidate_scoring_contract",
        "owner_review_required",
    }:
        add(
            "candidate dry-run requires an explicitly candidate or "
            "owner-review-required dataset specification, never an official one"
        )
    if dataset_spec.get("ranking") != dataset.get("ranking"):
        add("benchmark specification ranking policy does not match the leaderboard manifest")
    spec_split = next((item for item in dataset_spec["splits"] if item["id"] == split["id"]), None)
    if spec_split is None:
        add(f"benchmark specification does not define split {split['id']!r}")
        return errors, stats

    validate_metadata(
        add,
        path,
        submission,
        dataset,
        split,
        require_registered_path=not candidate_dry_run,
    )
    if submission.get("dataset_version") != dataset_spec.get("dataset_version"):
        add(f"dataset_version must equal {dataset_spec.get('dataset_version')!r} from the benchmark specification")
    if submission.get("evaluation", {}).get("reference_version") != dataset_spec.get("evaluation_reference_version"):
        add(
            "evaluation.reference_version must equal the evaluator release declared by the benchmark specification"
        )
    validate_metrics(add, submission, dataset, manifest)
    if submission_schema_version == "3.0" or "methodology" in submission:
        for error in methodology_schema_errors(submission.get("methodology")):
            add(f"methodology schema validation failed: {error}")
        expected_method_case_count = spec_split.get("case_count")
        if not isinstance(expected_method_case_count, int):
            expected_method_case_count = None
        for error in methodology_errors(
            submission,
            expected_case_count=expected_method_case_count,
            contract=methodology_contract,
        ):
            add(f"methodology validation failed: {error}")
    split_case_ids: list[str] = []
    case_metrics: dict[str, Any] | None = None
    has_regional_diagnostics = isinstance(submission.get("regional_diagnostics"), dict)
    if submission_schema_version == "3.0" or has_regional_diagnostics:
        split_path = ROOT / "benchmark-specs" / submission["dataset_id"] / spec_split["index_file"]
        if split_path.is_file():
            try:
                split_index = load_json(split_path)
            except (OSError, json.JSONDecodeError) as error:
                add(f"cannot read benchmark split index for submitted evidence: {error}")
            else:
                split_case_ids = split_index.get("case_ids", [])
                if not isinstance(split_case_ids, list) or any(
                    not isinstance(case_id, str) for case_id in split_case_ids
                ):
                    add("benchmark split index case_ids must be a string array")
                    split_case_ids = []
        if submission_schema_version == "3.0":
            support_manifest, support_case_index = validate_v3_scoring_support(
                add,
                submission,
                dataset_spec,
                spec_split,
                candidate_dry_run=candidate_contract_validation,
            )
            case_metrics = validate_v3_case_metrics(
                add,
                path.parent,
                submission,
                split_case_ids,
                support_manifest,
                support_case_index,
            )
            validate_v3_discretization(
                add,
                path.parent,
                submission,
                split_case_ids,
                support_manifest,
                support_case_index,
            )
            validate_v3_prediction_metadata(
                add,
                path.parent,
                submission,
                split_case_ids,
                contributor_stage=contributor_stage,
                candidate_dry_run=candidate_contract_validation,
                case_metrics=case_metrics,
                dataset_spec=dataset_spec,
            )
    evidence = validate_evaluation_evidence(
        add,
        path.parent,
        submission,
        dataset_spec,
    )
    if submission_schema_version == "3.0" or has_regional_diagnostics:
        validate_regional_diagnostics(
            add,
            path.parent,
            submission,
            dataset_spec,
            split_case_ids,
            evidence,
        )
    validate_open_reproducibility(
        add,
        path.parent,
        submission,
        evidence,
        manifest,
        dataset_spec,
        spec_split,
        contributor_stage=contributor_stage,
        candidate_dry_run=candidate_contract_validation,
    )
    stats = validate_profiles(
        add,
        path.parent,
        submission,
        dataset_spec,
        spec_split,
        candidate_dry_run=candidate_contract_validation,
        registered_preview=registered_preview,
        candidate_profile_truth_release=candidate_profile_truth_release,
        candidate_compact_profile_support_release=(
            candidate_compact_profile_support_release
        ),
        opened_candidate_compact_profile_support_release=(
            opened_candidate_compact_profile_support_release
        ),
        case_metrics=case_metrics,
    )
    if preview_binding is not None and not contributor_stage:
        validate_registered_hiliftaeroml_preview_archive(
            add,
            path.parent,
            preview_binding,
        )
    return errors, stats


def _preopened_hilift_compact_support_for_submission(
    *,
    path: Path,
    manifest: dict[str, Any],
    support_release_root: Path,
    cache: dict[
        tuple[Path, str, str, tuple[str, ...]], HiLiftCompactSupportRelease
    ],
) -> HiLiftCompactSupportRelease | None:
    """Open one exact compact-support case set once for batch candidate checks.

    This is an optimization only.  Any malformed submission or failed pre-open
    falls back to the ordinary per-package validation path, which reports the
    normal validation error.  A cached handle remains keyed by the resolved
    release path, manifest digest, case-set ID, and ordered case IDs.
    """

    try:
        submission = load_submission_json(path)
        if (
            submission.get("dataset_id") != "hiliftaeroml"
            or submission.get("profile_data", {}).get("format")
            != HILIFT_COMPACT_PROFILE_FORMAT
        ):
            return None
        dataset = next(
            (
                item
                for item in manifest["datasets"]
                if item.get("slug") == "hiliftaeroml"
            ),
            None,
        )
        if not isinstance(dataset, dict):
            return None
        split_id = submission.get("split_id")
        if not isinstance(split_id, str):
            return None
        manifest_split = split_entry(dataset, split_id)
        if not isinstance(manifest_split, dict):
            return None
        specification_path = (
            ROOT
            / "benchmark-specs"
            / "hiliftaeroml"
            / "submission-spec.json"
        )
        specification = load_json(specification_path)
        specification_split = next(
            (
                item
                for item in specification.get("splits", [])
                if isinstance(item, dict) and item.get("id") == split_id
            ),
            None,
        )
        if not isinstance(specification_split, dict):
            return None
        index_file = specification_split.get("index_file")
        case_set_id = specification_split.get("case_set_id")
        if not isinstance(index_file, str) or not isinstance(case_set_id, str):
            return None
        split_index = load_json(specification_path.parent / index_file)
        case_ids = split_index.get("case_ids")
        if (
            split_index.get("case_set_id") != case_set_id
            or not isinstance(case_ids, list)
            or not case_ids
            or any(not isinstance(case_id, str) for case_id in case_ids)
        ):
            return None
        compact_definition = specification.get("compact_profile_definition", {})
        declaration = compact_definition.get(
            "candidate_dry_run_evaluator_support", {}
        )
        expected_manifest = declaration.get("manifest_sha256")
        if not isinstance(expected_manifest, str):
            return None
        key = (
            support_release_root.resolve(strict=True),
            expected_manifest,
            case_set_id,
            tuple(case_ids),
        )
        if key not in cache:
            cache[key] = open_hilift_compact_support_release(
                release_root=support_release_root,
                expected_manifest_sha256=expected_manifest,
                expected_case_ids=case_ids,
                case_set_id=case_set_id,
            )
        return cache[key]
    except (
        AttributeError,
        OSError,
        json.JSONDecodeError,
        SubmissionJSONError,
        HiLiftCompactProfileEvaluationError,
        TypeError,
    ):
        return None


def validate_many(
    paths: list[Path] | None = None,
    *,
    contributor_stage: bool = False,
    candidate_dry_run: bool = False,
    candidate_profile_truth_release: Path | None = None,
    candidate_compact_profile_support_release: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, int]]:
    if contributor_stage and candidate_dry_run:
        return [
            "--contributor-stage and --candidate-dry-run are mutually exclusive"
        ], {"submissions": 0, "cases": 0, "series": 0}
    if candidate_profile_truth_release is not None and not candidate_dry_run:
        return [
            "--candidate-profile-truth-release requires --candidate-dry-run"
        ], {"submissions": 0, "cases": 0, "series": 0}
    if (
        candidate_compact_profile_support_release is not None
        and not candidate_dry_run
    ):
        return [
            "--candidate-compact-profile-support-release requires "
            "--candidate-dry-run"
        ], {"submissions": 0, "cases": 0, "series": 0}
    files = submission_files(paths)
    if not files:
        return ["no submission.json files found"], {"submissions": 0, "cases": 0, "series": 0}
    manifest = manifest_with_benchmark_contract(manifest or load_json(MANIFEST_PATH))
    errors: list[str] = []
    totals = {"submissions": len(files), "cases": 0, "series": 0}
    seen_ids: dict[str, Path] = {}
    compact_support_cache: dict[
        tuple[Path, str, str, tuple[str, ...]], HiLiftCompactSupportRelease
    ] = {}
    for path in files:
        opened_compact_support = None
        if (
            candidate_dry_run
            and candidate_compact_profile_support_release is not None
        ):
            opened_compact_support = (
                _preopened_hilift_compact_support_for_submission(
                    path=path,
                    manifest=manifest,
                    support_release_root=(
                        candidate_compact_profile_support_release
                    ),
                    cache=compact_support_cache,
                )
            )
        current_errors, stats = validate_submission_file(
            path,
            manifest,
            contributor_stage=contributor_stage,
            candidate_dry_run=candidate_dry_run,
            candidate_profile_truth_release=candidate_profile_truth_release,
            candidate_compact_profile_support_release=(
                candidate_compact_profile_support_release
            ),
            opened_candidate_compact_profile_support_release=(
                opened_compact_support
            ),
        )
        errors.extend(current_errors)
        totals["cases"] += stats["cases"]
        totals["series"] += stats["series"]
        try:
            submission_id = load_json(path).get("submission_id")
        except (OSError, json.JSONDecodeError):
            continue
        if submission_id in seen_ids:
            errors.append(f"{path.relative_to(ROOT)}: duplicate submission_id also used by {seen_ids[submission_id].relative_to(ROOT)}")
        elif submission_id:
            seen_ids[submission_id] = path
    revision_paths = {path.resolve() for path in files}
    if paths:
        revision_paths.update(path.resolve() for path in submission_files())
    revision_records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(revision_paths):
        try:
            submission = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(submission, dict):
            revision_records.append((path, submission))
    errors.extend(
        validate_result_revisions(
            revision_records,
            focus_paths=set(files) if paths else None,
        )
    )
    return errors, totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="submission directories or submission.json files")
    lifecycle_mode = parser.add_mutually_exclusive_group()
    lifecycle_mode.add_argument(
        "--contributor-stage",
        action="store_true",
        help="reject maintainer approval/validation metadata in contributor-authored packages",
    )
    lifecycle_mode.add_argument(
        "--candidate-dry-run",
        action="store_true",
        help=(
            "validate a closed schema-v3 candidate contract without granting "
            "official acceptance or approval"
        ),
    )
    parser.add_argument(
        "--prediction-check",
        choices=("metadata",),
        help=(
            "validate optional prediction-artifact declarations/check records without downloading "
            "remote artifacts"
        ),
    )
    parser.add_argument(
        "--candidate-profile-truth-release",
        type=Path,
        help=(
            "local inactive HiLiftAeroML profile-truth release; valid only for "
            "--candidate-dry-run and never activates public intake"
        ),
    )
    parser.add_argument(
        "--candidate-compact-profile-support-release",
        type=Path,
        help=(
            "local inactive HiLiftAeroML compact evaluator-support release; "
            "valid only for --candidate-dry-run and never activates public intake"
        ),
    )
    args = parser.parse_args(argv)
    errors, totals = validate_many(
        args.paths or None,
        contributor_stage=args.contributor_stage,
        candidate_dry_run=args.candidate_dry_run,
        candidate_profile_truth_release=args.candidate_profile_truth_release,
        candidate_compact_profile_support_release=(
            args.candidate_compact_profile_support_release
        ),
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    if args.candidate_dry_run:
        message = (
            "CANDIDATE DRY-RUN VALID: "
            f"{totals['submissions']} package(s), {totals['cases']} test cases, "
            f"and {totals['series']} profile series. This is not official "
            "acceptance, approval, or leaderboard eligibility."
        )
    else:
        message = (
            f"PASS {totals['submissions']} submission(s), {totals['cases']} test cases, "
            f"and {totals['series']} profile series."
        )
    if args.prediction_check == "metadata":
        message += " Prediction-artifact metadata checked; no remote artifacts were downloaded."
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
