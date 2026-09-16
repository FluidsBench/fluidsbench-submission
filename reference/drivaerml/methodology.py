"""Semantic checks for the dataset-neutral FluidsBench method record."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
# Generous for model reporting while remaining below the exact-integer range of
# IEEE-754 binary64, which backs the legacy millions-valued display field.
MAX_PARAMETER_COUNT = 1_000_000_000_000_000


class MethodologyError(ValueError):
    """Raised when method metadata is inconsistent with its result package."""


# Retained for callers of the first, DrivAerML-specific implementation.
DrivAerMethodologyError = MethodologyError


def _named_values(value: Any, key: str) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [item.get(key) for item in value if isinstance(item, dict)]


def _duplicate_values(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    duplicates: set[Any] = set()
    for value in values:
        try:
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        except TypeError:
            continue
    return sorted(duplicates, key=str)


def derived_parameter_count_millions(methodology: Any) -> float:
    """Return the exact total learned parameter count in millions."""

    if not isinstance(methodology, dict):
        raise MethodologyError("methodology must be an object")
    architecture = methodology.get("architecture")
    if not isinstance(architecture, dict):
        raise MethodologyError("methodology.architecture must be an object")
    parameter_count = architecture.get("total_parameter_count")
    if (
        not isinstance(parameter_count, int)
        or isinstance(parameter_count, bool)
        or parameter_count < 0
        or parameter_count > MAX_PARAMETER_COUNT
    ):
        raise MethodologyError(
            "methodology.architecture.total_parameter_count must be a nonnegative "
            f"integer no greater than {MAX_PARAMETER_COUNT}"
        )
    try:
        derived = parameter_count / 1_000_000.0
    except OverflowError as error:  # Defensive if the bound changes later.
        raise MethodologyError(
            "methodology.architecture.total_parameter_count cannot be represented "
            "in millions"
        ) from error
    if not math.isfinite(derived):
        raise MethodologyError(
            "methodology.architecture.total_parameter_count cannot be represented "
            "in millions"
        )
    return derived


def _nonfinite_number_paths(value: Any, path: str = "methodology") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(_nonfinite_number_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_nonfinite_number_paths(item, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        paths.append(path)
    return paths


def _known_reference_errors(
    entries: Any, *, label: str, known_component_ids: set[str]
) -> list[str]:
    errors: list[str] = []
    if not isinstance(entries, list):
        return errors
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        component_ids = entry.get("component_ids")
        if not isinstance(component_ids, list) or not all(
            isinstance(component_id, str) for component_id in component_ids
        ):
            continue
        unknown = sorted(set(component_ids) - known_component_ids)
        if unknown:
            errors.append(
                f"methodology.{label}[{index}].component_ids references unknown "
                f"architecture components: {unknown}"
            )
    return errors


def _compute_capacity_error(
    compute: Any,
    *,
    label: str,
    campaign_key: str,
    aggregate_key: str,
) -> str | None:
    """Reject aggregate device time that cannot fit inside the campaign span."""

    if not isinstance(compute, dict):
        return None
    device_count = compute.get("max_concurrent_device_count")
    campaign = compute.get(campaign_key)
    aggregate = compute.get(aggregate_key)
    values = (campaign, aggregate)
    if (
        not isinstance(device_count, int)
        or isinstance(device_count, bool)
        or device_count < 1
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            for value in values
        )
    ):
        return None
    try:
        campaign_value = float(campaign)
        aggregate_value = float(aggregate)
        capacity = device_count * campaign_value
    except OverflowError:
        return f"{label} timing values must have a finite numeric representation"
    if not all(
        math.isfinite(value)
        for value in (campaign_value, aggregate_value, capacity)
    ):
        return None  # The general non-finite-value walk reports these paths.
    if aggregate_value > capacity and not math.isclose(
        aggregate_value,
        capacity,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return (
            f"{label}.{aggregate_key} cannot exceed "
            f"max_concurrent_device_count * {campaign_key}"
        )
    return None


def methodology_errors(
    submission: Mapping[str, Any],
    *,
    expected_case_count: int | None = None,
    contract: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return cross-field errors not expressible cleanly in JSON Schema."""

    errors: list[str] = []
    methodology = submission.get("methodology")
    if not isinstance(methodology, dict):
        return ["methodology must be an object"]

    record_kind = methodology.get("record_kind")
    if record_kind not in {
        "submitter_reported",
        "prototype_fixture",
        "format_example",
    }:
        errors.append("methodology.record_kind is not recognized")

    for path in _nonfinite_number_paths(methodology):
        errors.append(f"{path} must be a finite number")

    architecture = methodology.get("architecture")
    if not isinstance(architecture, dict):
        return [*errors, "methodology.architecture must be an object"]

    try:
        expected_millions = derived_parameter_count_millions(methodology)
    except MethodologyError as error:
        errors.append(str(error))
    else:
        declared_millions = submission.get("parameter_count_millions")
        try:
            finite_declared_millions = (
                isinstance(declared_millions, (int, float))
                and not isinstance(declared_millions, bool)
                and math.isfinite(float(declared_millions))
            )
        except OverflowError:
            finite_declared_millions = False
        if not finite_declared_millions or float(declared_millions) != expected_millions:
            errors.append(
                "parameter_count_millions must equal "
                "methodology.architecture.total_parameter_count / 1,000,000"
            )

    total_parameter_count = architecture.get("total_parameter_count")
    parameter_count_basis = architecture.get("parameter_count_basis")
    if record_kind in {"submitter_reported", "format_example"}:
        if parameter_count_basis != "exact":
            errors.append(
                "methodology.architecture.parameter_count_basis must be exact for "
                f"record_kind={record_kind!r}"
            )
    elif record_kind == "prototype_fixture" and parameter_count_basis not in {
        "exact",
        "rounded_from_reported_millions",
    }:
        errors.append(
            "prototype methodology parameter_count_basis must be exact or "
            "rounded_from_reported_millions"
        )
    submitter_trainable_count = architecture.get(
        "submitter_trainable_parameter_count"
    )
    if (
        isinstance(total_parameter_count, int)
        and not isinstance(total_parameter_count, bool)
        and isinstance(submitter_trainable_count, int)
        and not isinstance(submitter_trainable_count, bool)
        and submitter_trainable_count > total_parameter_count
    ):
        errors.append(
            "methodology.architecture.submitter_trainable_parameter_count cannot "
            "exceed total_parameter_count"
        )

    components = architecture.get("components")
    component_ids = _named_values(components, "id")
    duplicate_component_ids = _duplicate_values(component_ids)
    if duplicate_component_ids:
        errors.append(
            "methodology.architecture.components id values must be unique; "
            f"duplicates: {duplicate_component_ids}"
        )
    known_component_ids = {
        component_id for component_id in component_ids if isinstance(component_id, str)
    }
    if isinstance(components, list) and isinstance(total_parameter_count, int):
        component_counts = [
            component.get("parameter_count")
            for component in components
            if isinstance(component, dict)
        ]
        if component_counts and all(
            isinstance(count, int) and not isinstance(count, bool)
            for count in component_counts
        ):
            if sum(component_counts) != total_parameter_count:
                errors.append(
                    "methodology.architecture.total_parameter_count must equal the "
                    "sum of architecture.components parameter_count values"
                )

    predicted_field_entries = architecture.get("predicted_fields")
    predicted_fields = _named_values(predicted_field_entries, "field_id")
    if not all(isinstance(value, str) for value in predicted_fields):
        errors.append(
            "methodology.architecture.predicted_fields field_id values must be strings"
        )
        observed_fields: set[str] = set()
    else:
        observed_fields = set(predicted_fields)
        if len(predicted_fields) != len(observed_fields):
            errors.append(
                "methodology.architecture.predicted_fields field_id values must be unique"
            )
    if contract is not None:
        contract_dataset_id = contract.get("dataset_id")
        if contract_dataset_id != submission.get("dataset_id"):
            errors.append(
                "methodology contract dataset_id must match submission.dataset_id"
            )
        prediction_scope = submission.get("prediction_scope")
        scoped_entries = contract.get("required_predicted_fields_by_prediction_scope")
        if (
            isinstance(prediction_scope, str)
            and isinstance(scoped_entries, Mapping)
            and isinstance(scoped_entries.get(prediction_scope), list)
        ):
            expected_entries = scoped_entries[prediction_scope]
        else:
            expected_entries = contract.get("required_predicted_fields")
        expected_by_id = {
            entry.get("field_id"): entry
            for entry in expected_entries
            if isinstance(entry, dict) and isinstance(entry.get("field_id"), str)
        } if isinstance(expected_entries, list) else {}
        expected_fields = set(expected_by_id)
        missing_fields = sorted(expected_fields - observed_fields)
        if missing_fields:
            errors.append(
                "methodology.architecture.predicted_fields is missing required "
                f"{contract_dataset_id} fields: {missing_fields}"
            )
        if contract.get("allow_additional_predicted_fields") is False:
            additional_fields = sorted(observed_fields - expected_fields)
            if additional_fields:
                errors.append(
                    "methodology.architecture.predicted_fields contains fields not "
                    f"allowed by the {contract_dataset_id} contract: {additional_fields}"
                )
        observed_by_id = {
            entry.get("field_id"): entry
            for entry in predicted_field_entries
            if isinstance(entry, dict) and isinstance(entry.get("field_id"), str)
        } if isinstance(predicted_field_entries, list) else {}
        for field_id, expected_entry in expected_by_id.items():
            observed_entry = observed_by_id.get(field_id)
            if observed_entry is None:
                continue
            for key in ("domain", "component_count"):
                if observed_entry.get(key) != expected_entry.get(key):
                    errors.append(
                        "methodology.architecture.predicted_fields entry "
                        f"{field_id!r} {key} must equal "
                        f"{expected_entry.get(key)!r} from the dataset contract"
                    )

    training = methodology.get("training")
    training_stages = training.get("stages") if isinstance(training, dict) else None
    for collection, label, prefix in (
        (
            architecture.get("key_hyperparameters"),
            "id",
            "architecture.key_hyperparameters",
        ),
        (architecture.get("input_features"), "id", "architecture.input_features"),
        (training_stages, "id", "training.stages"),
        (methodology.get("checkpoints"), "id", "checkpoints"),
    ):
        duplicates = _duplicate_values(_named_values(collection, label))
        if duplicates:
            errors.append(
                f"methodology.{prefix} {label} values must be unique; "
                f"duplicates: {duplicates}"
            )

    for entries, label in (
        (architecture.get("key_hyperparameters"), "architecture.key_hyperparameters"),
        (architecture.get("input_features"), "architecture.input_features"),
        (architecture.get("predicted_fields"), "architecture.predicted_fields"),
        (training_stages, "training.stages"),
        (methodology.get("checkpoints"), "checkpoints"),
    ):
        errors.extend(
            _known_reference_errors(
                entries,
                label=label,
                known_component_ids=known_component_ids,
            )
        )

    checkpoints = methodology.get("checkpoints")
    if isinstance(checkpoints, list):
        for index, checkpoint in enumerate(checkpoints):
            if not isinstance(checkpoint, dict):
                continue
            digest = checkpoint.get("sha256")
            if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
                errors.append(
                    f"methodology.checkpoints[{index}].sha256 must be a lowercase SHA-256"
                )

    submitter_stage_count = 0
    trained_component_ids: set[str] = set()
    if isinstance(training_stages, list):
        for index, stage in enumerate(training_stages):
            if not isinstance(stage, dict):
                continue
            stage_component_ids = stage.get("component_ids")
            if isinstance(stage_component_ids, list):
                trained_component_ids.update(
                    component_id
                    for component_id in stage_component_ids
                    if isinstance(component_id, str)
                )
            if stage.get("status") != "performed_by_submitter":
                if (
                    stage.get("status") == "prototype_not_recorded"
                    and record_kind != "prototype_fixture"
                ):
                    errors.append(
                        f"methodology.training.stages[{index}] may use "
                        "prototype_not_recorded only for prototype_fixture records"
                    )
                continue
            submitter_stage_count += 1
            run_count = stage.get("run_count")
            random_seeds = stage.get("random_seeds")
            stochastic = stage.get("stochastic")
            if (
                isinstance(run_count, int)
                and not isinstance(run_count, bool)
                and isinstance(random_seeds, list)
            ):
                if stochastic is True and len(random_seeds) != run_count:
                    errors.append(
                        f"methodology.training.stages[{index}].random_seeds must "
                        "contain exactly one seed for each stochastic run"
                    )
                elif stochastic is False and random_seeds:
                    errors.append(
                        f"methodology.training.stages[{index}].random_seeds must be "
                        "empty when stochastic is false"
                    )
            compute_error = _compute_capacity_error(
                stage.get("compute"),
                label=f"methodology.training.stages[{index}].compute",
                campaign_key="campaign_wall_time_hours",
                aggregate_key="aggregate_device_hours",
            )
            if compute_error is not None:
                errors.append(compute_error)

    missing_training_provenance = sorted(
        known_component_ids - trained_component_ids
    )
    if missing_training_provenance:
        errors.append(
            "methodology.training.stages must disclose training provenance for every "
            f"architecture component; missing: {missing_training_provenance}"
        )
    if (
        submitter_stage_count == 0
        and isinstance(submitter_trainable_count, int)
        and submitter_trainable_count != 0
    ):
        errors.append(
            "methodology.architecture.submitter_trainable_parameter_count must be "
            "zero when no training stage was performed by the submitter"
        )

    checkpoint_component_ids: set[str] = set()
    if isinstance(checkpoints, list):
        for checkpoint in checkpoints:
            if isinstance(checkpoint, dict) and isinstance(
                checkpoint.get("component_ids"), list
            ):
                checkpoint_component_ids.update(
                    component_id
                    for component_id in checkpoint["component_ids"]
                    if isinstance(component_id, str)
                )
    component_entries = components if isinstance(components, list) else []
    parameterized_component_ids = {
        component.get("id")
        for component in component_entries
        if isinstance(component, dict)
        and isinstance(component.get("id"), str)
        and isinstance(component.get("parameter_count"), int)
        and not isinstance(component.get("parameter_count"), bool)
        and component.get("parameter_count") > 0
    }
    missing_checkpoint_components = sorted(
        parameterized_component_ids - checkpoint_component_ids
    )
    if missing_checkpoint_components and record_kind != "prototype_fixture":
        errors.append(
            "methodology.checkpoints must bind every parameterized architecture "
            f"component; missing: {missing_checkpoint_components}"
        )

    inference = methodology.get("inference_compute")
    inference_status = inference.get("status") if isinstance(inference, dict) else None
    if (
        record_kind in {"submitter_reported", "format_example"}
        and inference_status != "measured"
    ):
        errors.append(
            "methodology.inference_compute.status must be measured for "
            f"record_kind={record_kind!r}"
        )
    if (
        expected_case_count is not None
        and isinstance(inference, dict)
        and inference_status == "measured"
    ):
        if inference.get("case_count") != expected_case_count:
            errors.append(
                "methodology.inference_compute.case_count must equal the official "
                f"evaluation case count {expected_case_count}"
            )
    inference_compute_error = _compute_capacity_error(
        inference,
        label="methodology.inference_compute",
        campaign_key="campaign_wall_time_seconds",
        aggregate_key="aggregate_device_time_seconds",
    )
    if inference_compute_error is not None:
        errors.append(inference_compute_error)

    return errors


def require_methodology(
    submission: Mapping[str, Any],
    *,
    expected_case_count: int | None = None,
    contract: Mapping[str, Any] | None = None,
) -> None:
    """Raise one stable exception containing every semantic inconsistency."""

    errors = methodology_errors(
        submission,
        expected_case_count=expected_case_count,
        contract=contract,
    )
    if errors:
        raise MethodologyError("; ".join(errors))
