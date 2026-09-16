"""Reference implementations of benchmark-level score reductions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


LEGACY_ERROR_CAPS = {
    "surface_pressure_rel_l2": 15.0,
    "surface_wall_shear_rel_l2": 20.0,
    "volume_velocity_rel_l2": 12.0,
    "volume_pressure_rel_l2": 15.0,
}
LEGACY_ERROR_WEIGHTS = {
    "surface_pressure_rel_l2": 0.15,
    "surface_wall_shear_rel_l2": 0.10,
    "volume_velocity_rel_l2": 0.15,
    "volume_pressure_rel_l2": 0.10,
}

LEGACY_ERROR_ALIASES = {
    "volume_velocity_rel_l2": ("volume_velocity_rel_l2", "flow_domain_velocity_rel_l2"),
    "volume_pressure_rel_l2": ("volume_pressure_rel_l2", "flow_domain_pressure_rel_l2"),
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def arithmetic_mean(values: Mapping[str, float], metric_ids: Sequence[str]) -> float:
    """Return the unweighted mean of the named component metrics."""

    if not metric_ids:
        raise ValueError("at least one source metric is required")
    return sum(float(values[metric_id]) for metric_id in metric_ids) / len(metric_ids)


def _transformed_components(
    values: Mapping[str, float],
    declaration: Mapping[str, Any],
    *,
    fixed_zero_component_ids: Sequence[str] = (),
) -> dict[str, tuple[float, float]]:
    """Return ``metric_id -> (weight, transformed component score)``."""

    if declaration.get("operation") != "weighted_component_scores":
        raise ValueError("unsupported overall-score composite operation")
    components = declaration.get("components")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)) or not components:
        raise ValueError("overall-score composite requires at least one component")

    result: dict[str, tuple[float, float]] = {}
    fixed_zero = set(fixed_zero_component_ids)
    for component in components:
        if not isinstance(component, Mapping):
            raise ValueError("overall-score components must be objects")
        metric_id = component.get("metric_id")
        if not isinstance(metric_id, str) or not metric_id or metric_id in result:
            raise ValueError("overall-score component metric IDs must be non-empty and unique")
        weight = float(component.get("weight"))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("overall-score component weights must be finite and non-negative")
        if metric_id in fixed_zero:
            # Scope-unavailable components retain their published weight and
            # contribute a literal zero score.  Their raw field/profile metric
            # is intentionally absent rather than represented by fake data.
            result[metric_id] = (weight, 0.0)
            continue
        source_value = float(values[metric_id])
        if not math.isfinite(source_value):
            raise ValueError("overall-score component values must be finite")

        transform = component.get("transform")
        if transform == "bounded_error":
            cap = float(component.get("cap"))
            if not math.isfinite(cap) or cap <= 0.0:
                raise ValueError("bounded-error components require a finite positive cap")
            component_score = _clamp(100.0 * (1.0 - source_value / cap), 0.0, 100.0)
        elif transform == "bounded_quality":
            if "cap" in component:
                raise ValueError("bounded-quality components must not declare an error cap")
            component_score = 100.0 * _clamp(source_value, 0.0, 1.0)
        elif transform == "physics_null_skill":
            if source_value < 0.0:
                raise ValueError("physics-null skill requires a non-negative error metric")
            baseline_error = float(component.get("baseline_error"))
            if not math.isfinite(baseline_error) or baseline_error <= 0.0:
                raise ValueError(
                    "physics-null skill requires a finite positive baseline_error"
                )
            component_score = 100.0 * (1.0 - source_value / baseline_error)
        else:
            raise ValueError("unsupported overall-score component transform")
        result[metric_id] = (weight, component_score)
    return result


def _require_normalized_weights(components: Mapping[str, tuple[float, float]]) -> None:
    if not math.isclose(
        sum(weight for weight, _ in components.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("overall-score component weights must sum to one")


def composite_overall_score(
    values: Mapping[str, float],
    declaration: Mapping[str, Any],
    *,
    fixed_zero_component_ids: Sequence[str] = (),
) -> float:
    """Evaluate a dataset-declared weighted composite score.

    ``bounded_error`` components convert an error ``e`` with published cap
    ``c`` to ``clip(100 * (1 - e / c), 0, 100)``. ``bounded_quality``
    components convert a quality value such as R2 to
    ``100 * clip(q, 0, 1)``. ``physics_null_skill`` converts an error ``e``
    using a frozen physics-null error ``b`` to ``100 * (1 - e / b)``. The
    latter is deliberately not clipped: methods worse than the null baseline
    retain negative skill. Component weights must be non-negative and sum to
    one.
    """

    components = _transformed_components(
        values, declaration, fixed_zero_component_ids=fixed_zero_component_ids
    )
    _require_normalized_weights(components)
    return sum(weight * score for weight, score in components.values())


def composite_component_group_scores(
    values: Mapping[str, float],
    overall_declaration: Mapping[str, Any],
    group_declaration: Mapping[str, Any],
    *,
    fixed_zero_component_ids: Sequence[str] = (),
) -> dict[str, float]:
    """Evaluate normalized intermediate scores from overall-score components.

    A group names a result metric (for example ``field_score``) and the
    component metric IDs that contribute to it. Component transformations and
    weights are inherited from ``overall_score_composite`` so there is only one
    numerical source of truth. Groups must partition the overall components.
    """

    if group_declaration.get("operation") != "normalized_weighted_component_scores":
        raise ValueError("unsupported component-score group operation")
    groups = group_declaration.get("groups")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)) or not groups:
        raise ValueError("component-score declaration requires at least one group")

    components = _transformed_components(
        values,
        overall_declaration,
        fixed_zero_component_ids=fixed_zero_component_ids,
    )
    _require_normalized_weights(components)
    results: dict[str, float] = {}
    grouped_component_ids: list[str] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("component-score groups must be objects")
        target_metric_id = group.get("metric_id")
        if (
            not isinstance(target_metric_id, str)
            or not target_metric_id
            or target_metric_id in results
        ):
            raise ValueError("component-score target metric IDs must be non-empty and unique")
        component_metric_ids = group.get("component_metric_ids")
        if (
            not isinstance(component_metric_ids, Sequence)
            or isinstance(component_metric_ids, (str, bytes))
            or not component_metric_ids
            or any(not isinstance(metric_id, str) or not metric_id for metric_id in component_metric_ids)
            or len(component_metric_ids) != len(set(component_metric_ids))
        ):
            raise ValueError("each component-score group requires unique component metric IDs")
        unknown = set(component_metric_ids) - set(components)
        if unknown:
            raise ValueError(f"component-score group references unknown components: {sorted(unknown)}")

        grouped_component_ids.extend(component_metric_ids)
        group_weight = sum(components[metric_id][0] for metric_id in component_metric_ids)
        if group_weight <= 0.0:
            raise ValueError("component-score group weights must sum to a positive value")
        results[target_metric_id] = (
            sum(
                components[metric_id][0] * components[metric_id][1]
                for metric_id in component_metric_ids
            )
            / group_weight
        )

    if len(grouped_component_ids) != len(set(grouped_component_ids)):
        raise ValueError("overall-score components may belong to only one component-score group")
    if set(grouped_component_ids) != set(components):
        missing = sorted(set(components) - set(grouped_component_ids))
        raise ValueError(f"component-score groups do not cover overall components: {missing}")
    return results


def _legacy_error_value(values: Mapping[str, float], metric_id: str) -> float:
    aliases = LEGACY_ERROR_ALIASES.get(metric_id, (metric_id,))
    matches = [alias for alias in aliases if alias in values]
    if len(matches) != 1:
        raise KeyError(f"expected exactly one of: {', '.join(aliases)}")
    return float(values[matches[0]])


def legacy_aero_scores(values: Mapping[str, float]) -> dict[str, float]:
    """Return the four scores used by the prototype external-aero datasets."""

    field_score = sum(
        LEGACY_ERROR_WEIGHTS[metric_id]
        * _clamp(100.0 * (1.0 - _legacy_error_value(values, metric_id) / LEGACY_ERROR_CAPS[metric_id]), 0.0, 100.0)
        for metric_id in LEGACY_ERROR_WEIGHTS
    ) / 0.5
    force_score = (
        0.15 * _clamp(float(values["cd_r2"]), 0.0, 1.0) * 100.0
        + 0.10 * _clamp(float(values["cl_r2"]), 0.0, 1.0) * 100.0
    ) / 0.25
    profile_score = (
        0.15 * _clamp(float(values["velocity_profile_r2"]), 0.0, 1.0) * 100.0
        + 0.10 * _clamp(float(values["cp_cut_r2"]), 0.0, 1.0) * 100.0
    ) / 0.25
    return {
        "field_score": field_score,
        "force_score": force_score,
        "diagnostic_score": profile_score,
        "overall_score": 0.5 * field_score + 0.25 * force_score + 0.25 * profile_score,
    }
