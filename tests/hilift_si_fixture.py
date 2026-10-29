"""Synthetic SI conversion regression helper; not a submission migration tool."""
from typing import Any
import copy
import math
from reference.hiliftaeroml.dimensional_units import METRIC_FACTORS, CONTRACT_ID, native_error_to_si, export_binding

def corrected_documents(
    submission: dict[str, Any], evidence: dict[str, Any], cases: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive a correction without mutating inputs; refuse a second conversion."""
    if submission.get("dataset_id") != "hiliftaeroml" or evidence.get("dataset_id") != "hiliftaeroml":
        raise ValueError("SI correction is only defined for HiLiftAeroML")
    if "dimensional_unit_conversion" in evidence:
        raise ValueError("dimensional unit conversion is already recorded")
    if submission["metric_values"] != evidence["metric_values"]:
        raise ValueError("source submission/evidence metrics differ")
    if cases.get("metric_values") != submission["metric_values"]:
        raise ValueError("source case aggregate/submission metrics differ")
    result, proof, case_document = copy.deepcopy((submission, evidence, cases))
    reductions = {metric: [] for metric in METRIC_FACTORS}
    for case in case_document["cases"]:
        seen = set()
        for support in case["supports"]:
            values = support["metric_values"]
            for metric in METRIC_FACTORS.keys() & values.keys():
                if metric in seen:
                    raise ValueError(f"{case['case_id']}: duplicate {metric}")
                seen.add(metric)
                values[metric] = native_error_to_si(metric, values[metric])
                reductions[metric].append(values[metric])
        if seen != METRIC_FACTORS.keys():
            raise ValueError(f"{case['case_id']}: incomplete dimensional metric inventory")
    if not case_document["cases"]:
        raise ValueError("cannot correct an empty case set")
    for metric, values in reductions.items():
        corrected = math.fsum(values) / len(values)
        if not math.isclose(corrected, native_error_to_si(metric, result["metric_values"][metric]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{metric}: source aggregate is not the equal-case mean")
        result["metric_values"][metric] = corrected
        proof["metric_values"][metric] = corrected
        case_document["metric_values"][metric] = corrected
    proof["dimensional_unit_conversion"] = export_binding()
    proof["notes"] = (proof.get("notes", "") + " Dimensional field MAE/RMSE is converted from solver-native units to SI by "
                      + CONTRACT_ID + "; this export stage is separate from the earlier frozen native-evaluator binding. "
                      "Predictions, relative statistics, profiles, coefficients and composite scores are unchanged.")
    return result, proof, case_document

