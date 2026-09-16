#!/usr/bin/env python3
"""Build truthful structured method records for prototype leaderboard fixtures."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


SPATIAL_DIMENSIONS = {
    "airfrans": 2,
    "vki-ls59": 2,
}

MODEL_FAMILY_NOTES = {
    "AB-UPT": "Attention-based Universal Physics Transformer family.",
    "Fourier Neural Operator": "Fourier-domain neural-operator family.",
    "GINO": "Geometry-informed neural-operator family coupling geometric and operator representations.",
    "GeoTransformer": "Geometry-aware transformer family for irregular spatial supports.",
    "Graph U-Net": "Multiscale graph encoder-decoder family with graph pooling and unpooling.",
    "GraphSAGE": "Graph neighborhood-aggregation family.",
    "LNO": "Latent neural-operator family.",
    "MeshGraphNets": "Encode-process-decode message-passing graph-network family.",
    "MLP": "Pointwise multilayer-perceptron family.",
    "OFormer": "Operator-transformer family.",
    "Point Transformer": "Point-cloud transformer family.",
    "Point-MAE": "Masked point-cloud autoencoder family adapted to field prediction.",
    "PointNet": "PointNet family using shared pointwise features and permutation-invariant aggregation.",
    "PointNet-Large": "Larger PointNet-family variant.",
    "RegDGCNN": "Dynamic graph-convolution family adapted to regression.",
    "Transolver": "Physics-attention transformer family.",
    "Transolver++": "Enhanced Transolver physics-attention family.",
    "Transolver-Large": "Larger Transolver physics-attention variant.",
    "TriPNet": "Implicit-field prediction family using tri-plane representations.",
    "UPT": "Universal Physics Transformer family.",
}


class PrototypeMethodologyError(ValueError):
    """Raised when a prototype submission cannot be represented truthfully."""


def nominal_parameter_count(parameter_count_millions: Any) -> int:
    """Convert the displayed millions value to a deterministic nominal integer."""

    try:
        value = Decimal(str(parameter_count_millions)) * Decimal(1_000_000)
    except (InvalidOperation, ValueError) as error:
        raise PrototypeMethodologyError(
            "parameter_count_millions must be a finite decimal value"
        ) from error
    integral = value.to_integral_value()
    if value != integral or integral < 0:
        raise PrototypeMethodologyError(
            "parameter_count_millions must map to a nonnegative whole parameter count"
        )
    return int(integral)


def build_prototype_methodology(
    submission: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a dataset-aware record without inventing unpublished run details."""

    dataset_id = submission.get("dataset_id")
    if contract.get("dataset_id") != dataset_id:
        raise PrototypeMethodologyError(
            "methodology contract dataset_id does not match the submission"
        )
    model = submission.get("model")
    model_type = submission.get("model_type")
    if not isinstance(model, str) or not model or not isinstance(model_type, str) or not model_type:
        raise PrototypeMethodologyError("submission model and model_type must be non-empty strings")
    total_parameters = nominal_parameter_count(submission.get("parameter_count_millions"))
    required_fields = contract.get("required_predicted_fields")
    if not isinstance(required_fields, list) or not required_fields:
        raise PrototypeMethodologyError("methodology contract has no required predicted fields")

    component_id = "prototype-surrogate"
    family_note = MODEL_FAMILY_NOTES.get(
        model,
        f"{model_type} surrogate-model family.",
    )
    spatial_dimension = SPATIAL_DIMENSIONS.get(str(dataset_id), 3)
    predicted_fields = []
    for field in required_fields:
        if not isinstance(field, dict):
            raise PrototypeMethodologyError("methodology contract fields must be objects")
        predicted_fields.append(
            {
                "field_id": field["field_id"],
                "domain": field["domain"],
                "component_count": field["component_count"],
                "component_ids": [component_id],
                "production": (
                    "derived_from_model_output"
                    if field["domain"] == "case"
                    else "direct_model_output"
                ),
                "description": (
                    f"Prototype declaration for {field['description']} The fixture does "
                    "not retain executable predictions proving whether the named "
                    "architecture produced this quantity directly."
                ),
            }
        )

    return {
        "format": "fluidsbench-method-v1",
        "record_kind": "prototype_fixture",
        "record_note": (
            "Leaderboard interface fixture only. The model name, broad family, and "
            "nominal parameter count mirror the prototype row; component breakdown, "
            "hyperparameters, training run, checkpoint, and timing were not submitted "
            "or independently verified. This record must not be cited as the authors' "
            "methodology."
        ),
        "architecture": {
            "description": (
                f"{model} prototype row ({model_type}). {family_note} The repository "
                "does not contain a verified layer-by-layer configuration, so the "
                "nominal parameter count is represented as one aggregate component."
            ),
            "total_parameter_count": total_parameters,
            "parameter_count_basis": "rounded_from_reported_millions",
            "submitter_trainable_parameter_count": 0,
            "components": [
                {
                    "id": component_id,
                    "family": model_type,
                    "role": f"Prototype {dataset_id} field predictor.",
                    "description": (
                        f"Aggregate placeholder for the named {model} architecture. "
                        f"{family_note} No unverified internal split is invented."
                    ),
                    "parameter_count": total_parameters,
                }
            ],
            "key_hyperparameters": [
                {
                    "id": "reported-model-variant",
                    "component_ids": [component_id],
                    "name": "reported_model_variant",
                    "value": model,
                    "description": (
                        "Only the named prototype model variant is recorded; exact "
                        "depth, width, attention, neighborhood, and optimizer settings "
                        "were not encoded in this dummy submission."
                    ),
                }
            ],
            "input_features": [
                {
                    "id": "native-geometry",
                    "component_ids": [component_id],
                    "name": "native_geometry_coordinates",
                    "domain": "mesh",
                    "component_count": spatial_dimension,
                    "description": (
                        f"{spatial_dimension}D geometry or mesh coordinates associated "
                        "with the dataset's official supports. Exact feature engineering "
                        "was not recorded for this prototype row."
                    ),
                }
            ],
            "predicted_fields": predicted_fields,
        },
        "data_handling": {
            "normalization": (
                "Not recorded for this prototype fixture; no normalization statistics "
                "are asserted."
            ),
            "preprocessing": (
                "Not recorded for this prototype fixture; no remeshing or feature "
                "construction procedure is asserted."
            ),
            "sampling": (
                "Not recorded for this prototype fixture; the leaderboard's dummy "
                "coverage must not be interpreted as the named model's sampling policy."
            ),
        },
        "training": {
            "stages": [
                {
                    "id": "prototype-training-not-recorded",
                    "status": "prototype_not_recorded",
                    "component_ids": [component_id],
                    "description": (
                        "No executable training run, optimizer configuration, random "
                        "seed, or training-compute receipt accompanies this dummy row."
                    ),
                }
            ]
        },
        "checkpoints": [],
        "inference_compute": {
            "status": "not_measured",
            "reason": (
                "The prototype leaderboard row was not produced by a retained model "
                "checkpoint, so end-to-end inference time and hardware are unavailable."
            ),
        },
    }
