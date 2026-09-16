"""HiLiftAeroML dataset-specific reference contracts."""

from .regional_aggregate import (
    AGGREGATE_REGIONAL_REPORT_SCHEMA,
    REGIONAL_DEFINITION_ID,
    REGIONAL_DIAGNOSTICS_CONTRACT_SHA256,
    HiLiftRegionalAggregateError,
    validate_aggregate_regional_diagnostics,
)

__all__ = [
    "AGGREGATE_REGIONAL_REPORT_SCHEMA",
    "REGIONAL_DEFINITION_ID",
    "REGIONAL_DIAGNOSTICS_CONTRACT_SHA256",
    "HiLiftRegionalAggregateError",
    "validate_aggregate_regional_diagnostics",
]
