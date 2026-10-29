"""Candidate native WindsorML evaluator support.

The package is deliberately candidate-only while the velocity-profile stations
and score composition remain under owner review.  A result produced by these
modules is therefore development evidence, never an official submission.
"""

from .contract import (
    WindsorMLContractError,
    WindsorMLForceTruth,
    WindsorMLSourceCase,
    WindsorMLSourceIdentity,
    load_source_identity,
)

__all__ = [
    "WindsorMLContractError",
    "WindsorMLForceTruth",
    "WindsorMLSourceCase",
    "WindsorMLSourceIdentity",
    "load_source_identity",
]
