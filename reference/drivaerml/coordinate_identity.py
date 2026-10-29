"""Canonical identities for submitted DrivAerML profile coordinates.

The identity encoding is deliberately independent of JSON spelling, host byte
order, and platform-native floating-point serialization.  Its byte stream is:

``DOMAIN || uint64_be(count) || binary64_be(value_0) || ...``

where ``DOMAIN`` is the NUL-terminated ASCII constant below, ``uint64_be`` is
an unsigned eight-byte big-endian count, and each value is one IEEE-754
binary64 encoded in big-endian order.  Non-finite values and booleans are
rejected.  Both signs of zero are encoded as positive zero.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable


COORDINATE_ARRAY_IDENTITY_DOMAIN = (
    b"fluidsbench-drivaerml-coordinate-array-v1\x00"
)


class CoordinateIdentityError(ValueError):
    """Raised when a coordinate array has no canonical identity."""


def canonical_coordinate_array_bytes(values: Iterable[object]) -> bytes:
    """Return the canonical, platform-independent encoding of ``values``."""

    try:
        raw_values = list(values)
    except TypeError as error:
        raise CoordinateIdentityError("coordinate array must be iterable") from error
    if len(raw_values) > (1 << 64) - 1:
        raise CoordinateIdentityError("coordinate array is too long")

    encoded = bytearray(COORDINATE_ARRAY_IDENTITY_DOMAIN)
    encoded.extend(struct.pack(">Q", len(raw_values)))
    for index, raw_value in enumerate(raw_values):
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise CoordinateIdentityError(
                f"coordinate {index} must be an integer or binary64 value"
            )
        try:
            value = float(raw_value)
        except (OverflowError, ValueError) as error:
            raise CoordinateIdentityError(
                f"coordinate {index} cannot be represented as binary64"
            ) from error
        if not math.isfinite(value):
            raise CoordinateIdentityError(f"coordinate {index} must be finite")
        if value == 0.0:
            value = 0.0
        encoded.extend(struct.pack(">d", value))
    return bytes(encoded)


def coordinate_array_identity_sha256(values: Iterable[object]) -> str:
    """Return the lowercase SHA-256 of the canonical coordinate-array bytes."""

    return hashlib.sha256(canonical_coordinate_array_bytes(values)).hexdigest()
