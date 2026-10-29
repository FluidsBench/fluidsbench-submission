from reference.hosting import support_url
import pytest


def test_release_transport_changes_without_touching_asset_identity():
    assert support_url("https://github.com/neilashton/fluidsbench-submission/releases/download/ahmedml-derived-cache-v1-dev/a.zip") == "https://github.com/FluidsBench/fluidsbench-submission/releases/download/ahmedml-derived-cache-v1/a.zip"
    other = "https://example.org/data/a.zip"
    assert support_url(other) == other
    with pytest.raises(ValueError):
        support_url("https://github.com/neilashton/fluidsbench-submission/releases/download/tag/a.zip?mutable=1")
