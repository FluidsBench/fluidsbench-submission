"""Regression tests for issues found in the first real-data participant run."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reference.airfrans import participant as w

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "examples/airfrans-v3-candidate/pack_native_predictions.py"


def test_worker_imports_the_airfrans_library_not_the_cli_script(tmp_path):
    """Running scripts/airfrans.py must not let the script shadow ``import airfrans``.

    The per-case worker is launched exactly like this, by file path. A stand-in
    ``airfrans`` package on PYTHONPATH must be the one imported; before the fix
    the script itself was found and the extractor failed with "'airfrans' is
    not a package".
    """
    stub = tmp_path / "stub" / "airfrans"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(
        'raise ImportError("stand-in airfrans library imported")\n'
    )
    pythonpath = os.pathsep.join(
        filter(None, [str(stub.parent), os.environ.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/airfrans.py"),
            "_evaluate-case",
            "--case-id", "case",
            "--prediction", str(tmp_path / "case.npz"),
            "--dataset-root", str(tmp_path),
            "--truth", str(tmp_path / "truth.json"),
            "--output", str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )
    assert result.returncode == 2
    assert "stand-in airfrans library imported" in result.stderr
    assert "is not a package" not in result.stderr


def load_template():
    spec = importlib.util.spec_from_file_location("airfrans_pack_template", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packing_template_writes_exactly_the_arrays_validate_entry_accepts(
    tmp_path, monkeypatch
):
    case_id = "case"
    internal = np.column_stack(
        [np.linspace(0.0, 1.0, 6), np.linspace(-1.0, 1.0, 6), np.zeros(6)]
    )
    airfoil = internal[:3]
    meshes = {f"{case_id}_internal.vtu": internal, f"{case_id}_aerofoil.vtp": airfoil}
    # The template reads native coordinates with PyVista; stand in for it.
    monkeypatch.setitem(
        sys.modules,
        "pyvista",
        SimpleNamespace(read=lambda path: SimpleNamespace(points=meshes[Path(path).name])),
    )
    template = load_template()
    (tmp_path / "dataset" / case_id).mkdir(parents=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    np.savez(raw / f"{case_id}_preds.npz", coords=internal[:, :2], pred_fields=np.ones((6, 3)))
    output = tmp_path / "packed"
    output.mkdir()

    template.pack_case(tmp_path / "dataset", raw, output, case_id, 1e-5)

    packed = output / f"{case_id}.npz"
    with np.load(packed) as data:
        assert set(data.files) == w.PREDICTION_KEYS
    arrays = w.validate_prediction(packed, (len(internal), len(airfoil)))
    np.testing.assert_array_equal(arrays["internal_points"], internal)
    np.testing.assert_array_equal(arrays["airfoil_points"], airfoil)
