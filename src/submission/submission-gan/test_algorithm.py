#!/usr/bin/env python3
"""Automated test for the Pix2PixHD GAN baseline container.

Runs do_test_run.sh and verifies that the output file is produced,
is a valid 2-D .mha image, and has z-score values in a plausible range.

Prerequisites:
  - Docker installed and running
  - GPU available OR environment variable USE_GPU=0 set
  - A z-score-normalised pre-contrast .mha file in:
      test/input/images/pre-contrast-dce-mri-slice-breast/

Run:
  pytest test_algorithm.py -v
  # or directly:
  python test_algorithm.py

CPU-only machines:
  USE_GPU=0 pytest test_algorithm.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import SimpleITK as sitk
import numpy as np

SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = (
    SCRIPT_DIR
    / "test" / "input" / "images" / "pre-contrast-dce-mri-slice-breast"
)
OUTPUT_DIR = (
    SCRIPT_DIR
    / "test" / "output" / "images" / "synthetic-contrast-dce-mri-slice-breast"
)

# Maximum acceptable z-score range for a breast DCE-MRI slice.
# Post-contrast images may have higher enhancement than pre-contrast,
# but values far beyond ±20 σ are almost certainly normalisation errors.
Z_SCORE_MAX_ABS = 50.0


def _run_container() -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Propagate USE_GPU for CPU-only CI environments
    return subprocess.run(
        ["bash", str(SCRIPT_DIR / "do_test_run.sh")],
        capture_output=True,
        text=True,
        cwd=str(SCRIPT_DIR),
        env=env,
    )


def test_input_file_present():
    """A test .mha file must exist before the container can be tested."""
    mha_files = glob(str(INPUT_DIR / "*.mha"))
    assert mha_files, (
        f"No .mha file found in {INPUT_DIR}.\n"
        "Place a z-score-normalised pre-contrast .mha slice there first.\n"
        "You can copy one from the identity-baseline test directory:\n"
        "  src/submission/identity-baseline/test/input/images/"
        "pre-contrast-dce-mri-slice-breast/"
    )


def test_container_runs_successfully():
    """Container must exit with code 0."""
    result = _run_container()
    assert result.returncode == 0, (
        f"do_test_run.sh exited with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_output_file_created():
    """Container must produce exactly one .mha output file."""
    output_file = OUTPUT_DIR / "output.mha"
    assert output_file.exists(), (
        f"Expected output not found at {output_file}\n"
        f"Files under test/output: {list((SCRIPT_DIR / 'test' / 'output').rglob('*'))}"
    )


def test_output_is_readable_2d_image():
    """Output .mha must be readable by SimpleITK and be a 2-D image."""
    output_file = OUTPUT_DIR / "output.mha"
    assert output_file.exists(), "Run test_container_runs_successfully() first."

    img = sitk.ReadImage(str(output_file))
    arr = sitk.GetArrayFromImage(img)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    assert arr.ndim == 2, f"Expected 2-D output array, got shape {arr.shape}"


def test_output_matches_input_shape():
    """Output spatial dimensions must match the input (no silent resizing)."""
    input_files = glob(str(INPUT_DIR / "*.mha"))
    assert input_files, "No input .mha found."

    in_img = sitk.ReadImage(input_files[0])
    out_img = sitk.ReadImage(str(OUTPUT_DIR / "output.mha"))

    assert in_img.GetSize() == out_img.GetSize(), (
        f"Shape mismatch: input {in_img.GetSize()} vs output {out_img.GetSize()}"
    )


def test_output_is_float32():
    """Output must be float32 (as expected by the evaluation pipeline)."""
    out_img = sitk.ReadImage(str(OUTPUT_DIR / "output.mha"))
    arr = sitk.GetArrayFromImage(out_img)
    assert arr.dtype == np.float32, (
        f"Expected float32 output, got {arr.dtype}"
    )


def test_output_zscore_range_plausible():
    """Output z-scores must be within a plausible range (not exploded NaN/inf)."""
    out_img = sitk.ReadImage(str(OUTPUT_DIR / "output.mha"))
    arr = sitk.GetArrayFromImage(out_img).astype(np.float64)

    assert not np.any(np.isnan(arr)), "Output contains NaN values"
    assert not np.any(np.isinf(arr)), "Output contains Inf values"

    abs_max = float(np.max(np.abs(arr)))
    assert abs_max < Z_SCORE_MAX_ABS, (
        f"Output z-scores suspiciously large (|max|={abs_max:.1f} > {Z_SCORE_MAX_ABS}). "
        "Check normalisation pipeline."
    )


def test_output_has_contrast_enhancement():
    """Synthesised post-contrast image should differ from the input (not identity)."""
    input_files = glob(str(INPUT_DIR / "*.mha"))
    assert input_files, "No input .mha found."

    in_arr = sitk.GetArrayFromImage(sitk.ReadImage(input_files[0])).astype(np.float64)
    out_arr = sitk.GetArrayFromImage(
        sitk.ReadImage(str(OUTPUT_DIR / "output.mha"))
    ).astype(np.float64)

    if in_arr.ndim == 3 and in_arr.shape[0] == 1:
        in_arr = in_arr[0]

    mse = float(np.mean((in_arr - out_arr) ** 2))
    # A pure identity output (MSE == 0) would mean the GAN did nothing.
    # We just verify there is *some* difference.
    assert mse > 1e-6, (
        f"Output is identical to input (MSE={mse:.2e}). "
        "The GAN may not have run correctly."
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
