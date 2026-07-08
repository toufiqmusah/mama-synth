#!/usr/bin/env python3
"""MAMA-SYNTH — DualScaleSwinMedNeXt Translation via nnUNetPredictor.

Grand Challenge I/O contract
-----------------------------
Input  (one file):
    /input/images/pre-contrast-dce-mri-slice-breast/<uuid>.mha

Output (one file):
    /output/images/synthetic-contrast-dce-mri-slice-breast/<uuid>.mha

Environment overrides
---------------------
    MAMA_INPUT_DIR       default: /input
    MAMA_OUTPUT_DIR      default: /output
    MAMA_INPUT_SLUG      default: pre-contrast-dce-mri-slice-breast
    MAMA_PREDICTION_SLUG default: synthetic-contrast-dce-mri-slice-breast
    MAMA_MODEL_FOLDER    default: /opt/app/models/102
    MAMA_GPU_ID          default: 0  (set to -1 for CPU)
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import torch

INPUT_PATH = Path(os.environ.get("MAMA_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("MAMA_OUTPUT_DIR", "/output"))
INPUT_SLUG = os.environ.get(
    "MAMA_INPUT_SLUG", "pre-contrast-dce-mri-slice-breast"
)
OUTPUT_SLUG = os.environ.get(
    "MAMA_PREDICTION_SLUG", "synthetic-contrast-dce-mri-slice-breast"
)
MODEL_FOLDER = Path(
    os.environ.get("MAMA_MODEL_FOLDER", "/opt/app/models/102")
)

gpu_id = int(os.environ.get("MAMA_GPU_ID", "0"))
if gpu_id >= 0 and torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{gpu_id}")
else:
    DEVICE = torch.device("cpu")


def find_input_images() -> list[Path]:
    search_dir = INPUT_PATH / "images" / INPUT_SLUG
    candidates: list[str] = []
    for ext in ("*.mha", "*.nii.gz", "*.nii"):
        candidates.extend(glob(str(search_dir / ext)))
    if not candidates:
        raise FileNotFoundError(f"No input image found in {search_dir}")
    return sorted(Path(c) for c in candidates)


def run() -> int:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    print("=" * 60)
    print("  MAMA-SYNTH — DualScaleSwinMedNeXt (nnUNetPredictor)")
    print("=" * 60)
    print(f"  Device : {DEVICE}")

    input_files = find_input_images()
    print(f"  Found {len(input_files)} input(s)")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=DEVICE,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_FOLDER),
        use_folds=("all",),
        checkpoint_name="checkpoint_best.pth",
    )
    print(f"  Model   : {MODEL_FOLDER}")
    print(f"  Trainer : {predictor.trainer_name}")

    for input_file in input_files:
        print(f"\n--- {input_file.name} ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor.predict_from_files(
                [[str(input_file)]],
                tmpdir,
                overwrite=True,
                save_probabilities=False,
                folder_with_segs_from_prev_stage=None,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )
            outputs = list(Path(tmpdir).iterdir())
            if not outputs:
                raise RuntimeError("nnUNetPredictor produced no output files")
            raw_out = outputs[0]
            out_path = OUTPUT_PATH / "images" / OUTPUT_SLUG / input_file.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(raw_out), str(out_path))
            print(f"  Output : {out_path}")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run())
