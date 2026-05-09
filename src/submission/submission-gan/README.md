# MAMA-SYNTH — Pix2PixHD GAN Baseline

A Grand Challenge algorithm submission that uses the **Pix2PixHD** model
(medigan model `00023`) to synthesise post-contrast breast DCE-MRI slices
from pre-contrast inputs.

## Quick start

### 1. Prerequisites

- Docker with NVIDIA GPU support (`nvidia-container-toolkit`)
- Model weights at `~/Desktop/baseline\ GAN/00023/30_net_G.pth`  
  (override: `export MODEL_WEIGHTS_DIR=/path/to/00023`)
- A test `.mha` slice in `test/input/images/pre-contrast-dce-mri-slice-breast/`

### 2. Build

```bash
./do_build.sh
```

`do_build.sh` stages the model weights from `MODEL_WEIGHTS_DIR` into
`models/00023/` (excluded from git) and then runs `docker build`.

### 3. Test locally

```bash
./do_test_run.sh           # GPU
USE_GPU=0 ./do_test_run.sh # CPU-only
```

Output: `test/output/images/synthetic-contrast-dce-mri-slice-breast/output.mha`

### 4. Run automated tests

```bash
pip install pytest SimpleITK numpy
pytest test_algorithm.py -v
```

### 5. Export for Grand Challenge upload

```bash
./do_save.sh
```

Upload the produced `.tar.gz` to:  
GC → Algorithm → Container Management → Upload a new container

---

## Normalisation

Input `.mha` files from GC are z-score normalised with the MAMA-SYNTH
pre-contrast reference statistics (`training_pre_stats.json`).  The
Pix2PixHD model expects 8-bit PNG [0, 255] inputs.  The pipeline:

| Step | Direction | Operation |
|------|-----------|-----------|
| 1 | MHA → PNG | Invert z-score: `raw = z × std + mean` (clip to [0, ∞)) |
| 2 | MHA → PNG | Per-image min-max scale to [0, 255] uint8 |
| 3 | PNG → MHA | Inverse scale using input's (raw\_min, raw\_max) |
| 4 | PNG → MHA | Re-apply z-score: `z = (raw − mean) / std` |

This matches the per-image normalisation used in the original medigan
reference synthesis pipeline while keeping outputs on the same z-score
scale as the evaluation pipeline.
