#!/usr/bin/env bash
# Run the DualScaleSwinMedNeXt container locally against test input.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

# Rebuild
bash "$SCRIPT_DIR/do_build.sh"

# Clean and prepare output dirs (must exist before container writes)
rm -rf "$SCRIPT_DIR/test/output"
mkdir -p "$SCRIPT_DIR/test/output/images/synthetic-contrast-dce-mri-slice-breast"
chmod 777 "$SCRIPT_DIR/test/output/images/synthetic-contrast-dce-mri-slice-breast"

# Verify test input exists
INPUT_DIR="$SCRIPT_DIR/test/input/images/pre-contrast-dce-mri-slice-breast"
if [ -z "$(ls -A "$INPUT_DIR"/*.mha 2>/dev/null)" ]; then
    echo "ERROR: No .mha file found in $INPUT_DIR"
    exit 1
fi

USE_GPU="${USE_GPU:-1}"
if [ "$USE_GPU" = "1" ]; then
    GPU_FLAG="--gpus device=0"
    GPU_ENV="-e MAMA_GPU_ID=0"
else
    GPU_FLAG=""
    GPU_ENV="-e MAMA_GPU_ID=-1"
fi

docker run --rm \
    --network=none \
    --memory=16g \
    --shm-size=8g \
    $GPU_FLAG \
    $GPU_ENV \
    -v "$SCRIPT_DIR/test/input:/input:ro" \
    -v "$SCRIPT_DIR/test/output:/output" \
    mama-synth-dual-scale-swin

echo ""
echo "=== Output ==="
find "$SCRIPT_DIR/test/output" -type f
