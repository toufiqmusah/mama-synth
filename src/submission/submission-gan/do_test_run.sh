#!/usr/bin/env bash
# Run the GAN baseline container locally against a test input.
#
# Prerequisites:
#   • A z-score-normalised pre-contrast .mha file placed in:
#       test/input/images/pre-contrast-dce-mri-slice-breast/
#   • Docker installed and running.
#
# GPU note:
#   By default this script attempts to use GPU 0 via --gpus device=0.
#   If GPU is not available locally, set USE_GPU=0 to force CPU:
#       USE_GPU=0 ./do_test_run.sh
#
# After a successful run the synthetic output is at:
#   test/output/images/synthetic-contrast-dce-mri-slice-breast/output.mha
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

# Rebuild the image before each test run (picks up any code changes)
bash "$SCRIPT_DIR/do_build.sh"

# Clean previous output
rm -rf "$SCRIPT_DIR/test/output"
mkdir -p "$SCRIPT_DIR/test/output"

# Verify a test input file exists
INPUT_DIR="$SCRIPT_DIR/test/input/images/pre-contrast-dce-mri-slice-breast"
if [ -z "$(ls -A "$INPUT_DIR"/*.mha 2>/dev/null)" ]; then
    echo ""
    echo "ERROR: No .mha file found in:"
    echo "  $INPUT_DIR"
    echo ""
    echo "Place a z-score-normalised pre-contrast .mha slice there before"
    echo "running this script.  You can copy one from:"
    echo "  src/submission/identity-baseline/test/input/images/pre-contrast-dce-mri-slice-breast/"
    echo ""
    exit 1
fi

# GPU flag (override with USE_GPU=0 ./do_test_run.sh for CPU-only machines)
USE_GPU="${USE_GPU:-1}"
if [ "$USE_GPU" = "1" ]; then
    GPU_FLAG="--gpus device=0"
    GPU_ENV="-e MAMA_GPU_ID=0"
    echo "Running with GPU (USE_GPU=1)"
else
    GPU_FLAG=""
    GPU_ENV="-e MAMA_GPU_ID=-1"
    echo "Running on CPU (USE_GPU=0)"
fi

docker run --rm \
    --network=none \
    --memory=16g \
    $GPU_FLAG \
    $GPU_ENV \
    -v "$SCRIPT_DIR/test/input:/input:ro" \
    -v "$SCRIPT_DIR/test/output:/output" \
    mama-synth-gan-baseline

echo ""
echo "=== Output files ==="
find "$SCRIPT_DIR/test/output" -type f
