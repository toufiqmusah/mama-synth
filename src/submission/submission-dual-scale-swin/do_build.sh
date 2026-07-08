#!/usr/bin/env bash
# MAMA-SYNTH — DualScaleSwinMedNeXt Translation Model
#
# Usage:
#   ./do_build.sh
#
# Expects the following already present in this directory:
#   nn-translation/          (cloned sibling repo)
#   models/102/fold_all/     (trained checkpoint + dataset.json + plans.json)
#
# Optional:
#   DOCKER_NO_CACHE=1  ./do_build.sh    # force no-cache build
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

errors=0

# ── Validate nn-translation ─────────────────────────────────────────────────
if [ ! -d "$SCRIPT_DIR/nn-translation/nnunetv2" ]; then
    echo "ERROR: nn-translation/ not found in $SCRIPT_DIR"
    echo "  Clone it:  cd $SCRIPT_DIR && git clone <url> nn-translation"
    errors=1
fi

# ── Validate model folder ───────────────────────────────────────────────────
MODEL_DIR="$SCRIPT_DIR/models/102"
for f in dataset.json plans.json fold_all/checkpoint_best.pth; do
    if [ ! -f "$MODEL_DIR/$f" ]; then
        echo "ERROR: Missing $MODEL_DIR/$f"
        errors=1
    fi
done

if [ "$errors" -ne 0 ]; then
    exit 1
fi

echo "nn-translation found at $SCRIPT_DIR/nn-translation"
echo "Model folder validated at $MODEL_DIR"

# ── Build Docker image ──────────────────────────────────────────────────────
echo ""
echo "Building Docker image: mama-synth-dual-scale-swin"
docker build ${DOCKER_NO_CACHE:+--no-cache} -t mama-synth-dual-scale-swin "$SCRIPT_DIR"
echo ""
echo "Done.  Run:  ./do_test_run.sh"
