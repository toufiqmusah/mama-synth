#!/usr/bin/env bash
# Export the DualScaleSwinMedNeXt container for Grand Challenge upload.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

VERSION="${VERSION:-v1.0.0}"
OUT_FILE="$SCRIPT_DIR/mama-synth-dual-scale-swin-${VERSION}.tar.gz"

bash "$SCRIPT_DIR/do_build.sh"
docker save mama-synth-dual-scale-swin | gzip -c > "$OUT_FILE"
echo "Saved to $OUT_FILE"
