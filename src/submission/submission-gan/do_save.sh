#!/usr/bin/env bash
# Export the GAN baseline container for upload to Grand Challenge.
#
# Bump VERSION before each upload to distinguish different builds.
#
# Upload instructions:
#   1. Run this script to produce the .tar.gz archive.
#   2. On grand-challenge.org → your Algorithm → Container Management
#      → Upload a new container.
#   3. Upload the produced .tar.gz file.
#   4. Wait for GC to validate the image (up to 24 h).
#   5. Submit the algorithm to the desired challenge phase.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

VERSION="v1.0.0"
OUT_FILE="$SCRIPT_DIR/mama-synth-gan-baseline-${VERSION}.tar.gz"

# Pass --no-rebuild (or set NO_REBUILD=1) to skip the docker build step and
# just re-export whichever image is already tagged mama-synth-gan-baseline.
# Useful when the image is already built and you only need the .tar.gz:
#   ./do_save.sh --no-rebuild
NO_REBUILD="${NO_REBUILD:-0}"
for arg in "$@"; do
    [ "$arg" = "--no-rebuild" ] && NO_REBUILD=1
done

if [ "$NO_REBUILD" != "1" ]; then
    bash "$SCRIPT_DIR/do_build.sh"
else
    echo "Skipping rebuild (--no-rebuild / NO_REBUILD=1)."
    if ! docker image inspect mama-synth-gan-baseline >/dev/null 2>&1; then
        echo "ERROR: Image mama-synth-gan-baseline not found. Run do_build.sh first."
        exit 1
    fi
fi

docker save mama-synth-gan-baseline | gzip -c > "$OUT_FILE"
echo ""
echo "Saved to $OUT_FILE"
echo "Upload this file to GC → Algorithm → Container Management."
