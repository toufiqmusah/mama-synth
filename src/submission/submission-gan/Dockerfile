# MAMA-SYNTH — Pix2PixHD GAN Baseline
#
# Uses the official PyTorch CUDA base image so that the medigan Pix2PixHD
# model can run on GPU (as configured for algorithm submissions on GC).
#
# Build: run do_build.sh (it stages the model weights first)
# Test:  run do_test_run.sh
FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Prevents Python stdout/stderr buffering (important for GC log streaming)
ENV PYTHONUNBUFFERED=1

# Create a non-root user (Grand Challenge security requirement)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# System libraries required by SimpleITK on slim/runtime images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
# torch + torchvision are already present in the base image;
# only additional packages are installed here.
COPY --chown=user:user requirements.txt /opt/app/
RUN pip install --no-cache-dir --requirement /opt/app/requirements.txt

# Pre-contrast normalisation statistics (used for MHA ↔ PNG conversion)
COPY --chown=user:user training_pre_stats.json /opt/app/

# Algorithm entry point
COPY --chown=user:user inference.py /opt/app/

# Pix2PixHD model weights + source
# (staged into models/00023/ by do_build.sh before docker build)
COPY --chown=user:user models/ /opt/app/models/

# GC mounts /output as read-write; ensure it exists and is owned by our user
RUN mkdir -p /output && chown user:user /output

USER user

ENTRYPOINT ["python", "inference.py"]
