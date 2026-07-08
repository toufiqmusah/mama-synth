# MAMA-SYNTH — DualScaleSwinMedNeXt Translation Model
FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# nn-translation source (staged by do_build.sh from sibling dir)
COPY --chown=user:user nn-translation/ /opt/app/nn-translation/

# Install nnunetv2 and its dependencies
RUN pip install --no-cache-dir -e /opt/app/nn-translation/

# Install MedNeXt blocks (required by DualScaleSwinMedNeXt)
RUN pip install --no-cache-dir git+https://github.com/MIC-DKFZ/MedNeXt.git

# Additional runtime deps
COPY --chown=user:user requirements.txt /opt/app/
RUN pip install --no-cache-dir --requirement /opt/app/requirements.txt

# Model folder (staged by do_build.sh — contains dataset.json, plans.json, fold_X/)
COPY --chown=user:user models/ /opt/app/models/

# Inference entry point
COPY --chown=user:user inference.py /opt/app/

# nn-translation root-level imports (dual_scale_swin_mednext.py, swin_unetr.py)
# sit at the nn-translation package root
ENV PYTHONPATH=/opt/app/nn-translation:${PYTHONPATH}

RUN mkdir -p /output && chown user:user /output

USER user

ENTRYPOINT ["python", "inference.py"]
