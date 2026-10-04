# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
#
# Build and runtime image for Eke Hive.
#   docker build -t eke-hive -f docker/Dockerfile .
# The base image is NVIDIA's CUDA devel image (NVIDIA Deep Learning Container License);
# it is pulled from NVIDIA's registry and is not redistributed by this project.
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      cmake ninja-build g++ git ca-certificates \
      libnuma-dev numactl \
      python3 python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

# Python side: API server (FastAPI/Uvicorn), tokenizer and image preprocessing
# (transformers, Pillow, NumPy, PyTorch CPU build), video decoding for GLM (PyAV), oracle and tools (safetensors).
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && /opt/venv/bin/pip install -r /tmp/requirements.txt
ENV PATH=/opt/venv/bin:${PATH}

WORKDIR /hive
