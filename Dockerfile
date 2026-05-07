# Copyright@SCLBD
# This Dockerfile aims to build the base image for Deepfakbench.
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

LABEL maintainer="Deepfake"

# Install dependencies outside of the base image
RUN DEBIAN_FRONTEND=noninteractive apt update && \
	apt install -y --no-install-recommends automake \
    build-essential  \
    ca-certificates  \
    libfreetype6-dev  \
    libgl1  \
    libtool  \
    pkg-config  \
    python-dev  \
    python-distutils-extra \
    python3.7-dev  \
    python3-pip \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
	&& \
    rm -rf /var/lib/apt/lists/* \
    && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.7 0  \
    && \
    python3.7 -m pip install pip --upgrade 

WORKDIR /

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python dependencies
RUN python3.7 -m pip install --no-cache-dir --upgrade pip setuptools wheel certifi \
    && \
    python3.7 -m pip --no-cache-dir install dlib==19.24.0\
    imageio==2.9.0\
    albumentations==1.1.0\
    imgaug==0.4.0\
    scipy==1.7.3\
    seaborn==0.11.2\
    pyyaml==6.0\
    imutils==0.5.4\
    opencv-python==4.6.0.66\
    scikit-image==0.19.2\
    scikit-learn==1.0.2\
    efficientnet-pytorch==0.7.1\
    timm==0.6.12\
    segmentation-models-pytorch==0.3.2\
    transformers==4.30.2\
    safetensors==0.3.3\
    torchtoolbox==0.1.8.2\
    tensorboard==2.10.1

ENV MODEL_NAME=deepfakebench

# Expose port
EXPOSE 6000
