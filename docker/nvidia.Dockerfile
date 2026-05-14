FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    make \
    perl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# HIP NVIDIA backend, matching the local working setup.
RUN cd /tmp && \
    wget https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb && \
    apt-get update && \
    apt-get install -y ./amdgpu-install_6.2.60204-1_all.deb && \
    apt-get update && \
    apt-get install -y hip-runtime-nvidia hip-dev && \
    rm -rf /var/lib/apt/lists/*

# HIPIFY
RUN git clone https://github.com/ROCm/HIPIFY.git /opt/HIPIFY && \
    HIPIFY_PERL_PATH=$(find /opt/HIPIFY -name "hipify-perl" -type f | head -n 1) && \
    chmod +x "$HIPIFY_PERL_PATH" && \
    ln -sf "$HIPIFY_PERL_PATH" /usr/local/bin/hipify-perl

ENV CUDA_PATH=/usr/local/cuda-12.8
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV CUDACXX=/usr/local/cuda-12.8/bin/nvcc
ENV HIP_PLATFORM=nvidia
ENV PATH=/opt/rocm/bin:/usr/local/cuda-12.8/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

WORKDIR /work