FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    make \
    perl \
    wget \
    ca-certificates \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ROCm AMD WSL userspace.
RUN cd /tmp && \
    wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/jammy/amdgpu-install_7.2.70200-1_all.deb && \
    apt-get update && \
    apt-get install -y ./amdgpu-install_7.2.70200-1_all.deb && \
    amdgpu-install -y --usecase=wsl,rocm --no-dkms && \
    apt-get install -y rocminfo rocm-device-libs && \
    rm -rf /var/lib/apt/lists/*

# HIPIFY
RUN git clone https://github.com/ROCm/HIPIFY.git /opt/HIPIFY && \
    HIPIFY_PERL_PATH=$(find /opt/HIPIFY -name "hipify-perl" -type f | head -n 1) && \
    chmod +x "$HIPIFY_PERL_PATH" && \
    ln -sf "$HIPIFY_PERL_PATH" /usr/local/bin/hipify-perl

ENV HIP_PLATFORM=amd
ENV HSA_ENABLE_DXG_DETECTION=1
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/wsl/lib:/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /work