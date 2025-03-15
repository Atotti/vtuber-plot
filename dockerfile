FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ARG PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive

ENV HOME /app
WORKDIR $HOME

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        liblzma-dev \
        liblzma-dev \
        libffi-dev \
        curl \
        clang \
        git-lfs \
        make \
        cmake \
        pkg-config \
        libgoogle-perftools-dev

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv python install $PYTHON_VERSION
ENV HF_HUB_CACHE /nfs/.cache/huggingface

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH /root/.cargo/bin:$PATH

ENV VIRTUAL_ENV /nfs/.gpu-venv
ENV UV_VIRTUAL_ENV /nfs/.gpu-venv
ENV UV_PROJECT_ENVIRONMENT /nfs/.gpu-venv


CMD ["sleep", "infinity"]
