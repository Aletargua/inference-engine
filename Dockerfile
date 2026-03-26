FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    build-essential \
    g++-14 \
    cmake \
    ninja-build \
    git \
    libtbb-dev \
    libgtest-dev \
    python3 \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*


RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100


WORKDIR /workspace

CMD ["/bin/bash"]