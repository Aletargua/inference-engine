# Base image: Ubuntu 24.04 (Noble Numbat)
FROM ubuntu:24.04

# Evitar prompts interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Actualizar e instalar herramientas base, compilador y librerías
RUN apt-get update && apt-get install -y \
    build-essential \
    g++-14 \
    cmake \
    ninja-build \
    git \
    libtbb-dev \
    libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

# Configurar GCC 14 como el compilador por defecto
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100

# Directorio de trabajo
WORKDIR /workspace

# Comando por defecto (te deja en la terminal para compilar manualmente)
CMD ["/bin/bash"]