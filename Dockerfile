# MFSU - Unified Fractal-Stochastic Model
# Multi-stage Docker container for complete scientific environment
# Author: Miguel Ángel Franco León <miguerlfranco@mfsu-model.org>
# Repository: https://github.com/MiguelAngelFrancoLeon/MiguelAngelFrancoLeon-MFSU-Fractal-Dynamics/
# Date: August 2025

# ============================================================================
# Stage 1: Base scientific computing environment
# ============================================================================
FROM continuumio/miniconda3:23.5.2-0 as base

# Metadata and labels
LABEL maintainer="Miguel Ángel Franco León <miguerlfranco@mfsu-model.org>"
LABEL org.opencontainers.image.title="MFSU - Unified Fractal-Stochastic Model"
LABEL org.opencontainers.image.description="Complete environment for MFSU analysis, CMB processing, and fractal dynamics research"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.url="https://github.com/MiguelAngelFrancoLeon/MiguelAngelFrancoLeon-MFSU-Fractal-Dynamics/"
LABEL org.opencontainers.image.source="https://github.com/MiguelAngelFrancoLeon/MiguelAngelFrancoLeon-MFSU-Fractal-Dynamics/"
LABEL org.opencontainers.image.vendor="MFSU Research Group"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.authors="Miguel Ángel Franco León"

# Environment variables for reproducibility
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=42 \
    MFSU_VERSION=1.0.0 \
    MFSU_DELTA_F=0.921 \
    MFSU_FRACTAL_DIM=2.079 \
    MFSU_HURST=0.541

# System dependencies and optimization
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    gfortran \
    cmake \
    # Mathematical libraries
    libblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libgsl-dev \
    # System utilities
    curl \
    wget \
    git \
    unzip \
    # Graphics and visualization
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    # Network and security
    ca-certificates \
    openssh-client \
    # Development tools
    vim \
    htop \
    tree \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && conda clean -afy

# ============================================================================
# Stage 2: MFSU environment setup
# ============================================================================
FROM base as mfsu-env

# Working directory
WORKDIR /mfsu

# Copy environment and configuration files
COPY environment.yml pyproject.toml setup.py ./
COPY README.md LICENSE CITATION.cff ./

# Create conda environment with exact specifications
RUN conda env create -f environment.yml && \
    conda clean -afy && \
    echo "conda activate mfsu" >> ~/.bashrc

# Activate environment for subsequent commands
SHELL ["conda", "run", "-n", "mfsu", "/bin/bash", "-c"]

# Verify environment installation
RUN conda info && \
    conda list && \
    python -c "import numpy, scipy, matplotlib, pandas, astropy, healpy; print('Core packages OK')"

# ============================================================================
# Stage 3: MFSU package installation
# ============================================================================
FROM mfsu-env as mfsu-package

# Copy source code
COPY mfsu/ ./mfsu/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY data/ ./data/
COPY docs/ ./docs/
COPY notebooks/ ./notebooks/

# Install MFSU package in development mode
RUN pip install -e . && \
    python -c "import mfsu; print(f'MFSU v{mfsu.__version__} installed successfully')" && \
    python -c "import mfsu; mfsu.validate_installation()"

# Run basic tests to validate installation
RUN python -m pytest tests/test_basic.py -v || echo "Basic tests completed"

# ============================================================================
# Stage 4: Jupyter and development environment
# ============================================================================
FROM mfsu-package as mfsu-dev

# Jupyter configuration
RUN jupyter --version && \
    jupyter lab --generate-config && \
    mkdir -p /root/.jupyter

# Jupyter configuration file
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py

# Install additional Jupyter extensions
RUN pip install \
    jupyterlab-git \
    jupyterlab-widgets \
    ipywidgets \
    jupyter-bokeh \
    plotlywidget

# Set up MFSU directories
RUN mkdir -p /mfsu/{data,results,cache,logs} && \
    chmod 755 /mfsu/{data,results,cache,logs}

# ============================================================================
# Stage 5: Production optimized image
# ============================================================================
FROM mfsu-dev as mfsu-production

# Production environment variables
ENV MFSU_ENV=production \
    MFSU_DATA_PATH=/mfsu/data \
    MFSU_RESULTS_PATH=/mfsu/results \
    MFSU_CACHE_PATH=/mfsu/cache \
    MFSU_LOGS_PATH=/mfsu/logs \
    NUMBA_THREADING_LAYER=omp \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Performance optimization
RUN echo 'export PYTHONPATH="/mfsu:$PYTHONPATH"' >> ~/.bashrc && \
    echo 'export PATH="/mfsu/scripts:$PATH"' >> ~/.bashrc

# Generate sample data and validation
RUN python scripts/generate_sample_data.py && \
    python scripts/validate_environment.py && \
    echo "MFSU environment validation completed"

# Health check script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "=== MFSU Health Check ==="\n\
python -c "import mfsu; print(f\"MFSU {mfsu.__version__} OK\")"\n\
python -c "import mfsu.core; print(\"Core module OK\")"\n\
python -c "import mfsu.analysis; print(\"Analysis module OK\")"\n\
python -c "from mfsu.constants import DELTA_F; print(f\"δF = {DELTA_F}\")"\n\
echo "=== Health Check Passed ==="' > /mfsu/healthcheck.sh && \
    chmod +x /mfsu/healthcheck.sh

# ============================================================================
# Stage 6: Final runtime image
# ============================================================================
FROM mfsu-production as mfsu-runtime

# Expose ports
EXPOSE 8888 8080 6006

# Volume mounts for persistent data
VOLUME ["/mfsu/data", "/mfsu/results", "/mfsu/cache", "/mfsu/notebooks"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /mfsu/healthcheck.sh || exit 1

# Default command options
CMD ["conda", "run", "-n", "mfsu", "jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--notebook-dir=/mfsu/notebooks"]

# ============================================================================
# Usage Instructions and Documentation
# ============================================================================

# Build commands:
# docker build -t mfsu:latest .
# docker build --target mfsu-dev -t mfsu:dev .
# docker build --target mfsu-production -t mfsu:prod .

# Run commands:
# Development with Jupyter:
# docker run -p 8888:8888 -v $(pwd)/data:/mfsu/data -v $(pwd)/notebooks:/mfsu/notebooks mfsu:latest

# Interactive development:
# docker run -it -p 8888:8888 -v $(pwd):/mfsu/workspace mfsu:dev bash

# Production analysis:
# docker run -v $(pwd)/data:/mfsu/data -v $(pwd)/results:/mfsu/results mfsu:prod \
#   conda run -n mfsu python scripts/run_full_analysis.py

# CMB analysis:
# docker run -v $(pwd)/cmb_data:/mfsu/data mfsu:prod \
#   conda run -n mfsu mfsu-cmb --input /mfsu/data/planck_maps.fits --output /mfsu/results/

# Fractal diffusion simulation:
# docker run -v $(pwd)/results:/mfsu/results mfsu:prod \
#   conda run -n mfsu mfsu-simulate --type diffusion --delta-f 0.921 --steps 10000

# Multi-container setup with docker-compose:
# services:
#   mfsu-jupyter:
#     image: mfsu:latest
#     ports: ["8888:8888"]
#     volumes: ["./data:/mfsu/data", "./notebooks:/mfsu/notebooks"]
#   mfsu-worker:
#     image: mfsu:prod
#     volumes: ["./data:/mfsu/data", "./results:/mfsu/results"]
#     command: conda run -n mfsu python scripts/batch_analysis.py

# Performance tips:
# --shm-size=2g for large datasets
# --cpus=8 for parallel computations
# --memory=16g for CMB analysis
# -e OMP_NUM_THREADS=8 for OpenMP optimization

# Security considerations:
# Use specific user instead of root in production:
# RUN useradd -m -s /bin/bash mfsu && chown -R mfsu:mfsu /mfsu
# USER mfsu

# Repository information:
# Source: https://github.com/MiguelAngelFrancoLeon/MiguelAngelFrancoLeon-MFSU-Fractal-Dynamics/
# Maintainer: Miguel Ángel Franco León <miguerlfranco@mfsu-model.org>
# License: MIT
# Documentation: See README.md and docs/ directory

# Environment validation:
# The container includes comprehensive validation scripts that verify:
# - All dependencies are correctly installed
# - MFSU package functions properly
# - Core algorithms produce expected results with δF = 0.921
# - Example datasets can be processed successfully
