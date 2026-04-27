FROM python:3.13-slim

# petsc4py/slepc4py require PETSc configured with complex scalars and MUMPS.
# The petsc extra downloads and compiles PETSc, SLEPc, and MUMPS from source;
# the first build typically takes about 30-60 minutes.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=src \
    PETSC_CONFIGURE_OPTIONS="--with-scalar-type=complex --download-mumps --download-metis --download-parmetis --download-scalapack --download-fblaslapack" \
    SLEPC_CONFIGURE_OPTIONS="--with-clean"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        gfortran \
        make \
        libopenmpi-dev \
        openmpi-bin \
        liblapack-dev \
        libblas-dev \
        git \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency metadata first for Docker layer caching. uv.lock pins the
# exact Python dependency versions used by uv sync.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY tests/ ./tests/

# For CPU-only deployment, rebuild this layer with
# "uv sync --extra petsc --no-build-isolation" if the platform needs it.
RUN uv sync --extra petsc

CMD [".venv/bin/python", "-m", "pytest", "--tb=short", "-q", "tests/"]
