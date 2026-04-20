# Dockerfile â€” luxe-digit
#
# Uses the official ROOT image (includes PyROOT bindings, Geant4-compatible
# ecosystem, and CERN-standard build tools) as a reproducible base.
#
# Build:  docker build -t luxe-digit .
# Run:    docker run --rm -v $(pwd)/data:/app/data luxe-digit

FROM rootproject/root:6.30.04-ubuntu22.04

LABEL org.opencontainers.image.title="luxe-digit"
LABEL org.opencontainers.image.description="ML-augmented digitization pipeline for the LUXE Gamma Beam Profiler"
LABEL org.opencontainers.image.source="https://github.com/salthekal/luxe-digit"
LABEL org.opencontainers.image.licenses="GPL-3.0-or-later"

# Python tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-venv \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# CPU-only torch by default â€” uncomment cuda line if GPU is desired
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -e ".[ml]" \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Copy everything else
COPY . .

# Default entrypoint runs the CLI; override for other commands
ENTRYPOINT ["python3", "-m", "luxedigit.cli"]
CMD ["--help"]

