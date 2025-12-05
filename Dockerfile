# Stable Diffusion with NVIDIA CUDA support for WSL2
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Use --extra-index-url for PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY generate.py generate_sdxl.py ./

# Create output directory
RUN mkdir -p /app/output

# Set default output directory
ENV OUTPUT_DIR=/app/output

# Default entrypoint - SD 2.1 generator
ENTRYPOINT ["python3", "generate.py"]
CMD ["--help"]
