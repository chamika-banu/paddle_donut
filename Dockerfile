FROM python:3.11-slim-bullseye

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set a stable path for HuggingFace model cache so it can be persisted via a Docker named volume across container restarts (see docker-compose.yml).
ENV HF_HOME=/data/models/huggingface

# Install Python dependencies.
# --index-url whl/cpu      forces torch to resolve to the CPU-only wheel.
# --extra-index-url simple makes all other packages (paddlepaddle, fastapi, etc.)
#                          resolvable from PyPI as a secondary source.
# Single-pass installation prevents a duplicate OpenMP runtime conflict that
# occurs when torch and paddlepaddle are installed in separate pip calls.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

# No flash_attn stub needed — Donut (VisionEncoderDecoderModel) does not
# import flash_attn at module load time, unlike modeling_florence2.py.

# Copy application files
COPY schemas.py .
COPY coordinator.py .
COPY services/ ./services/
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
