FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# libgomp1 is required by PyTorch/sentence-transformers at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.runtime.txt ./

# Install the CPU-only PyTorch wheel first.  The torch requirement in
# requirements.txt is then already satisfied and will not pull CUDA libraries.
RUN python -m pip install --upgrade pip
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0"
RUN python -m pip install -r requirements.runtime.txt

COPY . .

RUN mkdir -p /app/data /app/vector_db /app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
