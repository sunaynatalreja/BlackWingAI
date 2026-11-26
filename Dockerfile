# syntax=docker/dockerfile:1
# BlackWing AI - single self-contained image (API + UI + RAG index builder + tests).
# python:3.12-slim is multi-arch (linux/amd64 + linux/arm64): Intel/AMD, Apple Silicon, Windows, Linux.
# pyproject.toml requires Python >=3.11, so 3.12 is valid.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf_cache \
    PYTHONPATH=/app
WORKDIR /app

# ---------------------------------------------------------------
# 0. System packages (TLS roots for the IIT HTTPS endpoints)
# ---------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------
# 1. Python dependencies - installed straight from the 3 requirements files.
#    Only these files are copied first, so this (slow) layer stays cached
#    until a requirements file changes.
# ---------------------------------------------------------------
COPY requirements.txt requirements-rag.txt requirements-dev.txt /tmp/reqs/

# sentence-transformers needs PyTorch; install the CPU-only build first (no multi-GB CUDA libs)
# so the requirements below see torch as already satisfied.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# requirements.txt      -> runtime: API + UI + agents
# requirements-rag.txt  -> RAG index building / chunking
# Runtime and RAG dependencies are resolved together for compatible pins.
RUN pip install --retries 10 --timeout 60 \
    -r /tmp/reqs/requirements.txt \
    -r /tmp/reqs/requirements-rag.txt \
    requests

# ---------------------------------------------------------------
# 4. Sanity check: all pins consistent + every direct import works
# ---------------------------------------------------------------
RUN pip check
RUN python -c "import fastapi, fastembed, httpx, langgraph, langsmith, pydantic, streamlit, uvicorn, transformers, pypdf, reportlab, openpyxl, numpy, fontTools, sentence_transformers, pymupdf, requests; print('runtime imports OK')"

# Tokenizer files only (no model weights), per README. Needs internet at build time.
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')"
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')"

# Local BGE model weights (fastembed/ONNX, ~130 MB) used by BLACKWING_METADATA_V2_CREATION.py (make_embeddings('local')).
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# ---------------------------------------------------------------
# 5. The whole project (see .dockerignore: only .git, venvs, caches and secrets are left out)
#    Code runs from /app via PYTHONPATH, so its relative paths (datasets/, work/, config/) resolve.
# ---------------------------------------------------------------
RUN useradd -m -u 1000 bw && mkdir -p /data /app/work /app/rag_cache /app/rag_indexes /app/logs /app/cache \
    && chown -R bw:bw /data /app /opt/hf_cache /opt/fastembed_cache
COPY --chown=bw:bw . /app
# Code expects datasets/BLACKWING_DATA; the extracted dataset lives in DATASETS_ATH.
RUN ln -sfn /app/DATASETS_ATH /app/datasets/BLACKWING_DATA

USER bw
ENV BLACKWING_CONFIG=/app/config/blackwing.properties \
    BLACKWING_API_HOST=0.0.0.0 \
    BLACKWING_API_PORT=8000 \
    BLACKWING_DATABASE_PATH=/data/blackwing.sqlite3 \
    BLACKWING_RAG_CACHE=/app/rag_cache/embeddings_iit.sqlite \
    BLACKWING_RAG_INDEX_DIR=/app/rag_indexes/sunayana_iit_bge_v1 \
    BLACKWING_GOLD_DIR=/app/DATASETS_ATH/02_GOLD_EVALUATION_PROTECTED \
    BLACKWING_LLM_CA_CERT=/app/certs/llm-ca.crt \
    BLACKWING_EMBEDDING_CA_CERT=/app/certs/embeddings-ca.crt \
    BLACKWING_EMBEDDING_URL=https://103.42.50.41:1035/v1 \
    BLACKWING_EMBEDDING_MODEL=bge-small-en-v1.5 \
    BLACKWING_LLM_PROVIDER=openai_compatible \
    BLACKWING_LLM_API_MODEL=Qwen/Qwen3-8B \
    BLACKWING_UI_TIMEOUT_SECONDS=1800 \
    BLACKWING_UI_API_URL=http://127.0.0.1:8000 \
    BLACKWING_LLM_API_TIMEOUT_SECONDS=300 \
    BLACKWING_LLM_API_TEMPERATURE=0.1 \
    BLACKWING_LLM_API_MAX_OUTPUT_TOKENS=768 \
    BLACKWING_EMBEDDING_TIMEOUT=60 \
    BLACKWING_EMBEDDING_RETRIES=2 \
    BLACKWING_RAG_BACKEND=iit

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/v1/openapi.json')" || exit 1

# Default role = API. UI role: override the command (see docker-compose.yml / DEPLOY.md).
CMD ["python", "-m", "uvicorn", "blackwing.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Keep test tooling out of the default production image.
FROM runtime AS test
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --retries 10 --timeout 60 -r /tmp/requirements-dev.txt
CMD ["python", "-m", "pytest", "-q"]

# Make the production target the default for `docker build .`.
FROM runtime AS production
