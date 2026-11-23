#!/usr/bin/env bash
# Run with: source scripts/use_iit.sh
# Set the assigned endpoint and group key in your shell before sourcing.
if [[ -z "${BLACKWING_LLM_API_BASE_URL:-}" ]]; then
  echo "Set BLACKWING_LLM_API_BASE_URL to the assigned group URL ending in /v1." >&2
  return 1 2>/dev/null || exit 1
fi
if [[ -z "${BLACKWING_LLM_API_KEY:-}" ]]; then
  echo "Set BLACKWING_LLM_API_KEY securely in this shell." >&2
  return 1 2>/dev/null || exit 1
fi
if [[ -z "${BLACKWING_EMBEDDING_API_KEY:-}" ]]; then
  echo "Set BLACKWING_EMBEDDING_API_KEY securely in this shell." >&2
  return 1 2>/dev/null || exit 1
fi
if [[ -z "${BLACKWING_LLM_CA_CERT:-}" ]]; then
  echo "Set BLACKWING_LLM_CA_CERT to the CA certificate assigned to your inference group." >&2
  return 1 2>/dev/null || exit 1
fi
[[ -f "$BLACKWING_LLM_CA_CERT" ]] || { echo "CA certificate not found: $BLACKWING_LLM_CA_CERT" >&2; return 1 2>/dev/null || exit 1; }
export BLACKWING_EMBEDDING_CA_CERT="${BLACKWING_EMBEDDING_CA_CERT:-$HOME/Downloads/VM02_Embeddings_Access_2/vm02-ca.crt}"
[[ -f "$BLACKWING_EMBEDDING_CA_CERT" ]] || { echo "Embeddings CA certificate not found: $BLACKWING_EMBEDDING_CA_CERT" >&2; return 1 2>/dev/null || exit 1; }
[[ "$BLACKWING_LLM_API_BASE_URL" == https://* ]] || { echo "The assigned model URL must use HTTPS." >&2; return 1 2>/dev/null || exit 1; }
export BLACKWING_EMBEDDING_URL="${BLACKWING_EMBEDDING_URL:-https://103.42.50.41:1035/v1}"

export BLACKWING_CONFIG="$PWD/config/blackwing.properties"
if [[ -z "${BLACKWING_GOLD_QUERY_FILE:-}" && -f "$PWD/BLACKWING_DATA/02_GOLD_EVALUATION_PROTECTED/GOLD_80_QUESTIONS/BW-GQ-001-GOLD80-WORKBOOK.xlsx" ]]; then
  export BLACKWING_GOLD_QUERY_FILE="$PWD/BLACKWING_DATA/02_GOLD_EVALUATION_PROTECTED/GOLD_80_QUESTIONS/BW-GQ-001-GOLD80-WORKBOOK.xlsx"
fi
export BLACKWING_API_HOST="127.0.0.1"
export BLACKWING_API_PORT="8000"
export BLACKWING_UI_API_URL="http://127.0.0.1:8000"
export BLACKWING_UI_TIMEOUT_SECONDS="1800"
export BLACKWING_LLM_PROVIDER="openai_compatible"
export BLACKWING_LLM_API_MODEL="Qwen/Qwen3-8B"
export BLACKWING_LLM_API_TIMEOUT_SECONDS="300"
export BLACKWING_LLM_API_TEMPERATURE="0.1"
export BLACKWING_LLM_API_MAX_OUTPUT_TOKENS="768"
export BLACKWING_API_JSON_RESPONSE_FORMAT="${BLACKWING_API_JSON_RESPONSE_FORMAT:-true}"
export BLACKWING_IIT_MIN_INTERVAL_SECONDS="1.1"
export BLACKWING_QWEN_TOKENIZER="Qwen/Qwen3-8B"
export BLACKWING_RAG_BACKEND="iit"
export BLACKWING_AGENT_RAG="${BLACKWING_AGENT_RAG:-1}"
export BLACKWING_AGENT_RAG_METHOD="${BLACKWING_AGENT_RAG_METHOD:-rerank}"
export BLACKWING_EMBEDDING_MODEL="bge-small-en-v1.5"
export BLACKWING_EMBEDDING_TIMEOUT="60"
export BLACKWING_EMBEDDING_RETRIES="2"
export BLACKWING_RAG_CACHE="$PWD/rag_cache/embeddings_iit.sqlite"
export BLACKWING_RAG_INDEX_DIR="${BLACKWING_RAG_INDEX_DIR:-$PWD/rag_indexes/blackwing-kb-iit-bge-v1}"
echo "Hosted Qwen3-8B configured. Confirm that the CA certificate matches the assigned group URL."
