# BlackWing AI – Docker deployment (macOS, Windows, Linux)

Same commands on every OS. Only the Docker install and the shell differ.

## 0. Prerequisites
| OS | Install |
|---|---|
| macOS | Docker Desktop (Intel or Apple Silicon) |
| Windows 10/11 | Docker Desktop with WSL2 backend. Run commands in PowerShell, or inside WSL for best file speed |
| Linux | Docker Engine + Compose plugin (`docker compose version` must be v2.24+) |

Also needed: network access to the IIT LLM and embedding endpoints (campus network or VPN if required), and about 4 GB free disk.

## 1. Folder layout (project root = this folder)
```
DATASETS_ATH/      extracted dataset (01_RAG_KNOWLEDGE_BASE ... 06_*), baked into the image
certs/             llm-ca.crt, embeddings-ca.crt   (you provide; gitignored)
rag_indexes/       generated vector index          (created in step 4)
.env               secrets/config                  (created in step 2; gitignored)
```

## 2. Configure
macOS / Linux: `cp .env.example .env`
Windows PowerShell: `Copy-Item .env.example .env`

Edit `.env`: set `BLACKWING_LLM_API_BASE_URL`, `BLACKWING_LLM_API_KEY`, `BLACKWING_EMBEDDING_API_KEY`.
Put the two CA certificates in `certs/` with the names above. Never commit `.env` or `certs/`.

## 3. Build the image
```
docker compose build
```
Needs internet. Every requirement from requirements.txt, requirements-rag.txt and requirements-dev.txt is
installed by its own RUN line in the Dockerfile (CPU-only PyTorch), then the Qwen/BGE tokenizer files are
downloaded from Hugging Face. Run the tests inside the image: `docker compose --profile tools run --rm tests`.

## 4. Build the RAG index (once, or whenever the dataset changes)
```
docker compose --profile tools run --rm rag-build
```
Sends document chunks to the IIT embedding endpoint; takes a while. It refuses to overwrite an existing
index. To rebuild, delete `rag_indexes/blackwing-kb-iit-bge-v1` first.

## 5. Start
```
docker compose up -d
docker compose ps          # api should become "healthy"
```
- UI: http://localhost:8501
- API docs: http://localhost:8000/api/v1/docs

## 6. Operate
```
docker compose logs -f api         # logs
docker compose restart             # after changing .env
docker compose down                # stop (data volumes kept)
docker compose down -v             # stop AND delete SQLite history + embedding cache
```
Back up the history: `docker run --rm -v blackwing_bw_data:/d -v "$PWD":/b alpine cp /d/blackwing.sqlite3 /b/`
(volume name is `<project-folder>_bw_data`; check with `docker volume ls`).

## 7. Deploy to a server
Option A, build on the server (simplest): copy the project folder (including `DATASETS_ATH`, `certs`, `.env`,
and `rag_indexes` if already built) to the server, then run steps 3-5.

Option B, build once, run anywhere (multi-architecture image through a registry):
```
docker login
docker buildx create --use --name bw 2>/dev/null || docker buildx use bw
docker buildx build --platform linux/amd64,linux/arm64 -t <registry>/blackwing-ai:1.0 --push .
```
On the server: copy only `docker-compose.yml`, `.env`, `certs/`, `DATASETS_ATH/`, `rag_indexes/`, and replace
`build: .` / `image: blackwing-ai:latest` with `image: <registry>/blackwing-ai:1.0` (for both api and ui), then
`docker compose up -d`.

Production notes:
- Run exactly one `api` container (SQLite plus the API-key rate limit). Do not scale it.
- The app has no user authentication. Put a reverse proxy (Caddy/nginx/Traefik) with HTTPS and auth in front,
  and don't expose port 8000 publicly. Remove the `ports` entry for `api` if only the UI needs it.
- The API uses Streamlit on 8501 for users; the UI calls the API internally at `http://api:8000`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | Start Docker Desktop / `sudo systemctl start docker` |
| `.env not found` | Step 2 |
| `rag-build` fails with SSL error | Check `certs/embeddings-ca.crt` and that the URL is reachable from inside Docker |
| API healthy but retrieval is BM25-only | Embedding endpoint/key/cert wrong; see `docker compose logs api` |
| Windows: slow or mount errors | Keep the project inside the WSL filesystem or enable file sharing for the drive in Docker Desktop |
| Port already in use | Change the left side of `"8000:8000"` / `"8501:8501"` in the compose file |

---
# Release: share the image, not files

Maintainer (once, needs Docker running + IIT access):
```
# 1. put llm-ca.crt, embeddings-ca.crt in certs/ and build the index (step 4 above) so rag_indexes/ is filled
docker login
docker buildx create --use --name bw 2>/dev/null || docker buildx use bw
docker buildx build --platform linux/amd64,linux/arm64 \
  -t <registry>/blackwing-ai:1.0 --push .
```
Use a PRIVATE registry. The image holds the whole project, including the dataset, the protected gold
evaluation files and the IIT CA certs, but no API keys (.env, .pem and .key files are excluded).

Recipient (Mac / Linux; on Windows PowerShell replace `\` with a backtick):
```
docker network create blackwing

docker run -d --name bw-api --network blackwing -p 8000:8000 --restart unless-stopped \
  -e BLACKWING_LLM_API_BASE_URL=<assigned url> \
  -e BLACKWING_LLM_API_KEY=<key> \
  -e BLACKWING_EMBEDDING_API_KEY=<key> \
  -v bw_data:/data \
  <registry>/blackwing-ai:1.0

docker run -d --name bw-ui --network blackwing -p 8501:8501 --restart unless-stopped \
  -e BLACKWING_UI_API_URL=http://bw-api:8000 \
  <registry>/blackwing-ai:1.0 \
  python -m streamlit run blackwing/ui.py --server.address 0.0.0.0 --server.port 8501
```
Open http://localhost:8501. Stop: `docker rm -f bw-ui bw-api`. History persists in volume `bw_data`.
