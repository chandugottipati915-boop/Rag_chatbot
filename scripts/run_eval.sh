#!/usr/bin/env bash
# Ingest + evaluate one or more retrieval modes with the embedding provider set in .env.
# Usage:
#   bash scripts/run_eval.sh                        # dense and hybrid
#   bash scripts/run_eval.sh hybrid                 # only hybrid
#   bash scripts/run_eval.sh dense+rerank hybrid+rerank   # with the cross-encoder reranker
set -euo pipefail
cd "$(dirname "$0")/.."
modes=("${@:-dense hybrid}"); modes=(${modes[@]})

echo "== Checking Qdrant (hybrid needs >= 1.15.3 for built-in BM25)"
if ! version=$(curl -s --max-time 5 localhost:6333 | python3 -c "import sys, json; print(json.load(sys.stdin)['version'])"); then
  echo "Qdrant is not reachable on localhost:6333. Start it with: docker start qdrant"; exit 1
fi
echo "Qdrant $version"
if ! python3 - "$version" <<'PY'
import sys
v = tuple(int(x) for x in sys.argv[1].split(".")[:3])
sys.exit(0 if v >= (1, 15, 3) else 1)
PY
then
  echo "Qdrant $version is too old. Upgrading (your data volume is kept)..."
  docker pull qdrant/qdrant
  docker rm -f qdrant
  docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage --name qdrant qdrant/qdrant
  sleep 5
fi

provider=$(grep -E "^EMBEDDING_PROVIDER=" .env | cut -d= -f2 | tr -d ' "' || true)
provider=${provider:-local}
mkdir -p eval/results
log="eval/results/eval_${provider}_$(date +%Y%m%d-%H%M%S).log"
echo "== Embeddings: $provider | modes: ${modes[*]} | log: $log"

{
  for setup in "${modes[@]}"; do
    mode=${setup%%+*}                                   # "hybrid+rerank" -> "hybrid"
    rerank=false; [[ "$setup" == *+rerank ]] && rerank=true
    export RETRIEVAL_MODE=$mode RERANK=$rerank
    echo; echo "################ $provider / $mode / rerank=$rerank ################"
    echo "== Ingest ($mode)";      uv run python scripts/ingest.py
    echo "== Latency";             uv run python scripts/bench_latency.py
    echo "== Retrieval eval";      uv run python eval/eval_retriever.py
    echo "== Generator eval";      uv run python eval/eval_generator.py
  done
} 2>&1 | tee "$log"

echo "Done. Results saved in $log"
