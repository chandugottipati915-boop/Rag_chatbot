import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

"""Chunk, embed and load the PDFs in data/raw into Qdrant.

  uv run python scripts/ingest.py           # only new or changed PDFs
  uv run python scripts/ingest.py --force   # re-index everything
"""

from rag.ingestion.pipeline import run

if __name__ == "__main__":
    run(force="--force" in sys.argv)
