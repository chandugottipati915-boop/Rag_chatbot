import hashlib
import time

from rag.config import settings
from rag.embeddings import get_embedder
from rag.ingestion.chunker import chunk_file, load_catalog
from rag import vectorstore as vs


def run(force: bool = False) -> None:
    embedder = get_embedder()
    vs.ensure_collection(embedder.dim)
    catalog = load_catalog(settings.raw_dir)

    pdfs = sorted(settings.raw_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs → collection '{vs.COLL}'\n")

    total = 0
    for pdf in pdfs:
        doc_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]
        if not force and vs.doc_is_indexed(doc_hash):
            print(f"skip   {pdf.name} (unchanged)")
            continue

        t0 = time.time()
        vs.delete_doc(pdf.stem)                      # remove old version if any
        chunks = chunk_file(pdf, catalog)
        vectors = embedder.embed_documents([c["text"] for c in chunks])
        vs.upsert_chunks(chunks, vectors)
        total += len(chunks)
        print(f"index  {pdf.name}: {len(chunks)} chunks in {time.time() - t0:.1f}s")

    print(f"\nDone. {total} new chunks. "
          f"Collection now has {vs.client.count(vs.COLL).count} points.")


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)