import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import sys
from collections import Counter
from pathlib import Path

from rag.ingestion.chunker import chunk_file, load_catalog

from rag.config import settings
raw = settings.raw_dir
pdf = raw / (sys.argv[1] if len(sys.argv) > 1 else "2005.11401.pdf")
chunks = chunk_file(pdf, load_catalog(raw))

lengths = [len(c["text"]) for c in chunks]
print(f"\n{pdf.name}: {len(chunks)} chunks | "
      f"avg {sum(lengths)//len(lengths)} chars | min {min(lengths)} | max {max(lengths)}\n")

print("Sections found:")
for sec, n in Counter(c["metadata"]["section"] for c in chunks).items():
    print(f"  {n:3d}  {sec}")

print("\nSample chunks:")
for c in chunks[:: max(1, len(chunks) // 4)][:4]:
    m = c["metadata"]
    print(f"\n--- {m['section']} | p.{m['page_start']}-{m['page_end']} | {m['element_type']}")
    print(c["text"][:400], "...")