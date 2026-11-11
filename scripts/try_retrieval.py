import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import sys
from rag.retrieval import retrieve

question = " ".join(sys.argv[1:]) or "What retriever does the original RAG model use?"
results = retrieve(question, top_k=5)

print(f"\nQ: {question}\n")
for i, r in enumerate(results, 1):
    print(f"[{i}] score={r.score:.3f}  {r.citation}")
    print("    " + r.text[:300].replace("\n", " ") + "...\n")