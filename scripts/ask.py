import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import sys
from rag.generation import answer

q = " ".join(sys.argv[1:]) or "What is the difference between RAG-Sequence and RAG-Token?"
print(f"\nQ: {q}\n")
print(answer(q).format())