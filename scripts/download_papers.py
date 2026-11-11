import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json
import time
from pathlib import Path

import arxiv
import httpx

IDS = [
    "1706.03762", "1810.04805", "1908.10084", "2004.04906", "2004.12832",
    "2005.11401", "2210.03629", "2212.10496", "2307.03172", "2309.15217",
    "2310.11511", "2312.10997", "2401.15884", "2404.16130", "2005.14165",
]
from rag.config import settings
OUT = settings.raw_dir; OUT.mkdir(parents=True, exist_ok=True)

client = arxiv.Client(delay_seconds=3)
catalog = {}

with httpx.Client(
    follow_redirects=True,
    timeout=60,
    headers={"User-Agent": "rag-chatbot/0.1 (research project)"},
) as http:
    for paper in client.results(arxiv.Search(id_list=IDS, max_results=len(IDS))):
        pid = paper.get_short_id().split("v")[0]
        fname = f"{pid}.pdf"
        catalog[pid] = {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "published": paper.published.date().isoformat(),
            "url": paper.entry_id,
        }
        out = OUT / fname
        if out.exists():
            print("skip ", fname); continue

        r = http.get(paper.pdf_url)
        r.raise_for_status()
        out.write_bytes(r.content)
        print("saved", fname, "-", paper.title[:60])
        time.sleep(3)                     # be polite to arxiv.org

(OUT / "catalog.json").write_text(json.dumps(catalog, indent=2))
print(f"\n{len(catalog)} papers in catalog")