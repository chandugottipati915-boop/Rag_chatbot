import hashlib
import json
import re
import unicodedata
from pathlib import Path

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# "1 Introduction", "3.2 Retrieval Models", "Abstract"
# Title must start uppercase so table rows like "6 positional embedding ..." don't match.
SECTION_RE = re.compile(r"^((?i:abstract)|(\d+(\.\d+)*)\.?\s+[A-Z].{1,80})$")
# "References", "Bibliography"
REFS_RE = re.compile(r"^(references|bibliography)\s*$", re.IGNORECASE)
# "Appendix", "A Implementation Details", "B.1 Datasets"
APPENDIX_RE = re.compile(r"^(appendix.*|[A-H](\.\d+)*\.?\s+[A-Z].{1,80})$")
# arXiv's left-margin stamp: "arXiv:2005.11401v4 [cs.CL] 12 Apr 2021"
STAMP_RE = re.compile(r"^arXiv:\d{4}\.\d{4,5}")
# pdfminer reads the vertical stamp as letter-spaced, often reversed: "4 v 1 0 4 1 1 . 5 0 0 2 : v i X r a"
SPACED_RE = re.compile(r"^(\S )+\S$")
# A heading number that pdfminer split from its title: "1" + "Introduction", "3.3" + "Jeopardy ..."
HEADING_NUM_RE = re.compile(r"^([1-9]|1\d)(\.\d{1,2})*\.?$")
# Glyphs pdfminer couldn't map to Unicode: "(cid:80)"
CID_RE = re.compile(r"\(cid:\d+\)")

DROP_CATEGORIES = {"Header", "Footer", "PageNumber"}


def clean_text(text: str) -> str:
    """Expand ligatures (ﬁ → fi), drop unmapped glyphs, collapse runs of spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = CID_RE.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def is_stamp(text: str, page: int | None) -> bool:
    if STAMP_RE.match(text):
        return True
    # The arXiv stamp only appears on page 1, where pdfminer splits it into letter-spaced,
    # reversed fragments: "4 v 1 0 4 1 1 . 5 0 0 2 : v i X r a", "] L C . s c [", "r p A 2 1"
    return page == 1 and bool(SPACED_RE.match(text))


def section_major(heading: str) -> int | None:
    """'4.3 Jeopardy ...' -> 4; 'Abstract' -> None."""
    m = re.match(r"(\d+)", heading)
    return int(m.group(1)) if m else None


def load_catalog(raw_dir: Path) -> dict:
    path = raw_dir / "catalog.json"
    return json.loads(path.read_text()) if path.exists() else {}


def split_by_section(elements):
    """Group elements by section; drop the References section."""
    sections, current, in_refs = {}, "Front matter", False
    pending_num = None                 # bare heading number waiting for its title
    for el in elements:
        text = clean_text(el.text)
        if el.category in DROP_CATEGORIES or not text or is_stamp(text, el.metadata.page_number):
            continue

        if HEADING_NUM_RE.match(text):
            pending_num = text.rstrip(".")
            continue
        if pending_num is not None:
            # re-attach "3.3" to "Jeopardy Question Generation"; if the next element isn't a
            # heading, SECTION_RE won't match and the number simply stays in the text
            text, pending_num = f"{pending_num} {text}", None
        el.text = text
        is_heading = len(text) < 100

        if is_heading and REFS_RE.match(text):
            in_refs = True
            continue
        if in_refs:
            if is_heading and APPENDIX_RE.match(text):   # appendix after references
                in_refs, current = False, text
            else:
                continue                                  # skip citation entries
        elif is_heading and SECTION_RE.match(text):
            # Section numbers only stay put or step forward by one; anything else
            # ("10 RAG-Tok" from a chart axis) is figure/table debris, not a heading.
            new, cur = section_major(text), section_major(current)
            if new is None or (cur is None and new <= 1) or (cur is not None and new in (cur, cur + 1)):
                current = text

        sections.setdefault(current, []).append(el)
    return sections


def chunk_file(path: Path, catalog: dict, strategy: str = "fast") -> list[dict]:
    arxiv_id = path.stem                              # "2005.11401"
    meta = catalog.get(arxiv_id, {})
    doc_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    elements = partition_pdf(
        filename=str(path),
        strategy=strategy,
        infer_table_structure=(strategy == "hi_res"),
    )

    chunks = []
    for section, els in split_by_section(elements).items():
        for i, ch in enumerate(chunk_by_title(
            els,
            max_characters=2000,          # ≈ 450 tokens — papers are dense
            new_after_n_chars=1500,
            combine_text_under_n_chars=200,
            overlap=200,
            multipage_sections=True,
        )):
            text = ch.text.strip()
            if len(text) < 50:
                continue
            pages = sorted({
                e.metadata.page_number
                for e in (ch.metadata.orig_elements or [])
                if e.metadata.page_number
            }) or [ch.metadata.page_number]
            chunks.append({
                "id": f"{arxiv_id}-{doc_hash[:6]}-{len(chunks)}",
                "text": text,
                "metadata": {
                    "arxiv_id": arxiv_id,
                    "title": meta.get("title", arxiv_id),
                    "published": meta.get("published"),
                    "section": section,
                    "page_start": pages[0],
                    "page_end": pages[-1],
                    "element_type": type(ch).__name__,
                    "source_file": path.name,
                    "doc_hash": doc_hash,
                },
            })
    return chunks