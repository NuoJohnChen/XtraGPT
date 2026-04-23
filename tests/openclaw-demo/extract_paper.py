"""Extract Abstract + Introduction + title from the EMNLP PDF."""
import json
import pathlib
import re

import pypdf

HERE = pathlib.Path(__file__).parent
PDF = HERE / "input.pdf"
OUT = HERE / "paper_sections.json"

reader = pypdf.PdfReader(str(PDF))
pages = [p.extract_text() or "" for p in reader.pages]
full = "\n\n".join(pages)
full_clean = re.sub(r"-\n(?=\w)", "", full)
full_clean = re.sub(r"(?<!\n)\n(?!\n)", " ", full_clean)
full_clean = re.sub(r"[ \t]+", " ", full_clean).strip()

def between(start: str, end: str | None, text: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    if end is None:
        return text[i:].strip()
    j = text.find(end, i)
    return (text[i:j] if j > 0 else text[i:]).strip()

abstract = between("Abstract", "1 Introduction", full_clean)

intro = between("1 Introduction", "2 ", full_clean)
# Trim intro to first ~800 chars of coherent sentences.
sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", intro)
buf, total = [], 0
for s in sentences:
    buf.append(s)
    total += len(s) + 1
    if total > 700:
        break
intro_first_block = " ".join(buf).strip()

raw_lines = [ln.strip() for ln in pages[0].splitlines() if ln.strip()]
# ACL proceedings: lines 0-1 are header, title is lines 2+3 (optional wrap), then authors.
title = raw_lines[2] if len(raw_lines) > 2 else ""
if len(raw_lines) > 3 and not re.match(r"[A-Z][a-z]+ (Chen|Gao|Jin|Wang|Hu|Yan)", raw_lines[3]):
    title = (title + " " + raw_lines[3]).strip()

intro_first_block = re.sub(r"\*[^.]*corresponding authors?\.?", "", intro_first_block).strip()
intro_first_block = re.sub(r"\s+\*$", "", intro_first_block).strip()

paper = {
    "source_pdf": str(PDF),
    "title": title[:300],
    "abstract": abstract,
    "intro_first_block": intro_first_block,
    "full_paper_content": full_clean[:20000],
}
OUT.write_text(json.dumps(paper, indent=2, ensure_ascii=False))

print("title:", paper["title"])
print("abstract chars:", len(paper["abstract"]))
print("intro_first_block chars:", len(paper["intro_first_block"]))
print()
print("--- abstract ---")
print(paper["abstract"])
print()
print("--- intro_first_block ---")
print(paper["intro_first_block"])
