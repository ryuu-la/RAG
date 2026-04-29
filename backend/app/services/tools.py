from __future__ import annotations

import json
import uuid
from io import StringIO
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from app.config import settings
from app.services.retrieval import hybrid_search
from app.store import store


def _export_dir() -> Path:
    p = settings.export_path
    p.mkdir(parents=True, exist_ok=True)
    return p


@tool
def search_documents(query: str) -> str:
    """Search indexed user-uploaded RAG documents. ONLY use this when the user EXPLICITLY asks to search their uploaded files or documents."""
    chunks = hybrid_search(query, top_k=settings.top_k)
    if not chunks:
        return "No relevant documents found in the index."
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata") or {}
        src = meta.get("source", "unknown")
        pg = meta.get("page", "?")
        txt = (c.get("text") or "")[:600]
        parts.append(f"[{i}] {src} (page {pg}):\n{txt}")
    return "\n\n".join(parts)


@tool
def lookup_document(document_name: str) -> str:
    """Look up metadata for an indexed document by name (or partial name)."""
    for doc in store.documents.values():
        if document_name.lower() in doc.get("filename", "").lower():
            return json.dumps(
                {
                    "filename": doc["filename"],
                    "page_count": doc.get("page_count", 0),
                    "chunk_count": doc.get("chunk_count", 0),
                    "indexed_chunks": doc.get("indexed_chunk_count", 0),
                    "estimated_tokens": doc.get("estimated_tokens", 0),
                },
                indent=2,
            )
    return f"No document matching '{document_name}' found."


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use for news, facts, current events. You can call this multiple times with different queries."""
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=10)
        if not results:
            return "No web results found. Try different keywords or answer from your knowledge."
        return "\n\n".join(
            f"**{r['title']}**\n{r['body']}\nURL: {r['href']}" for r in results
        )
    except Exception as e:
        return f"Search error: {e}. Try different keywords or answer from your knowledge."


@tool
def export_csv(title: str, csv_content: str) -> str:
    """Export data as a CSV file.

    FORMATTING GUIDE:
    - First row MUST be column headers.
    - Use commas to separate columns.
    - Wrap values containing commas in double quotes: "New York, USA"
    - One row per line.
    - You can include as many columns and rows as needed.

    Example csv_content:
    Name,Age,City,Score
    Alice,28,New York,95.5
    Bob,34,"San Francisco, CA",88.0

    Returns a download link."""
    try:
        import pandas as pd

        df = pd.read_csv(StringIO(csv_content))
        fname = f"{uuid.uuid4().hex[:8]}_{title.replace(' ', '_')}.csv"
        path = _export_dir() / fname
        df.to_csv(str(path), index=False)
        return f"CSV exported: [Download {fname}](/api/exports/{fname})"
    except Exception as e:
        return f"CSV export failed: {e}"


def _sanitize_latin1(text: str) -> str:
    out: list[str] = []
    for ch in text:
        try:
            ch.encode("latin-1")
            out.append(ch)
        except UnicodeEncodeError:
            if ch in ("\u2014", "\u2013"):
                out.append("-")
            elif ch in ("\u2018", "\u2019"):
                out.append("'")
            elif ch in ("\u201c", "\u201d"):
                out.append('"')
            elif ch == "\u2026":
                out.append("...")
            elif ch == "\u2022":
                out.append("*")
            else:
                out.append(" ")
    return "".join(out)


def _render_pdf_content(pdf, content: str) -> None:
    """Parse simple markup in content and render rich PDF formatting.

    Supported markup:
      # Heading 1        -> bold 18pt, top margin
      ## Heading 2       -> bold 14pt, top margin
      ### Heading 3      -> bold 12pt, top margin
      **bold text**      -> inline bold
      ---                -> horizontal line
      - bullet item      -> bulleted list
      * bullet item      -> bulleted list
      1. numbered item   -> numbered list
      | H1 | H2 |        -> table row (pipe-separated)
      [blank line]       -> paragraph spacing
      normal text        -> regular 11pt
    """
    import re

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = _sanitize_latin1(raw)
        stripped = line.strip()

        if not stripped:
            pdf.ln(4)
            i += 1
            continue

        if stripped == "---" or stripped == "___" or stripped == "***":
            y = pdf.get_y()
            pdf.set_draw_color(180, 180, 180)
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(6)
            i += 1
            continue

        if stripped.startswith("# ") and not stripped.startswith("## "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 8, stripped[2:])
            pdf.ln(3)
            i += 1
            continue

        if stripped.startswith("## ") and not stripped.startswith("### "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 7, stripped[3:])
            pdf.ln(2)
            i += 1
            continue

        if stripped.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 6, stripped[4:])
            pdf.ln(2)
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            table_rows: list[list[str]] = []
            while i < len(lines):
                r = _sanitize_latin1(lines[i]).strip()
                if not r.startswith("|"):
                    break
                if set(r.replace("|", "").replace("-", "").strip()) <= {" ", ""}:
                    i += 1
                    continue
                cells = [c.strip() for c in r.strip("|").split("|")]
                table_rows.append(cells)
                i += 1

            if table_rows:
                n_cols = max(len(r) for r in table_rows)
                col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / max(n_cols, 1)

                for ri, row in enumerate(table_rows):
                    while len(row) < n_cols:
                        row.append("")
                    if ri == 0:
                        pdf.set_font("Helvetica", "B", 10)
                        pdf.set_fill_color(230, 230, 230)
                        for cell in row:
                            pdf.cell(col_w, 7, cell[:30], border=1, fill=True)
                        pdf.ln()
                    else:
                        pdf.set_font("Helvetica", "", 10)
                        for cell in row:
                            pdf.cell(col_w, 7, cell[:30], border=1)
                        pdf.ln()
                pdf.ln(3)
            continue

        if re.match(r"^[-*]\s", stripped):
            text = stripped[2:]
            clean = _strip_bold(text)
            pdf.set_font("Helvetica", "", 11)
            indent = pdf.l_margin + 6
            pdf.set_x(pdf.l_margin)
            pdf.cell(6, 6, chr(149))
            _multi_cell_rich(pdf, pdf.w - indent - pdf.r_margin, 6, text, 11)
            pdf.ln(1)
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s", stripped)
        if m:
            text = stripped[len(m.group(0)):]
            pdf.set_font("Helvetica", "", 11)
            indent = pdf.l_margin + 8
            pdf.set_x(pdf.l_margin)
            pdf.cell(8, 6, m.group(1) + ".")
            _multi_cell_rich(pdf, pdf.w - indent - pdf.r_margin, 6, text, 11)
            pdf.ln(1)
            i += 1
            continue

        pdf.set_font("Helvetica", "", 11)
        _multi_cell_rich(pdf, pdf.w - pdf.l_margin - pdf.r_margin, 6, stripped, 11)
        pdf.ln(2)
        i += 1


def _strip_bold(text: str) -> str:
    import re
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", text)


def _multi_cell_rich(pdf, w: float, h: float, text: str, size: int) -> None:
    """multi_cell that handles **bold** inline markup without overlapping."""
    import re
    clean = _strip_bold(text)
    parts = re.split(r"(\*\*[^*]+\*\*)", text)

    has_bold = any(p.startswith("**") and p.endswith("**") for p in parts)
    if not has_bold:
        pdf.set_font("Helvetica", "", size)
        pdf.multi_cell(w, h, text)
        return

    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            pdf.set_font("Helvetica", "B", size)
            pdf.write(h, part[2:-2])
            pdf.set_font("Helvetica", "", size)
        else:
            pdf.write(h, part)
    pdf.ln(h)


@tool
def export_pdf(title: str, content: str) -> str:
    """Export a richly formatted PDF document. Returns a download link.

    FORMATTING GUIDE - use these in the `content` parameter:

    HEADINGS:
      # Main Title           -> large bold heading (18pt)
      ## Section Heading      -> medium bold heading (14pt)
      ### Sub-heading         -> small bold heading (12pt)

    TEXT STYLING:
      **bold text**           -> renders as bold inline
      Normal text             -> regular 11pt paragraph

    STRUCTURE:
      ---                     -> horizontal divider line
      [blank line]            -> paragraph spacing

    LISTS:
      - Bullet item           -> bulleted list
      * Another bullet        -> bulleted list
      1. First item           -> numbered list
      2. Second item          -> numbered list

    TABLES (pipe-separated):
      | Column 1 | Column 2 | Column 3 |
      |----------|----------|----------|
      | Data 1   | Data 2   | Data 3   |
      (Header row gets bold + gray background)

    EXAMPLE content:
      # Monthly AI Report
      ## Executive Summary
      This report covers the **top 3 breakthroughs** in AI for March 2026.
      ---
      ## 1. Model Architecture
      **Key finding:** Transformer variants continue to dominate.
      - Efficiency improved by 40%
      - Cost reduced significantly
      ---
      ## Comparison Table
      | Model | Parameters | Score |
      |-------|-----------|-------|
      | GPT-5 | 1.8T | 94.2 |
      | Gemini | 1.5T | 93.8 |
      ---
      ## Conclusion
      1. First takeaway point
      2. Second takeaway point

    TIPS:
      - Use # headings to create clear document structure
      - Use --- between sections for visual separation
      - Use tables for comparisons and data
      - Use **bold** for emphasis on key terms
      - Keep all text ASCII-safe (no emojis or special Unicode)
      - Write COMPLETE content, not summaries"""
    try:
        from fpdf import FPDF

        fname = f"{uuid.uuid4().hex[:8]}_{title.replace(' ', '_')}.pdf"
        path = _export_dir() / fname
        safe_title = _sanitize_latin1(title)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()
        pdf.set_margins(20, 20, 20)

        pdf.set_font("Helvetica", "B", 22)
        pdf.cell(0, 12, safe_title, new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(3)
        pdf.set_draw_color(16, 163, 127)
        pdf.set_line_width(0.8)
        y = pdf.get_y()
        mid = pdf.w / 2
        pdf.line(mid - 30, y, mid + 30, y)
        pdf.ln(8)

        _render_pdf_content(pdf, content)

        pdf.output(str(path))
        return f"PDF exported: [Download {fname}](/api/exports/{fname})"
    except Exception as e:
        return f"PDF export failed: {e}"


@tool
def read_url(url: str) -> str:
    """Read the content of a specific webpage or URL. Use this when the user gives you a direct link or if you need to read a full article after searching."""
    try:
        import urllib.request
        from bs4 import BeautifulSoup
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
            soup = BeautifulSoup(html, "html.parser")
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ')
            # Collapse whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long (e.g., 20000 chars)
            return f"--- Content from {url} ---\n\n" + text[:20000]
    except Exception as e:
        return f"Failed to read URL {url}: {e}"

def get_tools(rag_mode: bool) -> list:
    if rag_mode:
        # Re-order so web_search is first and clearly prioritized
        return [web_search, read_url, search_documents, lookup_document, export_csv, export_pdf]
    return [web_search, read_url, export_csv, export_pdf]
