from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, AsyncGenerator

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.services.tools import get_tools

MAX_ITERATIONS = 10

GOOGLE_MODELS = {"gemma-4-31b-it"}


def _date_preamble() -> str:
    now = datetime.now()
    return (
        f"Today is {now.strftime('%A, %B %d, %Y')}. Current year: {now.year}.\n"
        "Always use the current year in web searches unless the user says otherwise.\n\n"
    )


TOOL_SKILLS = """\
## TOOL SKILLS REFERENCE

### SKILL: search_documents(query)
Purpose: Search indexed PDFs using hybrid semantic + BM25 retrieval.
When to use: ALWAYS first when user asks about uploaded document content.
Strategy:
- Call MULTIPLE TIMES with different queries to gather comprehensive info.
- Use specific keywords, then broaden if results are thin.
- Cite results as [source_name, page X].

### SKILL: web_search(query)
Purpose: Live internet search via DuckDuckGo.
When to use: Current events, news, facts NOT in documents.
Strategy:
- Keep queries SHORT: 3-6 words work best.
- Call MULTIPLE TIMES with varied wording to get broad coverage.
- Try different angles: "AI news 2026", "latest AI models", "AI breakthroughs March".
- If one query returns nothing, rephrase and try again.

### SKILL: lookup_document(document_name)
Purpose: Get metadata about an indexed document.
When to use: User asks about file stats, page counts, what's indexed.

### SKILL: export_pdf(title, content)
Purpose: Generate a professionally formatted PDF document.
When to use: Reports, summaries, research papers, any document export.

**PDF FORMATTING SYNTAX** (use these in the `content` parameter):

HEADINGS (create document structure):
  # Main Title              -> large bold 18pt heading
  ## Section Heading         -> medium bold 14pt heading
  ### Sub-heading            -> small bold 12pt heading

TEXT:
  Normal text                -> regular 11pt paragraph
  **bold words**             -> inline bold emphasis

DIVIDERS:
  ---                        -> horizontal line (use between sections)

LISTS:
  - Bullet point             -> bulleted item
  * Another bullet           -> bulleted item
  1. First item              -> numbered list
  2. Second item             -> numbered list

TABLES (pipe-separated):
  | Column 1 | Column 2 | Column 3 |
  |----------|----------|----------|
  | Data A   | Data B   | Data C   |
  | Data D   | Data E   | Data F   |
  (First row = bold header with gray background, remaining = data rows)

SPACING:
  [blank line]               -> adds paragraph spacing

**EXAMPLE PDF CONTENT:**
```
# Quarterly AI Research Report

## Executive Summary

This report covers the **top breakthroughs** in AI for Q1 2026.

---

## 1. Foundation Models

**Key finding:** New architectures have improved efficiency by **40%**.

- Mixture-of-Experts scaling
- Improved training stability
- Lower inference costs

---

## 2. Comparison Table

| Model | Parameters | Benchmark | Release |
|-------|-----------|-----------|---------|
| GPT-5 | 1.8T | 94.2 | Jan 2026 |
| Gemini 2 | 1.5T | 93.8 | Feb 2026 |
| Claude 4 | 1.2T | 93.5 | Mar 2026 |

---

## Conclusion

1. Efficiency gains are the dominant trend
2. Open-source models are closing the gap
3. Multimodal capabilities are now standard
```

**PDF TIPS:**
- Use # headings to create clear sections.
- Use --- between major sections for visual separation.
- Use tables for any comparison or structured data.
- Use **bold** to highlight key terms, findings, names.
- Write COMPLETE content - never say "insert content here".
- Keep text ASCII-safe. No emojis or special Unicode characters.
- For long reports, use ## for each section and ### for sub-points.

### SKILL: export_csv(title, csv_content)
Purpose: Generate a downloadable CSV spreadsheet.
When to use: Data tables, comparisons, structured datasets.

**CSV FORMAT:**
- First row MUST be column headers.
- Comma-separated values.
- Wrap values with commas in quotes: "San Francisco, CA"
- One row per line.

**EXAMPLE csv_content:**
```
Name,Category,Score,Year,Notes
GPT-5,Language Model,94.2,2026,Flagship model
Gemini 2,Multimodal,93.8,2026,"Vision, audio, text"
Claude 4,Language Model,93.5,2026,Strong reasoning
Llama 4,Open Source,91.0,2026,"Free, open weights"
```

**CSV TIPS:**
- Include meaningful column headers.
- Use consistent formatting in each column.
- Quote any values containing commas.
- Include enough rows to be useful.
"""

RAG_SYSTEM_TEMPLATE = """\
You are RAG Studio, an advanced agentic AI assistant with tool access.

## CRITICAL RULE — DOCUMENT-FIRST POLICY
You MUST ALWAYS call search_documents FIRST before doing anything else.
NEVER call web_search before you have searched the indexed documents.
The user uploaded documents specifically for you to answer from — use them.
Only use web_search if search_documents returned nothing relevant, or if the user explicitly asks for web/internet information.

""" + TOOL_SKILLS + """

## AGENTIC BEHAVIOR

1. ALWAYS call search_documents FIRST. This is your primary knowledge source.
2. THINK step by step. Plan what tools you need after searching documents.
3. Only use web_search if documents don't have the answer OR user asks for current/live info.
4. GATHER enough information. Do MULTIPLE document searches with different queries if needed.
5. SYNTHESIZE a thorough, well-structured answer with citations from documents.
6. When exporting files, write COMPLETE, professionally formatted content.
7. If a tool returns an error or empty results, try a DIFFERENT approach/query.

## RESPONSE FORMAT

- Use **bold** for emphasis.
- Cite document sources as [filename, page X].
- Include download links naturally when you create export files.
- NEVER paste raw CSV data or code blocks in your response. ALWAYS use the export_csv tool to create a downloadable file.
- NEVER paste raw table data as text. Use export_csv for tabular data and export_pdf for reports.
- Be thorough. Quality and completeness over brevity.
"""

CHAT_SYSTEM_TEMPLATE = """\
You are RAG Studio in chat mode - a capable agentic AI assistant.

""" + TOOL_SKILLS + """

## AGENTIC BEHAVIOR

1. THINK before acting. Plan your approach.
2. GATHER sufficient information - search MULTIPLE TIMES with different queries if needed.
3. SYNTHESIZE a thorough, well-structured answer.
4. When exporting files, write COMPLETE, professionally formatted content.
5. If a search returns nothing, rephrase and try again before giving up.
6. Be helpful, accurate, and proactive. Go above and beyond.

## RESPONSE FORMAT

- Use **bold** for emphasis.
- Include download links naturally when you create export files.
- NEVER paste raw CSV data or code blocks in your response. ALWAYS use the export_csv tool to create a downloadable file.
- NEVER paste raw table data as text. Use export_csv for tabular data and export_pdf for reports.
- Be thorough. Quality and completeness over brevity.
"""


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _text(content: Any) -> str:
    """Extract plain text from a chunk's .content (may be str, list, or dict)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    if isinstance(content, dict) and "text" in content:
        return content["text"]
    return ""


def _build_llm(model_id: str):
    """Return the right LangChain chat model based on model_id."""
    if model_id in GOOGLE_MODELS:
        if not settings.google_api_key.strip():
            raise ValueError("GOOGLE_API_KEY not configured in .env")
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=settings.google_api_key,
            temperature=0.15,
            max_output_tokens=8192,
        )
    if not settings.groq_api_key.strip():
        raise ValueError("GROQ_API_KEY not configured in .env")
    return ChatGroq(
        model=model_id,
        api_key=settings.groq_api_key,
        temperature=0.15,
        max_tokens=8192,
    )


async def run_agent_stream(
    question: str,
    rag_mode: bool = True,
    model_name: str | None = None,
    extra_context: str = "",
) -> AsyncGenerator[str, None]:
    t0 = time.perf_counter()
    model_id = model_name or settings.groq_model

    try:
        llm = _build_llm(model_id)
    except ValueError as exc:
        yield _sse("error", {"message": str(exc)})
        return

    tools = get_tools(rag_mode)
    tools_map = {t.name: t for t in tools}

    sys_text = _date_preamble() + (RAG_SYSTEM_TEMPLATE if rag_mode else CHAT_SYSTEM_TEMPLATE)
    if extra_context:
        sys_text += f"\n\nDirect-upload file context:\n{extra_context}"

    messages: list = [
        SystemMessage(content=sys_text),
        HumanMessage(content=question),
    ]
    citations: list[dict] = []
    export_links: list[dict] = []

    llm_with_tools = None
    if tools:
        try:
            llm_with_tools = llm.bind_tools(tools)
        except Exception:
            pass

    if llm_with_tools is None:
        yield _sse("step", {"type": "thinking", "content": "Generating response..."})
        async for chunk in llm.astream(messages):
            t = _text(chunk.content)
            if t:
                yield _sse("token", {"content": t})
        yield _sse("done", {
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "citations": [],
        })
        return

    agent_failed = False
    for _ in range(MAX_ITERATIONS):
        try:
            resp = await llm_with_tools.ainvoke(messages)
        except Exception as exc:
            yield _sse("step", {"type": "thinking", "content": f"LLM error: {str(exc)[:200]}"})
            agent_failed = True
            break

        if not resp.tool_calls:
            break

        thinking = _text(resp.content)
        if thinking:
            yield _sse("step", {"type": "thinking", "content": thinking})

        messages.append(resp)

        for tc in resp.tool_calls:
            name = tc["name"]
            args = tc["args"]
            tid = tc["id"]

            yield _sse("step", {
                "type": "tool_start",
                "tool": name,
                "input": json.dumps(args, ensure_ascii=False)[:300],
            })

            try:
                if name in tools_map:
                    result = tools_map[name].invoke(args)
                else:
                    result = f"Unknown tool: {name}"
            except Exception as exc:
                result = f"Tool error: {exc}"

            if name == "search_documents":
                for m in re.finditer(
                    r"\[(\d+)\]\s+(\S+)\s+\(page\s+(\d+)\)", str(result)
                ):
                    citations.append({
                        "source": m.group(2),
                        "page": int(m.group(3)),
                        "chunk_id": "",
                    })

            if name in ("export_pdf", "export_csv"):
                for m in re.finditer(r"\[([^\]]+)\]\((/api/exports/[^)]+)\)", str(result)):
                    export_links.append({"label": m.group(1), "href": m.group(2)})

            yield _sse("step", {
                "type": "tool_end",
                "tool": name,
                "output": str(result)[:500],
            })

            messages.append(ToolMessage(content=str(result), tool_call_id=tid))

    if not agent_failed:
        try:
            async for chunk in llm.astream(messages):
                t = _text(chunk.content)
                if t:
                    yield _sse("token", {"content": t})
        except Exception as exc:
            err_short = str(exc)[:150]
            if export_links:
                yield _sse("token", {"content": "Files generated successfully. See downloads below."})
            else:
                yield _sse("token", {"content": f"Error generating response: {err_short}"})
    elif export_links:
        yield _sse("token", {"content": "Files generated successfully. See downloads below."})

    if export_links:
        yield _sse("exports", export_links)

    yield _sse("done", {
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "citations": citations,
    })
