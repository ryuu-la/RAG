from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, AsyncGenerator

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.config import settings
from app.services.tools import get_tools

MAX_ITERATIONS = 10

AVAILABLE_MODELS = [
    {"id": "gemma-4-31b-it", "label": "Gemma 4 31B", "provider": "google"},
    {"id": "gemini-3.1-flash-lite-preview", "label": "Gemini 3.1 Flash Lite", "provider": "google", "grounding": True},
]


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

## DYNAMIC SEARCH POLICY (CRITICAL - FOLLOW EXACTLY)

You MUST decide which search tool to use based on the user's query:

### USE `search_documents` (Index/RAG Search) ONLY WHEN:
The user's query contains ANY of these explicit keywords or phrases:
- "index", "from index", "indexed", "from my index"
- "document", "documents", "my documents", "uploaded"
- "my files", "my PDF", "the PDF", "the report"
- "RAG", "search my files", "in my documents"
- Any reference to a specific uploaded filename

Examples that trigger index search:
- "search about langchain from index" -> use search_documents
- "what does my document say about X" -> use search_documents
- "find in my uploaded files" -> use search_documents

### USE `web_search` (DuckDuckGo) FOR EVERYTHING ELSE (DEFAULT):
If the user does NOT mention index, documents, uploaded files, or RAG,
ALWAYS default to `web_search` for live internet results.

Examples that trigger web search:
- "what is langchain" -> use web_search
- "latest AI news" -> use web_search
- "explain transformers" -> use web_search
- "search about langchain" (no mention of index) -> use web_search

### FALLBACK:
If `search_documents` returns nothing on a document-specific query,
fallback to `web_search` if the context allows.

""" + TOOL_SKILLS + """

## AGENTIC BEHAVIOR

1. Decide between `web_search` (default) and `search_documents` (only when explicitly requested) based on the Dynamic Search Policy above.
2. THINK step by step. Plan what tools you need.
3. GATHER enough information. Do MULTIPLE searches with different queries if needed.
4. SYNTHESIZE a thorough, well-structured answer with citations.
5. When exporting files, write COMPLETE, professionally formatted content.
6. If a tool returns an error or empty results, try a DIFFERENT approach/query.

## CITATION & SOURCE RULES
- When you used `web_search` or `read_url`: cite the actual WEBSITE URLs you visited. These are the real sources.
- When you used `search_documents`: cite as [filename, page X] from the indexed documents.
- NEVER cite PDF filenames when you only did a web search. Only cite the web URLs.
- NEVER cite web URLs when you only searched the index. Only cite document sources.

## RESPONSE FORMAT

- Use **bold** for emphasis.
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
    """Return a LangChain chat model for the given Google model_id."""
    if not settings.google_api_key.strip():
        raise ValueError("GOOGLE_API_KEY not configured in .env")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "langchain_google_genai is not installed. Install it to use Google models."
        ) from exc
    return ChatGoogleGenerativeAI(
        model=model_id,
        google_api_key=settings.google_api_key,
        temperature=0.15,
        max_output_tokens=8192,
    )


async def run_agent_stream(
    question: str,
    rag_mode: bool = True,
    model_name: str | None = None,
    extra_context: str = "",
    history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    t0 = time.perf_counter()
    model_id = model_name or settings.default_model

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

    messages: list = [SystemMessage(content=sys_text)]

    # Inject conversation history for multi-turn context
    if history:
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=question))
    web_citations: list[dict] = []
    doc_citations: list[dict] = []
    # Use a set to prevent duplicate citations
    seen_sources = set()
    export_links: list[dict] = []
    # Track which search tools were actually used
    used_web_search = False
    used_search_documents = False

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
                used_search_documents = True
                for m in re.finditer(
                    r"\[(\d+)\]\s+(\S+)\s+\(page\s+(\d+)\)", str(result)
                ):
                    src = m.group(2)
                    pg = m.group(3)
                    if (src, pg) not in seen_sources:
                        seen_sources.add((src, pg))
                        doc_citations.append({
                            "source": src,
                            "page": int(pg),
                            "chunk_id": "",
                        })
                    
            if name == "web_search":
                used_web_search = True
                for m in re.finditer(r"URL: (https?://[^\s\n]+)", str(result)):
                    url = m.group(1)
                    if url not in seen_sources:
                        seen_sources.add(url)
                        web_citations.append({
                            "source": url,
                            "page": None,
                            "chunk_id": "",
                        })

            if name == "read_url":
                used_web_search = True
                try:
                    url = args.get("url", "")
                    if url and isinstance(url, str) and url.startswith("http"):
                        if url not in seen_sources:
                            seen_sources.add(url)
                            web_citations.append({
                                "source": url,
                                "page": None,
                                "chunk_id": "",
                            })
                except Exception:
                    pass

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

    # Build final citations based on which tools were actually used
    # If only web_search was used -> only show web URLs (not PDF names)
    # If only search_documents was used -> only show document sources
    # If both were used -> show all
    final_citations: list[dict] = []
    if used_web_search:
        final_citations.extend(web_citations)
    if used_search_documents:
        final_citations.extend(doc_citations)

    yield _sse("done", {
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "citations": final_citations,
    })
