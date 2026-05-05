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
    {"id": "openai/gpt-oss-120b:free", "label": "GPT OSS 120B (Free)", "provider": "openrouter"},
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

### SKILL: web_search(query, timelimit="d"|"w"|"m"|"y"|None)
Purpose: Live internet search via DuckDuckGo.
When to use: Current events, news, facts NOT in documents.
Strategy:
- Use `timelimit="d"` or `"w"` if the user explicitly asks for the "latest", "recent", "today", or live scores/news.
- Keep queries SHORT: 3-6 words work best.
- For most simple questions or live data, search ONCE and answer immediately based on the snippet. Do not search multiple sites unnecessarily.
- Only call multiple times if the first search returns absolutely no useful information or if the user asks for a deep dive.
- CRITICAL FOR LIVE DATA: Search snippets often omit specific live numbers (like sports scores, stock prices). If the snippet doesn't contain the exact answer, use `read_url` on the most promising URL from the search results to extract the live data.

### SKILL: read_url(url)
Purpose: Read the full text content of a specific webpage.
When to use: 
- When the user provides a direct link to read.
- After using `web_search`, if the search snippet doesn't contain the specific data you need, use this to read the full article from the search results.

### SKILL: image_search(query, max_results=4)
Purpose: Fetch images from the internet to display in the chat UI.
When to use: 
- When the user explicitly asks for pictures, images, or screenshots.
- When visually describing something (e.g., plants, UI designs, places) where an image adds massive value.
Strategy:
- Keep queries simple (e.g., "red hibiscus flower", "modern UI dashboard").
- The tool returns markdown images. YOU MUST include these markdown images in your final response to the user so they can see them!

### SKILL: get_live_sports_scores()
Purpose: Get real-time, second-by-second live cricket/IPL scores.
When to use: ALWAYS use this FIRST when the user asks for the score of a currently ongoing IPL or cricket match. Do not use web_search for cricket scores.

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

### SKILL: generate_mindmap(title, markdown_content)
Purpose: Generate an interactive mind map visualization displayed in the chat UI.
When to use: When the user asks for a "mind map", "concept map", "knowledge map", or "visual overview" of their documents.
Strategy:
- FIRST, call `search_documents` MULTIPLE TIMES (3-5 calls) with different queries to gather comprehensive content covering ALL major topics in the document(s).
- Synthesize ALL gathered information into a DEEP hierarchical markdown structure.
- Use `#` for the single root/central topic.
- Use `##` for major branches (like chapters or main sections).
- Use `###` for sub-branches (subsections or key concepts).
- Use `- ` bullets for leaf nodes (specific facts, terms, definitions).
- Use `  - ` indented bullets for even deeper details.
- Go AT LEAST 3-4 levels deep to cover the FULL document content.
- Keep each node label SHORT: 3-8 words max for readability.
- Include specific facts, numbers, terms, and definitions as leaf nodes.
- Do NOT use bold/italic/links/code blocks in the markdown — plain text only.
- After calling generate_mindmap, write a brief text summary of the mind map for the user.
"""

RAG_SYSTEM_TEMPLATE = """\
You are the MAIN LLM of the RAG Studio AI system. You are the ONLY entity with the power to reason and think.

## ARCHITECTURE & SCRATCH PAD PROTOCOL
1. **User Query:** The user sends you a query.
2. **Breakdown & Assignment:** You MUST break down the query in your `<think>` block and assign roles to your completely mindless sub-agents (your tools). 
3. **Agent Execution:** The agents (tools like web_search, search_documents) have no ability to think. They just execute your precise instructions and return raw data.
4. **The Scratch Pad:** The message history / tool returns act as your INTERNAL SCRATCH PAD. It is a meeting place where agents paste their fetched information. This scratch pad is NOT displayed to the user.
5. **Final Synthesis:** Once all agents have placed their data in the scratch pad, YOU evaluate everything and generate the final polished content for the UI.

FORMAT YOUR REASONING EXACTLY LIKE THIS:
<think>
Thinking Process:
1. Analyze Request: [What is the user asking?]
2. Agent Assignment: [Which agents (tools) do I invoke concurrently? e.g. Document Agent for index, Web Agent for live data]
3. Expected Scratch Pad Data: [What do I expect them to bring back?]
4. Synthesis Strategy: [How will I combine the scratch pad data once they return it?]
</think>
[Once the scratch pad is full of agent data, write your final response to the user here]

## AGENT (TOOL) DELEGATION POLICY

You orchestrate these mindless agents by calling their respective tools. You should call multiple tools concurrently for parallel execution.

### Document Agent (`search_documents`)
- **When to assign:** User mentions "index", "document", "uploaded", "my files", "RAG", or a specific filename. Provide exact citations [1], [2].

### Web & Vision Agents (`web_search`, `image_search`)
- **When to assign:** User asks about live data, current events, or visuals not in their personal docs. 

""" + TOOL_SKILLS + """

## FINAL CONTENT RULES (AFTER AGENTS RETURN DATA)
- Combine all the information from the internal scratch pad smoothly.
- Embed any images fetched using `image_search` in markdown format directly in the text!
- NEVER output raw data block tables. Use `export_csv` for data and `export_pdf` for reports.
- `web_search` / `read_url`: Cite actual website URLs.
- `search_documents`: Use simple numeric citations `[1]`, `[2]`. NEVER cite raw internal filenames.
"""

CHAT_SYSTEM_TEMPLATE = """\
You are the MAIN LLM of the RAG Studio AI system. You are the ONLY entity with the power to reason and think.

## ARCHITECTURE & SCRATCH PAD PROTOCOL
1. **User Query:** The user sends you a query.
2. **Breakdown & Assignment:** You MUST break down the query in your `<think>` block and assign roles to your completely mindless sub-agents (your tools). 
3. **Agent Execution:** The agents (tools like web_search, search_documents) have no ability to think. They just execute your precise instructions and return raw data.
4. **The Scratch Pad:** The message history / tool returns act as your INTERNAL SCRATCH PAD. It is a meeting place where agents paste their fetched information.
5. **Final Synthesis:** Once all agents have placed their data in the scratch pad, YOU evaluate everything and generate the final polished content for the UI.

FORMAT YOUR REASONING EXACTLY LIKE THIS:
<think>
Thinking Process:
1. Analyze Request: [What the user is asking, identifying any attached multimodal media]
2. Agent Assignment: [Which agents (tools) do I invoke concurrently? e.g. Web Agent, Vision Agent]
3. Expected Scratch Pad Data: [What do I expect the agents to return?]
4. Synthesis Strategy: [How to construct the final response]
</think>
[Once the scratch pad is full of agent data, write your final response here]

""" + TOOL_SKILLS + """

## FINAL CONTENT RULES
- Deploy multiple agents via concurrent tool calls for maximum efficiency.
- Handle multimodal context confidently.
- Be helpful, deeply analytical, and highly structured in your final output, using the data from the scratch pad.
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
    """Return a LangChain chat model for the given model_id."""
    if model_id.startswith("openai/"):
        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file to use OpenRouter models."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ValueError(
                "Missing dependency: langchain-openai. "
                "Run: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(
            model=model_id,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.15,
            max_tokens=8192,
        )

    if not settings.has_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Add it to your .env file (in backend/ or project root). "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ValueError(
            "Missing dependency: langchain-google-genai. "
            "Run: pip install langchain-google-genai"
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
    media_data: list[str] | None = None,
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

    content_parts = [{"type": "text", "text": question}]
    if media_data:
        for md in media_data:
            content_parts.append({"type": "image_url", "image_url": {"url": md}})

    messages.append(HumanMessage(content=content_parts))
    web_citations: list[dict] = []
    doc_citations: list[dict] = []
    # Use a set to prevent duplicate citations
    seen_sources = set()
    export_links: list[dict] = []
    mindmap_events: list[dict] = []
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


        import asyncio

        # Execute parallel tools
        async def execute_tool(tc):
            name = tc["name"]
            args = tc["args"]
            tid = tc["id"]
            
            yield_events = []
            yield_events.append(_sse("step", {
                "type": "tool_start",
                "tool": name,
                "input": json.dumps(args, ensure_ascii=False)[:300],
            }))

            try:
                if name in tools_map:
                    # Run sync tool in thread for parallel execution if necessary
                    # LangChain's ainvoke works directly as well
                    result = await tools_map[name].ainvoke(args)
                else:
                    result = f"Unknown tool: {name}"
            except Exception as exc:
                result = f"Tool error: {exc}"

            yield_events.append(_sse("step", {
                "type": "tool_end",
                "tool": name,
                "output": str(result)[:500],
            }))
            
            return {"name": name, "result": result, "tid": tid, "args": args, "events": yield_events}

        tool_tasks = [execute_tool(tc) for tc in resp.tool_calls]
        tool_results = await asyncio.gather(*tool_tasks)

        for tr in tool_results:
            name = tr["name"]
            result = tr["result"]
            tid = tr["tid"]
            args = tr["args"]

            # Stream out the events generated during the async execution
            for evt in tr["events"]:
                yield evt

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

            if name == "generate_mindmap":
                try:
                    mindmap_data = json.loads(str(result))
                    mindmap_events.append(mindmap_data)
                    yield _sse("mindmap", mindmap_data)
                except Exception:
                    pass

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
