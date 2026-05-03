from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from app.config import settings


def build_context(chunks: list[dict[str, Any]]) -> str:
    context_blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source", "unknown.pdf")
        page = meta.get("page")
        chunk_id = meta.get("chunk_id", chunk.get("chunk_id", "na"))
        header = f"[{idx}] source={source} page={page} chunk_id={chunk_id}"
        body = chunk.get("text", "")
        context_blocks.append(f"{header}\n{body}")
    return "\n\n".join(context_blocks)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def trim_to_token_budget(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    extra_context: str = "",
    model_override: str | None = None,
) -> tuple[str, int | None, int]:
    rag_context = build_context(chunks)
    context = rag_context if not extra_context else f"{rag_context}\n\n{extra_context}"
    context = trim_to_token_budget(context, settings.max_context_tokens)
    context_tokens = estimate_tokens(context)

    if not settings.has_api_key:
        return (
            "⚠ GOOGLE_API_KEY is not set. "
            "Add it to your .env file (in backend/ or project root). "
            "Get a free key at https://aistudio.google.com/apikey",
            None,
            context_tokens,
        )

    system_prompt = (
        "You are a strict RAG assistant. You MUST answer ONLY from the provided context snippets below.\n"
        "RULES:\n"
        "- Do NOT use any external knowledge, training data, or general knowledge.\n"
        "- If the answer cannot be found in the provided context, respond EXACTLY: "
        "\"I don't have enough information in the provided documents to answer this question.\"\n"
        "- Never guess, speculate, or infer beyond what the context explicitly states.\n"
        "- Cite evidence using snippet numbers like [1], [2].\n"
        "- Be concise and factual. Every claim must trace to a snippet."
    )
    user_prompt = f"Question:\n{question}\n\nContext:\n{context}"

    try:
        client = genai.Client(api_key=settings.google_api_key)
        model_name = model_override or settings.default_model

        response = client.models.generate_content_stream(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.15,
                max_output_tokens=8192,
            ),
        )

        answer_parts: list[str] = []
        for chunk in response:
            if chunk.text:
                answer_parts.append(chunk.text)

        answer = "".join(answer_parts).strip()
        response_tokens = estimate_tokens(answer)
        return answer or "No answer returned by the model.", response_tokens, context_tokens

    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        if "rate_limit" in err_msg.lower() or "429" in err_msg or "quota" in err_msg.lower():
            return (
                "Google API rate limit reached. Please retry after a short wait.",
                None,
                context_tokens,
            )
        return (
            f"Google API request failed: {err_msg}",
            None,
            context_tokens,
        )
