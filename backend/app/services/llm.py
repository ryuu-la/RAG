from __future__ import annotations

from typing import Any

from groq import Groq

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

    if not settings.groq_api_key.strip():
        return (
            "Groq API key is not configured. Add GROQ_API_KEY in .env to enable answer generation.",
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
        client = Groq(api_key=settings.groq_api_key)
        completion = client.chat.completions.create(
            model=model_override or settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
            max_completion_tokens=8192,
            top_p=1,
            stream=True,
            stop=None,
        )

        answer_parts: list[str] = []
        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta.content:
                answer_parts.append(delta.content)

        answer = "".join(answer_parts).strip()
        response_tokens = estimate_tokens(answer)
        return answer or "No answer returned by the model.", response_tokens, context_tokens

    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        if "rate_limit" in err_msg.lower() or "429" in err_msg:
            return (
                "Groq rate limit reached. Please retry after a short wait or choose another model.",
                None,
                context_tokens,
            )
        return (
            f"Groq request failed: {err_msg}",
            None,
            context_tokens,
        )
