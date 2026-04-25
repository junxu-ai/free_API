from typing import Iterable

from free_llm_router.schemas import ChatMessage


def flatten_messages(messages: Iterable[ChatMessage]) -> str:
    parts = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
        elif content is not None:
            parts.append(str(content))
    return "\n".join(parts).lower()


def classify_scenario(messages: Iterable[ChatMessage]) -> str:
    text = flatten_messages(messages)

    if any(token in text for token in ["image", "vision", "screenshot", "ocr"]):
        return "vision"
    if any(token in text for token in ["agent", "workflow", "tool call", "tool-use", "browse", "multi-step plan", "orchestrate"]):
        return "agentic"
    if any(token in text for token in ["python", "javascript", "typescript", "refactor", "debug", "stack trace", "function", "class ", "write code", "sql query"]):
        return "coding"
    if any(token in text for token in ["reason step by step", "step by step", "prove", "derive", "logic puzzle", "analyze carefully", "reasoning", "chain of thought", "mathematics"]):
        return "reasoning"
    if any(token in text for token in ["summarize", "summary", "tldr", "brief this"]):
        return "summarization"
    return "generation"
