from __future__ import annotations

import json
import time
from typing import Any

import ollama


class OllamaError(RuntimeError):
    pass


def chat_with_retry(
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any],
    retries: int = 3,
    timeout_s: int = 90,
) -> tuple[str, float]:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        start = time.perf_counter()
        try:
            rsp = ollama.chat(model=model, messages=messages, options=options)
            text = rsp["message"]["content"]
            return text, time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise OllamaError(f"Ollama call failed after {retries} retries: {last_err}")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        left = text.find("{")
        right = text.rfind("}")
        if left != -1 and right != -1 and right > left:
            return json.loads(text[left : right + 1])
        raise
