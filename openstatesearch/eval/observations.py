from __future__ import annotations

import copy
import json
from typing import Any

from openstatesearch.training.areal_agent import SYSTEM_PROMPT


def _chat_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Non-Qwen test/tokenizer implementations may not expose the optional
        # chat-template keyword, while the production Qwen3.6 tokenizer does.
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    return len(rendered)


def _compact_tool_payload(observation: dict[str, Any]) -> None:
    result = observation.get("tool_result") or {}
    payload = result.get("payload") or {}
    for sentence in payload.get("sentences") or []:
        sentence["text"] = str(sentence.get("text", ""))[:768]
    for hit in payload.get("hits") or []:
        hit["snippet"] = str(hit.get("snippet", ""))[:256]


def transcript_policy_messages(
    *,
    question: str,
    constraints: list[str],
    initial_budget: dict[str, int],
    initial_remaining_turns: int,
    initial_legal_action_space: dict[str, Any],
    history: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_tokens: int,
) -> tuple[list[dict[str, str]], int]:
    """Render a gold-free raw transcript under a fixed sliding-window budget.

    The initial question is always retained.  When the transcript exceeds the
    budget, complete oldest assistant/tool-result pairs are dropped first.  The
    newest OPEN payload is then shortened only as much as required, preserving
    document and sentence identifiers in every retained sentence.
    """
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    work = copy.deepcopy(history)
    for item in work:
        observation = item.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("history observation must be an object")
        if not isinstance(item.get("assistant"), str):
            raise ValueError("history assistant content must be a string")
        _compact_tool_payload(observation)
    dropped = 0

    def messages() -> list[dict[str, str]]:
        values = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "constraints": constraints,
                        "budget": initial_budget,
                        "remaining_turns": initial_remaining_turns,
                        "legal_action_space": initial_legal_action_space,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        for index, item in enumerate(work):
            values.append({"role": "assistant", "content": item["assistant"]})
            observation = copy.deepcopy(item["observation"])
            if index == 0 and dropped:
                observation["dropped_prior_turns"] = dropped
            values.append(
                {
                    "role": "user",
                    "content": json.dumps(observation, ensure_ascii=False),
                }
            )
        return values

    while len(work) > 1 and _chat_tokens(tokenizer, messages()) > max_prompt_tokens:
        work.pop(0)
        dropped += 1

    if work:
        newest_payload = (work[-1]["observation"].get("tool_result") or {}).get("payload") or {}
        sentences = newest_payload.get("sentences") or []
        while len(sentences) > 1 and _chat_tokens(tokenizer, messages()) > max_prompt_tokens:
            sentences.pop()
        if _chat_tokens(tokenizer, messages()) > max_prompt_tokens:
            for item in work:
                payload = (item["observation"].get("tool_result") or {}).get("payload") or {}
                for hit in payload.get("hits") or []:
                    hit["snippet"] = ""
        while sentences and _chat_tokens(tokenizer, messages()) > max_prompt_tokens:
            text = str(sentences[-1].get("text", ""))
            if not text:
                sentences.pop()
            else:
                sentences[-1]["text"] = text[: len(text) // 2]

    rendered = messages()
    if _chat_tokens(tokenizer, rendered) > max_prompt_tokens:
        raise ValueError("latest transcript observation exceeds the configured memory budget")
    return rendered, dropped
