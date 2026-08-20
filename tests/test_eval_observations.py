import json

import pytest

from openstatesearch.eval.observations import transcript_policy_messages


class WordTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return " ".join(message["content"] for message in messages).split()


def _frontier():
    return {
        "allowed_types": ["SEARCH"],
        "openable_doc_ids": [],
        "legal_citations": [],
        "must_answer": False,
    }


def _history(turns: int, sentence_words: int = 80):
    return [
        {
            "assistant": json.dumps(
                {"type": "SEARCH", "query": f"query {index}", "target_constraint": "x"}
            ),
            "observation": {
                "tool_result": {
                    "ok": True,
                    "action": "OPEN",
                    "payload": {
                        "sentences": [{"sent_id": 0, "text": ("word " * sentence_words).strip()}]
                    },
                },
                "budget": {"search_left": 3, "open_left": 3, "token_left": 8000},
                "remaining_turns": 15 - index,
                "legal_action_space": _frontier(),
            },
        }
        for index in range(turns)
    ]


def _render(history, budget):
    return transcript_policy_messages(
        question="question without gold",
        constraints=[],
        initial_budget={"search_left": 4, "open_left": 4, "token_left": 8192},
        initial_remaining_turns=16,
        initial_legal_action_space=_frontier(),
        history=history,
        tokenizer=WordTokenizer(),
        max_prompt_tokens=budget,
    )


def test_transcript_preserves_question_and_full_history_when_it_fits():
    messages, dropped = _render(_history(2, sentence_words=5), 1000)
    assert dropped == 0
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "question without gold" in messages[1]["content"]
    assert "gold_evidence" not in "".join(message["content"] for message in messages)


def test_transcript_drops_complete_oldest_turns_under_budget():
    messages, dropped = _render(_history(4, sentence_words=80), 310)
    assert dropped > 0
    assert len(" ".join(message["content"] for message in messages).split()) <= 310
    first_retained_observation = json.loads(messages[3]["content"])
    assert first_retained_observation["dropped_prior_turns"] == dropped


def test_transcript_shrinks_latest_open_payload_without_losing_sentence_id():
    history = _history(1, sentence_words=2000)
    messages, dropped = _render(history, 300)
    assert dropped == 0
    observation = json.loads(messages[-1]["content"])
    sentences = observation["tool_result"]["payload"]["sentences"]
    assert sentences and sentences[0]["sent_id"] == 0
    assert len(" ".join(message["content"] for message in messages).split()) <= 300


def test_transcript_rejects_malformed_history():
    with pytest.raises(ValueError, match="assistant"):
        _render([{"assistant": {}, "observation": {}}], 1000)
