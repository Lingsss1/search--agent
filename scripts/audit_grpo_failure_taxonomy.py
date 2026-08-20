#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.agent.schemas import (
    ActionValidationError,
    AnswerAction,
    parse_action,
)
from openstatesearch.data.corpus import read_jsonl
from openstatesearch.eval.full_reward_audit import (
    extract_policy_input,
    load_rollout_trajectories,
)
from openstatesearch.rewards.metrics import answer_f1


def _parse_exported_action(completion: str) -> Any | None:
    text = _clean_completion(completion)
    decoder = json.JSONDecoder()
    parsed = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            parsed = parse_action(value)
        except (json.JSONDecodeError, ActionValidationError, TypeError):
            continue
    return parsed


def _clean_completion(completion: str) -> str:
    return completion.replace("<|im_end|>", "").strip()


def _accepted_answer(action: Any, policy_input: dict[str, Any]) -> bool:
    if not isinstance(action, AnswerAction):
        return False
    legal = policy_input.get("legal_action_space", {})
    if "ANSWER" not in legal.get("allowed_types", []):
        return False
    legal_refs = {
        (str(item["doc_id"]), int(sent_id))
        for item in legal.get("legal_citations", [])
        for sent_id in item.get("sent_ids", [])
    }
    return all(
        (citation.doc_id, sent_id) in legal_refs
        for citation in action.citations
        for sent_id in citation.sent_ids
    )


def classify(trajectory: dict[str, Any], references: list[str]) -> dict[str, Any]:
    interactions = trajectory["interactions"]
    parsed = [_parse_exported_action(str(item["completion"])) for item in interactions]
    cleaned = [_clean_completion(str(item["completion"])) for item in interactions]
    inputs = [extract_policy_input(str(item["prompt"])) for item in interactions]
    empty_completion = any(not value for value in cleaned)
    malformed_json = any(value and action is None for value, action in zip(cleaned, parsed))
    json_error = empty_completion or malformed_json
    state_error = False
    accepted: AnswerAction | None = None
    for index, action in enumerate(parsed):
        if action is None:
            continue
        current = inputs[index]
        allowed = current.get("legal_action_space", {}).get("allowed_types", [])
        if action.type.value not in allowed:
            state_error = True
        if isinstance(action, AnswerAction):
            if _accepted_answer(action, current):
                accepted = action
                break
            state_error = True
            continue
        if index + 1 < len(inputs):
            result = inputs[index + 1].get("last_tool_result")
            if isinstance(result, dict) and result.get("action") == action.type.value:
                state_error |= not bool(result.get("ok"))
    f1 = answer_f1(accepted.answer, references) if accepted else 0.0
    return {
        "trajectory_key": trajectory["trajectory_key"],
        "interactions": len(interactions),
        "tool_json_error": json_error,
        "empty_completion": empty_completion,
        "malformed_json": malformed_json,
        "terminal_empty_completion": not cleaned[-1],
        "answer_attempt": any(isinstance(action, AnswerAction) for action in parsed),
        "state_reference_error": state_error,
        "no_accepted_answer": accepted is None,
        "accepted_answer": accepted is not None,
        "answer_f1": f1,
        "legal_answer_low_quality": accepted is not None and f1 < 1.0,
        "legal_answer_exact": accepted is not None and f1 == 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prompts = {item["question"]: item for item in read_jsonl(args.prompts)}
    report: dict[str, Any] = {"schema_version": 1, "steps": {}}
    for step in args.steps:
        trajectories, _ = load_rollout_trajectories(args.rollout_root, step)
        records = []
        for trajectory in trajectories:
            prompt = prompts[trajectory["question"]]
            references = prompt.get("answers", [prompt.get("answer", "")])
            if isinstance(references, str):
                references = [references]
            records.append(classify(trajectory, [str(item) for item in references]))
        counts = Counter()
        for record in records:
            for key, value in record.items():
                if isinstance(value, bool) and value:
                    counts[key] += 1
        report["steps"][str(step)] = {
            "trajectories": len(records),
            "mean_interactions": sum(item["interactions"] for item in records) / len(records),
            "counts": dict(sorted(counts.items())),
            "rates": {key: value / len(records) for key, value in sorted(counts.items())},
            "accepted_answer_f1_mean": (
                sum(item["answer_f1"] for item in records if item["accepted_answer"])
                / max(1, sum(item["accepted_answer"] for item in records))
            ),
            "records": records,
        }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    compact = {
        "schema_version": report["schema_version"],
        "steps": {
            step: {key: value for key, value in result.items() if key != "records"}
            for step, result in report["steps"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
