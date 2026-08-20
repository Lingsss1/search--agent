#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.training.areal_agent import SYSTEM_PROMPT
from openstatesearch.retriever.types import split_sentences


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def policy_record(
    identity: str, action_name: str, state: dict[str, Any], result: Any, action: dict[str, Any]
):
    policy_input = {"state": state, "last_tool_result": result}
    return {
        "id": f"{identity}:{action_name}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(policy_input, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build protocol-alignment SFT examples from RL-train gold evidence"
    )
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-dataset", type=int, default=500)
    args = parser.parse_args()

    prompts = list(read_jsonl(Path(args.prompts)))
    datasets = sorted({str(item["dataset"]) for item in prompts})
    selected: list[dict[str, Any]] = []
    for dataset in datasets:
        values = [item for item in prompts if str(item["dataset"]) == dataset][: args.per_dataset]
        if len(values) != args.per_dataset:
            raise ValueError(f"{dataset} has {len(values)} prompts, expected {args.per_dataset}")
        selected.extend(values)

    needed = {
        str(item["doc_id"]) for prompt in selected for item in prompt.get("gold_evidence", [])
    }
    documents: dict[str, dict[str, Any]] = {}
    for document in read_jsonl(Path(args.corpus)):
        doc_id = str(document["doc_id"])
        if doc_id in needed:
            documents[doc_id] = document
    missing = needed - set(documents)
    if missing:
        raise ValueError(f"corpus is missing {len(missing)} gold documents")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    action_counts: dict[str, int] = {}
    records = 0
    with output.open("w", encoding="utf-8") as handle:
        for prompt in selected:
            identity = str(prompt["id"])
            question = str(prompt["question"])
            constraints = list(prompt.get("constraints", []))
            evidence_specs: list[tuple[dict[str, Any], list[int], str]] = []
            seen_docs: set[str] = set()
            for item in prompt.get("gold_evidence", []):
                doc_id = str(item["doc_id"])
                sent_ids = [int(value) for value in item["sent_ids"]]
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                document = documents[doc_id]
                sentences = list(split_sentences(str(document.get("text", ""))))
                legal_ids = [index for index in sent_ids if 0 <= index < len(sentences)]
                if not legal_ids:
                    continue
                claim = " ".join(str(sentences[index]) for index in legal_ids).strip()
                evidence_specs.append((document, legal_ids, claim))
            if not evidence_specs:
                continue

            state: dict[str, Any] = {
                "question": question,
                "constraints": constraints,
                "candidate_pool": [],
                "evidence": [],
                "conflicts": [],
                "query_history": [],
                "budget": {"search_left": 4, "open_left": 4, "token_left": 8192},
            }
            search_action = {
                "type": "SEARCH",
                "query": question,
                "target_constraint": "answer_evidence",
            }
            record = policy_record(identity, "search", state, None, search_action)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1
            action_counts["SEARCH"] = action_counts.get("SEARCH", 0) + 1

            hits = []
            for rank, (document, _, _) in enumerate(evidence_specs):
                snippet = " ".join(split_sentences(str(document.get("text", "")))[:2])[:512]
                hit = {
                    "doc_id": str(document["doc_id"]),
                    "title": str(document.get("title", "")),
                    "score": 1.0 - rank * 0.01,
                    "snippet": snippet,
                }
                hits.append(hit)
                state["candidate_pool"].append(hit)
            state["query_history"].append(question)
            state["budget"]["search_left"] -= 1
            last_result: dict[str, Any] = {
                "ok": True,
                "action": "SEARCH",
                "payload": {"query": question, "target": "answer_evidence", "hits": hits},
                "error": None,
                "duplicate": False,
            }

            citations = []
            for index, (document, sent_ids, claim) in enumerate(evidence_specs):
                doc_id = str(document["doc_id"])
                open_action = {"type": "OPEN", "doc_id": doc_id}
                record = policy_record(identity, f"open{index}", state, last_result, open_action)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records += 1
                action_counts["OPEN"] = action_counts.get("OPEN", 0) + 1

                state["budget"]["open_left"] -= 1
                last_result = {
                    "ok": True,
                    "action": "OPEN",
                    "payload": {
                        "doc_id": doc_id,
                        "title": str(document.get("title", "")),
                        "sentences": [
                            {"sent_id": sent_id, "text": str(text)}
                            for sent_id, text in enumerate(
                                split_sentences(str(document.get("text", "")))
                            )
                        ],
                    },
                    "error": None,
                    "duplicate": False,
                }
                keep_action = {
                    "type": "KEEP",
                    "doc_id": doc_id,
                    "sent_ids": sent_ids,
                    "claim": claim,
                    "constraint_id": "answer_evidence",
                }
                record = policy_record(identity, f"keep{index}", state, last_result, keep_action)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records += 1
                action_counts["KEEP"] = action_counts.get("KEEP", 0) + 1
                evidence = {
                    "doc_id": doc_id,
                    "sent_ids": sent_ids,
                    "claim": claim,
                    "constraint_id": "answer_evidence",
                }
                state["evidence"].append(evidence)
                last_result = {
                    "ok": True,
                    "action": "KEEP",
                    "payload": evidence,
                    "error": None,
                    "duplicate": False,
                }
                citations.append({"claim": claim, "doc_id": doc_id, "sent_ids": sent_ids})

            answer = str(prompt.get("answer") or prompt.get("answers", [""])[0])
            answer_action = {"type": "ANSWER", "answer": answer, "citations": citations}
            record = policy_record(identity, "answer", state, last_result, answer_action)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1
            action_counts["ANSWER"] = action_counts.get("ANSWER", 0) + 1

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": str(Path(args.prompts).resolve()),
        "held_out_used": False,
        "datasets": {dataset: args.per_dataset for dataset in datasets},
        "trajectories": len(selected),
        "records": records,
        "actions": action_counts,
        "sha256": digest,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
