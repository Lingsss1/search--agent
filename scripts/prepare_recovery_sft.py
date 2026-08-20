#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.retriever.types import split_sentences
from openstatesearch.training.areal_agent import SYSTEM_PROMPT


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def get_json(base_url: str, path: str) -> Any:
    with urlopen(f"{base_url.rstrip('/')}{path}", timeout=300) as response:
        return json.loads(response.read())


def record(identity: str, suffix: str, policy_input: dict[str, Any], action: dict[str, Any]):
    return {
        "id": f"{identity}:{suffix}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(policy_input, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build recovery SFT from real frozen R4 train rollouts"
    )
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-dataset", type=int, default=500)
    parser.add_argument("--offset", type=int, default=500)
    args = parser.parse_args()

    prompts = list(read_jsonl(Path(args.prompts)))
    datasets = sorted({str(item["dataset"]) for item in prompts})
    selected = []
    for dataset in datasets:
        values = [item for item in prompts if str(item["dataset"]) == dataset]
        subset = values[args.offset : args.offset + args.per_dataset]
        if len(subset) != args.per_dataset:
            raise ValueError(f"{dataset}: requested {args.per_dataset}, got {len(subset)}")
        selected.extend(subset)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    action_counts: dict[str, int] = {}
    rows = 0
    trajectories = 0
    with output.open("w", encoding="utf-8") as handle:
        for prompt_index, prompt in enumerate(selected):
            identity = str(prompt["id"])
            question = str(prompt["question"])
            constraints = list(prompt.get("constraints", []))
            gold = {
                str(item["doc_id"]): [int(v) for v in item["sent_ids"]]
                for item in prompt.get("gold_evidence", [])
            }
            raw_hits = get_json(
                args.retriever_url,
                f"/search?{urlencode({'q': question, 'k': 5})}",
            )
            if not raw_hits:
                continue
            hits = [
                {
                    "doc_id": str(hit["doc_id"]),
                    "title": str(hit.get("title", "")),
                    "score": float(hit["score"]),
                    "snippet": str(hit.get("snippet", "")),
                }
                for hit in raw_hits
            ]
            documents = {
                hit["doc_id"]: get_json(
                    args.retriever_url, f"/documents/{quote(hit['doc_id'], safe='')}"
                )
                for hit in hits[:4]
            }
            state = {
                "question": question,
                "constraints": constraints,
                "candidate_pool": [],
                "evidence": [],
                "conflicts": [],
                "query_history": [],
                "budget": {"search_left": 4, "open_left": 4, "token_left": 8192},
            }

            def emit(
                suffix: str, last: Any, opened: list[str], remaining: int, action: dict[str, Any]
            ):
                nonlocal rows
                payload = {
                    "state": json.loads(json.dumps(state)),
                    "opened_doc_ids": list(opened),
                    "remaining_turns": remaining,
                    "last_tool_result": last,
                }
                handle.write(
                    json.dumps(record(identity, suffix, payload, action), ensure_ascii=False) + "\n"
                )
                rows += 1
                action_counts[action["type"]] = action_counts.get(action["type"], 0) + 1

            search = {"type": "SEARCH", "query": question, "target_constraint": "answer_evidence"}
            emit("search", None, [], 16, search)
            state["candidate_pool"] = hits
            state["query_history"] = [question]
            state["budget"]["search_left"] = 3
            search_result = {
                "ok": True,
                "action": "SEARCH",
                "payload": {"query": question, "target": "answer_evidence", "hits": raw_hits},
                "error": None,
                "duplicate": False,
            }
            emit(
                "post_search", search_result, [], 15, {"type": "OPEN", "doc_id": hits[0]["doc_id"]}
            )
            duplicate_search = {
                "ok": True,
                "action": "SEARCH",
                "payload": {"query": question, "hits": []},
                "error": None,
                "duplicate": True,
            }
            emit(
                "duplicate_search",
                duplicate_search,
                [],
                14,
                {"type": "OPEN", "doc_id": hits[0]["doc_id"]},
            )

            opened: list[str] = []
            evidence: list[dict[str, Any]] = []
            last_open: dict[str, Any] | None = None
            for rank, hit in enumerate(hits[:4]):
                doc_id = hit["doc_id"]
                document = documents[doc_id]
                sentences = list(split_sentences(str(document.get("text", "")))) or [
                    str(document.get("text", ""))
                ]
                opened.append(doc_id)
                state["budget"]["open_left"] = 4 - len(opened)
                last_open = {
                    "ok": True,
                    "action": "OPEN",
                    "payload": {
                        "doc_id": doc_id,
                        "title": str(document.get("title", "")),
                        "sentences": [
                            {"sent_id": sent_id, "text": text}
                            for sent_id, text in enumerate(sentences)
                        ],
                    },
                    "error": None,
                    "duplicate": False,
                }
                legal_gold = [
                    index for index in gold.get(doc_id, []) if 0 <= index < len(sentences)
                ]
                if legal_gold:
                    legal_gold = legal_gold[:2]
                    claim = " ".join(sentences[index] for index in legal_gold)[:512]
                    keep = {
                        "type": "KEEP",
                        "doc_id": doc_id,
                        "sent_ids": legal_gold,
                        "claim": claim,
                        "constraint_id": "answer_evidence",
                    }
                    emit(f"keep_{rank}", last_open, opened, max(3, 14 - rank), keep)
                    evidence.append(
                        {k: keep[k] for k in ("doc_id", "sent_ids", "claim", "constraint_id")}
                    )
                    state["evidence"] = list(evidence)
                    duplicate_open = dict(last_open)
                    duplicate_open["duplicate"] = True
                    emit(
                        f"duplicate_open_keep_{rank}",
                        duplicate_open,
                        opened,
                        max(3, 13 - rank),
                        keep,
                    )
                elif rank + 1 < min(4, len(hits)):
                    next_open = {"type": "OPEN", "doc_id": hits[rank + 1]["doc_id"]}
                    emit(f"irrelevant_open_{rank}", last_open, opened, max(3, 14 - rank), next_open)
                    duplicate_open = dict(last_open)
                    duplicate_open["duplicate"] = True
                    emit(
                        f"duplicate_open_{rank}",
                        duplicate_open,
                        opened,
                        max(3, 13 - rank),
                        next_open,
                    )

            citations = [
                {"claim": item["claim"], "doc_id": item["doc_id"], "sent_ids": item["sent_ids"]}
                for item in evidence[:2]
            ]
            if not citations:
                first_id = opened[0]
                first_sentences = list(split_sentences(str(documents[first_id].get("text", ""))))
                citations = [
                    {"claim": first_sentences[0][:512], "doc_id": first_id, "sent_ids": [0]}
                ]
            raw_answers = prompt.get("answers", [prompt.get("answer", "")])
            answer = str(raw_answers[0] if isinstance(raw_answers, list) else raw_answers)
            final_action = {"type": "ANSWER", "answer": answer, "citations": citations}
            emit("terminal_answer", last_open, opened, 2, final_action)
            trajectories += 1
            if (prompt_index + 1) % 100 == 0:
                print(json.dumps({"processed": prompt_index + 1, "rows": rows}), flush=True)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": str(Path(args.prompts).resolve()),
        "held_out_used": False,
        "retriever": args.retriever_url,
        "selection": {"offset": args.offset, "per_dataset": args.per_dataset},
        "trajectories": trajectories,
        "records": rows,
        "actions": action_counts,
        "sha256": digest,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
