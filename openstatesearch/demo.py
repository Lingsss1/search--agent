from __future__ import annotations

import json

from openstatesearch.agent.harness import SearchHarness
from openstatesearch.agent.schemas import parse_action
from openstatesearch.agent.state import SearchState
from openstatesearch.retriever import Document, HybridRetriever


def demo_documents() -> list[Document]:
    return [
        Document.create(
            "Ada Lovelace",
            "Ada Lovelace was an English mathematician. She published notes on Charles Babbage's Analytical Engine in 1843.",
            "demo",
        ),
        Document.create(
            "Analytical Engine",
            "The Analytical Engine was designed by Charles Babbage. It was a proposed mechanical general-purpose computer.",
            "demo",
        ),
        Document.create(
            "Charles Babbage",
            "Charles Babbage was born in London. He originated the concept of a digital programmable computer.",
            "demo",
        ),
        Document.create("London", "London is the capital and largest city of England.", "demo"),
        Document.create(
            "Difference Engine",
            "The Difference Engine was designed to tabulate polynomial functions.",
            "demo",
        ),
        Document.create(
            "Mathematics",
            "Mathematics includes the study of numbers, structures, space and change.",
            "demo",
        ),
    ]


def run_demo() -> dict[str, object]:
    documents = demo_documents()
    retriever = HybridRetriever(documents)
    state = SearchState("Who designed the machine for which Ada Lovelace published notes?")
    harness = SearchHarness(state, retriever)

    search = parse_action(
        {
            "type": "SEARCH",
            "query": "Ada Lovelace Analytical Engine notes",
            "target_constraint": "machine",
        }
    )
    search_result = harness.apply(search)
    target = next(hit for hit in search_result.payload["hits"] if hit["title"] == "Ada Lovelace")
    harness.apply(parse_action({"type": "OPEN", "doc_id": target["doc_id"]}))
    harness.apply(
        parse_action(
            {
                "type": "KEEP",
                "doc_id": target["doc_id"],
                "sent_ids": [1],
                "claim": "Lovelace published notes on Babbage's Analytical Engine.",
                "constraint_id": "machine",
            }
        )
    )
    harness.apply(
        parse_action(
            {
                "type": "ANSWER",
                "answer": "Charles Babbage",
                "citations": [
                    {
                        "claim": "The notes were about Babbage's Analytical Engine.",
                        "doc_id": target["doc_id"],
                        "sent_ids": [1],
                    }
                ],
            }
        )
    )
    return harness.trajectory()


def main() -> None:
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
