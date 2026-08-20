from __future__ import annotations

import pytest

from openstatesearch.agent.schemas import KeepAction, OpenAction, SearchAction
from openstatesearch.rewards import (
    ABCCreditConfig,
    EvidenceCreditTracker,
    combine_group_advantages,
)


GOLD = (("gold-doc", 1), ("gold-doc", 3))


def test_abc_scores_first_search_open_keep_progress_only_once() -> None:
    tracker = EvidenceCreditTracker(GOLD)
    search = SearchAction("query", "answer")
    search_result = {
        "ok": True,
        "action": "SEARCH",
        "payload": {"hits": [{"doc_id": "gold-doc"}, {"doc_id": "other"}]},
    }

    first_search = tracker.score(search, search_result)
    repeated_search = tracker.score(search, search_result)
    opened = tracker.score(
        OpenAction("gold-doc"),
        {
            "ok": True,
            "action": "OPEN",
            "payload": {
                "sentences": [
                    {"sent_id": 0, "text": "x"},
                    {"sent_id": 1, "text": "gold"},
                    {"sent_id": 3, "text": "gold"},
                ]
            },
        },
    )
    kept = tracker.score(
        KeepAction("gold-doc", (1,), "claim", "answer"),
        {"ok": True, "action": "KEEP", "payload": {}},
    )
    repeated_keep = tracker.score(
        KeepAction("gold-doc", (1,), "different claim", "answer"),
        {"ok": True, "action": "KEEP", "payload": {}},
    )

    assert first_search.process_reward == pytest.approx(0.025)
    assert repeated_search.process_reward == 0.0
    assert opened.process_reward == pytest.approx(0.05)
    assert kept.process_reward == pytest.approx(0.0875)
    assert repeated_keep.process_reward == 0.0
    assert tracker.phi == pytest.approx(0.65)


def test_abc_keeps_useful_failed_prefix_distinct_from_terminal_failure() -> None:
    tracker = EvidenceCreditTracker((GOLD[0],))
    prefix = tracker.score(
        SearchAction("query", "answer"),
        {
            "ok": True,
            "action": "SEARCH",
            "payload": {"hits": [{"doc_id": "gold-doc"}]},
        },
    )
    malformed = tracker.score(None, {"ok": False, "action": "INVALID"})

    assert prefix.process_reward > 0.0
    assert malformed.process_reward == pytest.approx(-0.05)


def test_abc_caps_positive_and_negative_process_credit_separately() -> None:
    config = ABCCreditConfig(
        invalid_action_penalty=-0.2,
        process_positive_cap=0.25,
        process_negative_cap=0.10,
    )
    tracker = EvidenceCreditTracker((), config)

    rewards = [
        tracker.score(None, {"ok": False, "action": "INVALID"}).process_reward for _ in range(3)
    ]

    assert rewards == pytest.approx([-0.1, 0.0, 0.0])


def test_invalid_actions_do_not_consume_later_positive_evidence_credit() -> None:
    tracker = EvidenceCreditTracker((GOLD[0],))
    for _ in range(3):
        tracker.score(None, {"ok": False, "action": "INVALID"})

    search = tracker.score(
        SearchAction("query", "answer"),
        {
            "ok": True,
            "action": "SEARCH",
            "payload": {"hits": [{"doc_id": "gold-doc"}]},
        },
    )
    opened = tracker.score(
        OpenAction("gold-doc"),
        {
            "ok": True,
            "action": "OPEN",
            "payload": {"sentences": [{"sent_id": 1, "text": "gold"}]},
        },
    )
    kept = tracker.score(
        KeepAction("gold-doc", (1,), "claim", "answer"),
        {"ok": True, "action": "KEEP", "payload": {}},
    )

    assert [search.process_reward, opened.process_reward, kept.process_reward] == pytest.approx(
        [0.025, 0.05, 0.175]
    )


def test_identical_terminal_group_gets_zero_episode_but_nonzero_process_credit() -> None:
    combined = combine_group_advantages(
        [-1.0, -1.0, -1.0, -1.0],
        [[0.025, 0.05], [0.0], [-0.05], [0.0, 0.0]],
    )

    expected = [[0.025, 0.05], [0.0], [-0.05], [0.0, 0.0]]
    for actual_row, expected_row in zip(combined, expected):
        assert actual_row == pytest.approx(expected_row)


def test_process_credit_does_not_reverse_clear_terminal_ordering() -> None:
    combined = combine_group_advantages(
        [-1.0, -1.0, 1.0, 3.0],
        [[0.25], [0.0], [-0.25], [-0.25]],
    )

    assert combined[3][0] > combined[2][0] > combined[0][0] > combined[1][0]
