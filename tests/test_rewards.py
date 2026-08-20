import unittest

from openstatesearch.rewards import TrajectoryOutcome, answer_f1, compute_reward


def outcome(**overrides):
    values = dict(
        prediction="Charles Babbage",
        references=("Charles Babbage",),
        predicted_evidence=(("d", 1),),
        gold_evidence=(("d", 1),),
        citations=(("d", 1),),
        queries=("query",),
        returned_doc_ids=(("d",),),
        search_count=1,
        open_count=1,
        generated_tokens=100,
    )
    values.update(overrides)
    return TrajectoryOutcome(**values)


class RewardTests(unittest.TestCase):
    def test_answer_f1_uses_best_reference(self):
        self.assertEqual(answer_f1("Babbage", ["Charles Babbage", "Babbage"]), 1.0)

    def test_phase_a_exact_formula(self):
        reward = compute_reward(outcome(), "A")
        self.assertAlmostEqual(reward.total, 3.2)
        self.assertEqual(reward.search_cost, 0)

    def test_phase_b_cost_only_after_quality_gate(self):
        good = compute_reward(outcome(), "B")
        self.assertAlmostEqual(good.total, 3.15)
        poor = compute_reward(outcome(prediction="wrong"), "B")
        self.assertEqual(poor.search_cost, 0)
        self.assertAlmostEqual(poor.total, 0.7)

    def test_invalid_tool_or_citation_is_negative_one(self):
        self.assertEqual(compute_reward(outcome(valid_tools=False), "A").total, -1)
        self.assertEqual(compute_reward(outcome(valid_citations=False), "B").total, -1)

    def test_protocol_errors_are_bounded_and_recoverable(self):
        recovered = compute_reward(outcome(invalid_action_count=2), "A")
        self.assertAlmostEqual(recovered.protocol_penalty, 0.1)
        self.assertAlmostEqual(recovered.total, 3.1)
        failed = compute_reward(outcome(valid_tools=False, invalid_action_count=100), "A")
        self.assertEqual(failed.protocol_penalty, 0.5)
        self.assertEqual(failed.total, -1.0)

    def test_terminally_invalid_reward_does_not_prefer_early_exit(self):
        immediate_eos = compute_reward(outcome(valid_tools=False, invalid_action_count=1), "A")
        attempted_recovery = compute_reward(outcome(valid_tools=False, invalid_action_count=7), "A")
        self.assertEqual(immediate_eos.total, attempted_recovery.total)
        self.assertLess(immediate_eos.protocol_penalty, attempted_recovery.protocol_penalty)

    def test_duplicate_penalty(self):
        repeated = compute_reward(
            outcome(queries=("q", "Q"), returned_doc_ids=(("d",), ("d",))), "A"
        )
        self.assertAlmostEqual(repeated.duplicate_rate, 0.5)
        self.assertAlmostEqual(repeated.total, 3.125)


if __name__ == "__main__":
    unittest.main()
