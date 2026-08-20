from __future__ import annotations

import asyncio
import json

from openstatesearch.rewards import RewardBreakdown
from openstatesearch.training.areal_agent import (
    OpenStateSearchWorkflow,
    _stable_sampling_seed,
)


def _breakdown(valid: bool) -> RewardBreakdown:
    return RewardBreakdown(
        total=0.0,
        answer_f1=0.0,
        support_recall=0.0,
        citation_precision=0.0,
        duplicate_rate=0.0,
        search_cost=0.0,
        open_cost=0.0,
        token_cost=0.0,
        protocol_penalty=0.0,
        valid=valid,
        phase="A",
    )


def test_sampling_seed_is_stable_and_separates_group_turn_and_version() -> None:
    seed = _stable_sampling_seed(36, 0, "trajectory-a", 2, 3)

    assert seed == _stable_sampling_seed(36, 0, "trajectory-a", 2, 3)
    assert 0 <= seed < 2**31
    assert seed != _stable_sampling_seed(36, 0, "trajectory-a", 3, 3)
    assert seed != _stable_sampling_seed(36, 0, "trajectory-a", 2, 4)
    assert seed != _stable_sampling_seed(36, 1, "trajectory-a", 2, 3)


def test_reward_audit_cap_is_shared_across_workflow_instances(tmp_path) -> None:
    path = tmp_path / "reward_audit.jsonl"
    workflows = []
    for _ in range(16):
        workflow = OpenStateSearchWorkflow.__new__(OpenStateSearchWorkflow)
        workflow.audit_path = path
        workflow.audit_every_steps = 100
        workflow.audit_sample_size = 50
        workflows.append(workflow)

    breakdown = _breakdown(True)

    async def record_grouped_rollouts() -> None:
        await asyncio.gather(
            *(
                workflow._record_audit(
                    100,
                    {"id": f"trajectory-{group}-{sample}"},
                    breakdown,
                )
                for group, workflow in enumerate(workflows)
                for sample in range(4)
            )
        )

    asyncio.run(record_grouped_rollouts())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 50
    assert {record["step"] for record in records} == {100}


def test_reward_audit_cap_survives_new_workflow_and_step(tmp_path) -> None:
    path = tmp_path / "reward_audit.jsonl"

    async def record(version: int, count: int) -> None:
        for index in range(count):
            workflow = OpenStateSearchWorkflow.__new__(OpenStateSearchWorkflow)
            workflow.audit_path = path
            workflow.audit_every_steps = 100
            workflow.audit_sample_size = 2
            await workflow._record_audit(
                version,
                {"id": f"{version}-{index}"},
                _breakdown(False),
            )

    asyncio.run(record(100, 3))
    asyncio.run(record(200, 3))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["step"] for record in records].count(100) == 2
    assert [record["step"] for record in records].count(200) == 2
