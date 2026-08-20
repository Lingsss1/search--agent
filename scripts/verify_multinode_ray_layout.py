#!/usr/bin/env python3
"""Reserve the intended AReaL bundles and prove their physical placement."""

from __future__ import annotations

import json
import os

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_cpus=1, num_gpus=4)
def placement_probe(role: str) -> dict[str, str]:
    return {
        "role": role,
        "node_ip": ray.util.get_node_ip_address(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def probe(pg: object, bundle_index: int, role: str) -> dict[str, str]:
    strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=bundle_index,
        placement_group_capture_child_tasks=True,
    )
    return ray.get(placement_probe.options(scheduling_strategy=strategy).remote(role))


def main() -> None:
    ray.init(address="auto")
    actor_pg = placement_group([{"CPU": 4, "GPU": 4}, {"CPU": 4, "GPU": 4}], strategy="PACK")
    rollout_pg = None
    try:
        ray.get(actor_pg.ready(), timeout=60)
        actor = [probe(actor_pg, index, f"actor-bundle-{index}") for index in range(2)]
        rollout_pg = placement_group([{"CPU": 4, "GPU": 4}], strategy="PACK")
        ray.get(rollout_pg.ready(), timeout=60)
        rollout = probe(rollout_pg, 0, "rollout-tp4")
        report = {"actor": actor, "rollout": rollout}
        print(json.dumps(report, indent=2, sort_keys=True))
        actor_ips = {item["node_ip"] for item in actor}
        if actor_ips != {"10.82.123.139"}:
            raise SystemExit(f"actor bundles escaped the A800 node: {actor_ips}")
        if rollout["node_ip"] != "10.48.41.83":
            raise SystemExit(f"rollout did not land on H800: {rollout}")
    finally:
        if rollout_pg is not None:
            remove_placement_group(rollout_pg)
        remove_placement_group(actor_pg)
        ray.shutdown()


if __name__ == "__main__":
    main()
