#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.eval.final_results import assemble_final_results
from openstatesearch.training import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble evidence-backed final results and run every acceptance gate"
    )
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--retriever-results", required=True)
    parser.add_argument("--reward-audit-summary", required=True)
    parser.add_argument("--demo-manifest", required=True)
    parser.add_argument(
        "--cost-metric",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="repeat for main_test, in_domain_dev, and chinese_test",
    )
    parser.add_argument(
        "--dataset-manifest",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=("repeat for in_domain_dev, browsecomp_plus, xbench_deepsearch, and browsecomp_zh"),
    )
    parser.add_argument("--thresholds", default="configs/acceptance.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--retriever-target", default="R4")
    parser.add_argument("--retriever-baseline", default="R0")
    parser.add_argument("--external-target", default="D")
    parser.add_argument("--external-baseline", default="C")
    parser.add_argument("--grpo-target", default="F")
    parser.add_argument("--grpo-baseline", default="D")
    parser.add_argument("--comparison-budget", type=int, default=8192)
    args = parser.parse_args()
    cost_metrics = {}
    for value in args.cost_metric:
        if "=" not in value:
            parser.error("--cost-metric must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path or name in cost_metrics:
            parser.error("cost metric names/paths must be non-empty and unique")
        cost_metrics[name] = path
    dataset_manifests = {}
    for value in args.dataset_manifest:
        if "=" not in value:
            parser.error("--dataset-manifest must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path or name in dataset_manifests:
            parser.error("dataset manifest names/paths must be non-empty and unique")
        dataset_manifests[name] = path
    results, report = assemble_final_results(
        matrix_summary_path=args.matrix_summary,
        retriever_results_path=args.retriever_results,
        reward_audit_summary_path=args.reward_audit_summary,
        demo_manifest_path=args.demo_manifest,
        cost_metric_paths=cost_metrics,
        dataset_manifest_paths=dataset_manifests,
        thresholds=load_config(args.thresholds),
        output_path=args.output,
        retriever_target=args.retriever_target,
        retriever_baseline=args.retriever_baseline,
        external_target=args.external_target,
        external_baseline=args.external_baseline,
        grpo_target=args.grpo_target,
        grpo_baseline=args.grpo_baseline,
        comparison_budget=args.comparison_budget,
    )
    print(json.dumps({"results": results, "acceptance": report}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
