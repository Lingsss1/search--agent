import json

import pytest

from scripts.merge_retriever_results import merge_retriever_results


def _metrics(value: float, examples: int = 10) -> dict:
    return {
        "examples": examples,
        "recall_at_5": value,
        "recall_at_20": value + 0.1,
        "recall_at_100": value + 0.2,
        "ndcg_at_10": value + 0.05,
    }


def test_merge_retriever_results_preserves_all_pairwise_gains(tmp_path):
    inputs = []
    for index, name in enumerate(("R0", "R1", "R2", "R3", "R4")):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({name: _metrics(index / 10)}))
        inputs.append(path)
    output = tmp_path / "merged.json"
    result = merge_retriever_results(inputs, output)
    assert list(result["results"]) == ["R0", "R1", "R2", "R3", "R4"]
    assert result["pairwise_absolute_gains"]["R4-R1"]["recall_at_20"] == pytest.approx(0.3)
    assert result["pairwise_absolute_gains"]["R1-R4"]["recall_at_20"] == pytest.approx(-0.3)
    assert result["gain_selection"] is None
    assert output.with_suffix(".json.manifest.json").exists()


def test_merge_retriever_results_rejects_missing_and_conflicting_variants(tmp_path):
    one = tmp_path / "one.json"
    one.write_text(json.dumps({"R0": _metrics(0.0)}))
    with pytest.raises(ValueError, match="missing retriever"):
        merge_retriever_results([one], tmp_path / "out.json")
    two = tmp_path / "two.json"
    two.write_text(json.dumps({"R0": _metrics(0.2)}))
    with pytest.raises(ValueError, match="conflicting"):
        merge_retriever_results([one, two], tmp_path / "out.json")
