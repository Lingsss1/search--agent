import json

from scripts import run_sft_gate


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"choices": [{"text": "ok"}], "usage": {"completion_tokens": 2}}).encode()


def test_vllm_generation_forwards_deterministic_action_seed(monkeypatch):
    captured = {}

    class _Opener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            return _Response()

    monkeypatch.setattr(run_sft_gate, "build_opener", lambda *_args: _Opener())
    text, tokens = run_sft_gate._generate_http(
        "http://generation",
        "vllm",
        "checkpoint",
        [1, 2, 3],
        max_completion_tokens=16,
        temperature=0.8,
        top_p=0.95,
        top_k=20,
        seed=612,
        stop_token_ids=[4],
    )

    assert (text, tokens) == ("ok", 2)
    assert captured["url"] == "http://generation/v1/completions"
    assert captured["payload"]["seed"] == 612
    assert captured["payload"]["prompt"] == [1, 2, 3]
