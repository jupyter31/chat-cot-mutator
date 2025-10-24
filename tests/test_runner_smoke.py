import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.runner import run_experiment


class MockModelClient:
    def __init__(self):
        self.calls = 0

    def send_chat_request(self, model_name, request):
        self.calls += 1
        if any("Tool results" in message.get("content", "") for message in request["messages"]):
            mutated = "Mutated chain-of-thought [mock]"
            return {
                "choices": [{"message": {"content": mutated}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }

        user_prompt = request["messages"][-1]["content"]
        match = re.search(r"\[([^\]]+)\]", user_prompt)
        cite = match.group(1) if match else "EVIDENCE"
        content = (
            f"Reasoning: rely on [[{cite}]] to answer.\n"
            f"Final Answer: placeholder [[{cite}]]"
        )
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def send_batch_chat_request(self, model_name, batch_requests, batch_size=5):  # pragma: no cover
        raise NotImplementedError

    def send_stream_chat_completion_request(self, model_name, request_data):  # pragma: no cover
        raise NotImplementedError


def test_runner_smoke(tmp_path):
    output_dir = tmp_path / "results"
    config = {
        "input": "data/samples/pilot.jsonl",
        "output_dir": str(output_dir),
        "model": "mock:dummy",
        "temperature": 0.0,
        "seed": 42,
        "conditions": ["A", "B", "C", "D"],
        "judge": "prog",
        "mutation_policy": "pivotal",
        "max_samples": 2,
    }

    result = run_experiment(config, model_client=MockModelClient())

    samples_path = output_dir / "samples.jsonl"
    metrics_overall_path = output_dir / "metrics_overall.json"
    metrics_mutation_path = output_dir / "metrics_by_mutation.json"
    tokens_path = output_dir / "tokens_latency.csv"

    assert samples_path.exists()
    assert metrics_overall_path.exists()
    assert metrics_mutation_path.exists()
    assert tokens_path.exists()

    lines = samples_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8  # 2 samples * 4 conditions
    first_record = json.loads(lines[0])
    for key in ["sample_id", "condition", "response", "judge"]:
        assert key in first_record

    metrics_overall = json.loads(metrics_overall_path.read_text(encoding="utf-8"))
    assert "ACE" in metrics_overall
    assert result["metrics_overall"]["ACE"] == metrics_overall["ACE"]
    metrics_mutation = json.loads(metrics_mutation_path.read_text(encoding="utf-8"))
    assert "pivotal" in metrics_mutation
