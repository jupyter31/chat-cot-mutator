import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tools.runner import run_experiment


class FakeModelClient:
    def __init__(self):
        self.requests = []

    def send_chat_request(self, model_name: str, request: dict) -> dict:
        self.requests.append((model_name, request))
        prompt = request["messages"][1]["content"]
        if "STEPS TO FOLLOW:" in prompt:
            content = "Reasoning: following steps [[C1]]\nFinal Answer: mutated [[C2]]"
        elif "Final Answer:" in prompt and "Reasoning:" not in prompt:
            content = "Final Answer: answer [[C1]]"
        else:
            content = "Reasoning: baseline step [[C1]]\nFinal Answer: baseline [[C1]]"
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


@pytest.fixture()
def sample_jsonl(tmp_path: Path) -> Path:
    data = {
        "id": "sample-1",
        "query": "What is the answer?",
        "frozen_context": {
            "passages": [
                {"doc_id": "A", "text": "Some fact.", "cite": "C1"},
            ],
            "tool_outputs": [],
        },
        "cot_baseline": None,
        "mutation_directive": None,
        "grounding_rule": None,
        "answer_gold": "baseline",
    }
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return path


def _run(config: dict, model_client: FakeModelClient) -> dict:
    return run_experiment(config, model_client=model_client)


def test_generated_trace_reused_in_mutations(tmp_path: Path, sample_jsonl: Path) -> None:
    output_dir = tmp_path / "run_generated"
    cfg = {
        "input": str(sample_jsonl),
        "output_dir": str(output_dir),
        "model": "fake:stub",
        "baseline_cot_source": "generate",
        "reuse_cached_A_cots": False,
        "cot_cache_dir": str(tmp_path / "cache_generated"),
    }
    fake = FakeModelClient()
    result = _run(cfg, fake)
    records = result["results"]
    a_record = next(r for r in records if r["condition"] == "A")
    assert a_record["trace_A"], "Expected non-empty trace for generated baseline"
    baseline_trace = a_record["trace_A"]
    assert all(r["baseline_cot_used"] == "generated" for r in records)
    for condition in ("C", "D"):
        record = next(r for r in records if r["condition"] == condition)
        assert record["mutated_cot"].strip() == baseline_trace.strip()


def test_sample_provided_trace_used_when_requested(tmp_path: Path, sample_jsonl: Path) -> None:
    baseline = "Provided CoT step"
    samples = sample_jsonl
    data = json.loads(samples.read_text(encoding="utf-8"))
    data["cot_baseline"] = baseline
    samples.write_text(json.dumps(data) + "\n", encoding="utf-8")

    output_dir = tmp_path / "run_sample"
    cfg = {
        "input": str(samples),
        "output_dir": str(output_dir),
        "model": "fake:stub",
        "baseline_cot_source": "sample",
        "reuse_cached_A_cots": False,
        "cot_cache_dir": str(tmp_path / "cache_sample"),
    }
    fake = FakeModelClient()
    result = _run(cfg, fake)
    records = result["results"]
    a_record = next(r for r in records if r["condition"] == "A")
    assert a_record["trace_A"] == baseline
    assert all(r["baseline_cot_used"] == "sample" for r in records)
    for condition in ("C", "D"):
        record = next(r for r in records if r["condition"] == condition)
        assert record["mutated_cot"].strip() == baseline.strip()


def test_cached_trace_reused_on_subsequent_run(tmp_path: Path, sample_jsonl: Path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "run_cached"
    cfg = {
        "input": str(sample_jsonl),
        "output_dir": str(output_dir),
        "model": "fake:stub",
        "baseline_cot_source": "generate",
        "reuse_cached_A_cots": True,
        "cot_cache_dir": str(cache_dir),
    }
    fake = FakeModelClient()
    _run(cfg, fake)
    cache_files = list(cache_dir.glob("*.cot"))
    assert cache_files, "Expected cache file to be written"

    fake_second = FakeModelClient()
    second = _run(cfg, fake_second)
    records = second["results"]
    assert all(r["baseline_cot_used"] == "cache" for r in records)
    for cache_file in cache_files:
        assert cache_file.exists()
