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
        messages = request["messages"]
        prompt = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                prompt = message.get("content", "")
                if isinstance(prompt, list):
                    parts = []
                    for item in prompt:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        elif isinstance(item, str):
                            parts.append(item)
                    prompt = "".join(parts)
                break

        has_internal_steps = any(m.get("name") == "cot_instructions" for m in messages)

        if has_internal_steps:
            reasoning = None
            content = "Final Answer: mutated [[C2]]"
        elif "Final Answer:" in prompt and "Reasoning:" not in prompt:
            reasoning = None
            content = "Final Answer: answer [[C1]]"
        else:
            reasoning = "baseline step [[C1]]"
            content = "Final Answer: baseline [[C1]]"

        raw = {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return {
            "text": content,
            "usage": raw["usage"],
            "raw": raw,
            "reasoning_text": reasoning,
            "process_tokens": None,
            "flags": {"leak_think": False},
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
    assert a_record["trace_A_source"] == "think_stream"

    # Ensure evidence passages are emitted as tool messages
    a_request_messages = fake.requests[0][1]["messages"]
    tool_messages = [m for m in a_request_messages if m.get("name") == "evidence_passage"]
    assert tool_messages, "Expected tool messages carrying evidence passages"
    for tool_msg in tool_messages:
        payload = json.loads(tool_msg["content"])
        assert payload["type"] in {"passage", "tool_output"}
        assert payload["text"].strip()

    for condition in ("C", "D"):
        record = next(r for r in records if r["condition"] == condition)
        assert record["mutated_cot"].strip() == baseline_trace.strip()
        cot_messages = [m for m in record["messages"] if m.get("name") == "cot_instructions"]
        assert cot_messages, "Expected mutated CoT to be injected as a dedicated message"
        assert cot_messages[0]["content"].strip() == baseline_trace.strip()
    # A + B + C + D (mutation no-op)
    assert len(fake.requests) == 4


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
    assert a_record["trace_A_source"] == "sample"
    assert all(r["baseline_cot_used"] == "sample" for r in records)
    for condition in ("C", "D"):
        record = next(r for r in records if r["condition"] == condition)
        assert record["mutated_cot"].strip() == baseline.strip()
        cot_messages = [m for m in record["messages"] if m.get("name") == "cot_instructions"]
        assert cot_messages
        assert cot_messages[0]["content"].strip() == baseline.strip()
    # B + C + D (mutation no-op, A skipped)
    assert len(fake.requests) == 3


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
    # B + C + D (mutation no-op, A reused)
    assert len(fake_second.requests) == 3
