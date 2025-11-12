from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.schema import (
    FrozenContextRecord,
    FrozenPassageRecord,
    SampleRecord,
    load_jsonl,
    save_jsonl,
)
import pytest


def _build_sample(sample_id: str) -> SampleRecord:
    passage = FrozenPassageRecord(text="Example passage", cite="Doc1")
    context = FrozenContextRecord(passages=[passage], tool_outputs=[])
    return SampleRecord(
        id=sample_id,
        query="What is the answer?",
        frozen_context=context,
        cot_baseline="Initial reasoning",
        mutation_directive="Paraphrase()",
        grounding_rule="Ground in evidence.",
        answer_gold="Answer",
        meta={"dataset": "unit"},
    )


def test_load_save_round_trip(tmp_path):
    sample = _build_sample("sample-1")
    path = tmp_path / "samples.jsonl"
    save_jsonl(path, [sample])
    loaded = load_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].to_dict() == sample.to_dict()


def test_validation_missing_id():
    with pytest.raises(ValueError):
        SampleRecord.from_dict(
            {
                "query": "Missing id",
                "frozen_context": {"passages": [], "tool_outputs": []},
            }
        )
