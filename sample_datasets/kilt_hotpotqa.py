"""Adapter for the KILT-HotpotQA dataset."""
from __future__ import annotations

from typing import Iterable, List

from core.schema import FrozenContextRecord, FrozenPassageRecord, SampleRecord


def build_passages(supporting_facts: Iterable[dict]) -> List[FrozenPassageRecord]:
    passages: List[FrozenPassageRecord] = []
    for fact in supporting_facts:
        text = fact.get("text") or fact.get("sentence") or ""
        doc_id = fact.get("wikipedia_title") or fact.get("title")
        cite = doc_id
        passages.append(FrozenPassageRecord(text=text, doc_id=doc_id, cite=cite))
    return passages


def convert_row(row: dict, *, sample_id: str) -> SampleRecord:
    """Convert a KILT-HotpotQA item into the schema."""
    evidences = row.get("provenance") or row.get("evidence") or []
    passages = build_passages(evidences)
    context = FrozenContextRecord(passages=passages, tool_outputs=[])
    return SampleRecord(
        id=sample_id,
        query=row.get("question", ""),
        frozen_context=context,
        cot_baseline=None,
        mutation_directive=None,
        grounding_rule="If any step conflicts with EVIDENCE, ignore it.",
        answer_gold=row.get("answer"),
        meta={"dataset": "kilt_hotpotqa", "source_id": row.get("id")},
    )


__all__ = ["convert_row", "build_passages"]
