"""Utilities to convert WebGPT dataset rows into the frozen schema."""
from __future__ import annotations

from typing import Iterable, List, Optional

from core.schema import FrozenContextRecord, FrozenPassageRecord, SampleRecord


def build_passages(quote_texts: Iterable[str], quote_titles: Optional[Iterable[str]] = None) -> List[FrozenPassageRecord]:
    passages: List[FrozenPassageRecord] = []
    titles = list(quote_titles or [])
    for idx, text in enumerate(quote_texts):
        cite = titles[idx] if idx < len(titles) else None
        passages.append(FrozenPassageRecord(text=text, cite=cite))
    return passages


def convert_row(row: dict, *, sample_id: str) -> SampleRecord:
    """Convert a WebGPT dataset row to the shared SampleRecord."""
    quotes = row.get("quotes_text") or row.get("quotes") or []
    titles = row.get("quotes_title") or row.get("quotes_source") or []
    passages = build_passages(quotes, titles)
    context = FrozenContextRecord(passages=passages, tool_outputs=[])
    answer_gold = row.get("best_answer") or row.get("answer")
    return SampleRecord(
        id=sample_id,
        query=row.get("question", ""),
        frozen_context=context,
        cot_baseline=None,
        mutation_directive=None,
        grounding_rule="If any step conflicts with EVIDENCE, ignore it and use only EVIDENCE.",
        answer_gold=answer_gold,
        meta={"dataset": "webgpt", "source_id": row.get("id")},
    )


__all__ = ["convert_row", "build_passages"]
