"""Command-line utilities to convert HF datasets into frozen samples."""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Iterable, List, Sequence

from datasets import load_dataset

from core.schema import (
    FrozenContextRecord,
    FrozenPassageRecord,
    SampleRecord,
    save_jsonl,
)


LOGGER = logging.getLogger(__name__)


def _choose_webgpt_side(row: dict) -> int:
    """Return preferred answer side for a WebGPT row."""
    score_0 = row.get("score_0")
    score_1 = row.get("score_1")
    if score_0 is None or score_1 is None:
        return 0
    return 0 if score_0 >= score_1 else 1


def _webgpt_passages(quotes: Sequence[dict]) -> List[FrozenPassageRecord]:
    passages: List[FrozenPassageRecord] = []
    for quote in quotes or []:
        if not isinstance(quote, dict):
            continue
        text = (quote.get("extract") or "").strip()
        if not text:
            continue
        title = quote.get("title") or "Source"
        passages.append(
            FrozenPassageRecord(
                text=text,
                cite=title,
                doc_id=None,
            )
        )
    return passages


def build_webgpt_samples(split: str = "train") -> Iterable[SampleRecord]:
    """Yield SampleRecord instances converted from WebGPT."""
    dataset = load_dataset("openai/webgpt_comparisons", split=split)
    for row in dataset:
        side = _choose_webgpt_side(row)
        quotes = row.get(f"quotes_{side}") or []
        passages = _webgpt_passages(quotes)
        if not passages:
            continue

        question = row.get("question")
        if isinstance(question, dict):
            question_text = (question.get("text") or "").strip()
        else:
            question_text = (question or "").strip()
        if not question_text:
            continue

        source_id = row.get("id") or row.get("_id")
        sample_id = f"webgpt-{source_id}-{side}"
        context = FrozenContextRecord(passages=passages, tool_outputs=[])
        sample = SampleRecord(
            id=sample_id,
            query=question_text,
            frozen_context=context,
            cot_baseline=None,
            mutation_directive=None,
            grounding_rule="If any step conflicts with EVIDENCE, ignore it and use only EVIDENCE.",
            answer_gold=None,
            meta={"dataset": "webgpt", "side": side},
        )
        yield sample


def _hotpot_gold_passages(row: dict) -> List[FrozenPassageRecord]:
    passages: List[FrozenPassageRecord] = []
    context_map = {title: sentences for title, sentences in row.get("context", [])}
    gold_titles = sorted({title for title, _idx in row.get("supporting_facts", [])})
    for title in gold_titles:
        sentences = context_map.get(title) or []
        if not sentences:
            continue
        text = " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip())
        if not text:
            continue
        doc_id = f"en:{title.replace(' ', '_')}"
        passages.append(
            FrozenPassageRecord(
                text=text,
                doc_id=doc_id,
                cite=title,
            )
        )
    return passages


def build_hotpot_samples(split: str = "validation") -> Iterable[SampleRecord]:
    """Yield SampleRecord instances converted from HotpotQA."""
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    for row in dataset:
        question = (row.get("question") or "").strip()
        if not question:
            continue
        answer = (row.get("answer") or "").strip()
        passages = _hotpot_gold_passages(row)
        if not passages:
            continue
        sample_id = f"hotpot-{row.get('id')}"
        context = FrozenContextRecord(passages=passages, tool_outputs=[])
        sample = SampleRecord(
            id=sample_id,
            query=question,
            frozen_context=context,
            cot_baseline=None,
            mutation_directive=None,
            grounding_rule="If any step conflicts with EVIDENCE, ignore it.",
            answer_gold=answer or None,
            meta={"dataset": "hotpot", "num_gold": len(passages)},
        )
        yield sample


def _sample_records(records: Sequence[SampleRecord], limit: int, seed: int) -> List[SampleRecord]:
    if limit <= 0 or limit >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(list(records), k=limit)


def build_dataset_file(
    samples: Iterable[SampleRecord],
    output_path: Path,
    *,
    limit: int,
    seed: int,
) -> None:
    records = list(samples)
    if not records:
        LOGGER.warning("No samples generated for output %%s", output_path)
        return
    subset = _sample_records(records, limit, seed)
    save_jsonl(output_path, subset)
    LOGGER.info("Wrote %%d samples to %%s", len(subset), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=["webgpt", "hotpot", "all"],
        help="Dataset to convert",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional dataset split override",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of samples to include",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where JSONL files will be written",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = args.limit
    seed = args.seed

    if args.dataset in {"webgpt", "all"}:
        split = args.split or "train"
        output_path = output_dir / "webgpt_1k.jsonl"
        build_dataset_file(
            build_webgpt_samples(split=split),
            output_path,
            limit=limit,
            seed=seed,
        )

    if args.dataset in {"hotpot", "all"}:
        split = args.split or "validation"
        output_path = output_dir / "hotpot_distractor_1k.jsonl"
        build_dataset_file(
            build_hotpot_samples(split=split),
            output_path,
            limit=limit,
            seed=seed,
        )


if __name__ == "__main__":
    main()
