"""Command-line utilities to convert HF datasets into frozen samples."""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Any, Optional

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please install datasets: pip install datasets")

from core.schema import (
    FrozenContextRecord,
    FrozenPassageRecord,
    FrozenToolRecord,
    SampleRecord,
    save_jsonl,
)


LOGGER = logging.getLogger(__name__)

# WebGPT direct download URL (fallback when HF fails)
WEBGPT_URL = "https://openaipublic.blob.core.windows.net/webgpt-answer-viewer/comparisons.jsonl"


def _as_string_question(qobj: Any) -> str:
    """Extract question text from various formats."""
    if isinstance(qobj, dict):
        return (qobj.get("full_text")
                or qobj.get("text")
                or qobj.get("question")
                or "").strip()
    if isinstance(qobj, str):
        return qobj.strip()
    return ""


def _to_passages(quotes: list[dict]) -> List[FrozenPassageRecord]:
    """Convert quotes to passage records."""
    out = []
    for q in quotes or []:
        title = q.get("title") or "Source"
        # raw JSONL uses 'extract' (not 'excerpt'); support both
        text = q.get("extract") or q.get("excerpt") or ""
        if isinstance(text, str) and text.strip():
            out.append(FrozenPassageRecord(
                doc_id=None,
                cite=title,
                text=text.strip()
            ))
    return out


def _choose_side_pair(a0: dict, a1: dict) -> int:
    """Choose better side from a pair of answers."""
    s0, s1 = a0.get("score"), a1.get("score")
    if isinstance(s0, (int, float)) and isinstance(s1, (int, float)):
        return 0 if s0 >= s1 else 1
    # fallback: pick the side with more quotes
    return 0 if len(a0.get("quotes") or []) >= len(a1.get("quotes") or []) else 1


def _choose_side_row(row: dict) -> int:
    """Choose better side from HF row format."""
    s0, s1 = row.get("score_0"), row.get("score_1")
    if isinstance(s0, (int, float)) and isinstance(s1, (int, float)):
        return 0 if s0 >= s1 else 1
    # fallback: pick the side with more quotes
    q0, q1 = row.get("quotes_0") or [], row.get("quotes_1") or []
    return 0 if len(q0) >= len(q1) else 1


def _iter_raw_pairs(lines: Iterable[str]) -> Iterable[Tuple[str, List[FrozenPassageRecord], str, int]]:
    """Yield (question, passages, id, side) from raw pair-format JSONL."""
    sample_count = 0
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            LOGGER.debug(f"Line {line_num}: Failed to parse JSON: {e}")
            continue
        
        # Expect a pair [answer0, answer1]; skip otherwise
        if not (isinstance(obj, list) and len(obj) == 2):
            LOGGER.debug(f"Line {line_num}: Not a pair format")
            continue
        
        a0, a1 = obj[0], obj[1]
        q = _as_string_question(a0.get("question"))
        if not q:
            LOGGER.debug(f"Line {line_num}: No question found")
            continue
        
        side = _choose_side_pair(a0, a1)
        chosen = a0 if side == 0 else a1
        passages = _to_passages(chosen.get("quotes") or [])
        
        if not passages:
            LOGGER.debug(f"Line {line_num}: No valid passages")
            continue
        
        # Extract ID from the answer object
        sample_id = chosen.get("id") or chosen.get("_id") or f"{sample_count:06d}"
        sample_count += 1
        
        if sample_count % 1000 == 0:
            LOGGER.info(f"  Processed {sample_count} WebGPT samples...")
        
        yield q, passages, sample_id, side


def _iter_hf_rows(rows: Iterable[Dict[str, Any]]) -> Iterable[Tuple[str, List[FrozenPassageRecord], str, int]]:
    """Yield (question, passages, id, side) from HF-style dict rows with quotes_0/quotes_1."""
    for idx, r in enumerate(rows):
        q = _as_string_question(r.get("question"))
        if not q:
            LOGGER.debug(f"Row {idx}: No question")
            continue
        
        side = _choose_side_row(r)
        quotes = r.get(f"quotes_{side}") or []
        passages = _to_passages(quotes)
        
        if not passages:
            LOGGER.debug(f"Row {idx}: No valid passages")
            continue
        
        sample_id = r.get("id") or r.get("_id") or f"{idx:06d}"
        yield q, passages, sample_id, side


def build_webgpt_from_url(split: str = "train") -> Iterable[SampleRecord]:
    """Build WebGPT samples by downloading directly from OpenAI blob storage."""
    import requests
    
    LOGGER.info("Downloading WebGPT dataset from OpenAI blob storage...")
    try:
        response = requests.get(WEBGPT_URL, stream=True, timeout=60)
        response.raise_for_status()
    except Exception as e:
        LOGGER.error(f"Failed to download WebGPT dataset from URL: {e}")
        return
    
    samples_count = 0
    for q, passages, sample_id, side in _iter_raw_pairs(response.iter_lines(decode_unicode=True)):
        # Convert passages to synthetic tool outputs to simulate browser searches
        tool_outputs = []
        for passage in passages:
            tool_outputs.append(
                FrozenToolRecord(
                    tool="browser_search",
                    input=passage.cite,  # The source URL/title as search query
                    output=passage.text,  # The extracted text as result
                )
            )
        
        sample = SampleRecord(
            id=f"webgpt-{sample_id}-{side}",
            query=q,
            frozen_context=FrozenContextRecord(passages=passages, tool_outputs=tool_outputs),
            cot_baseline=None,
            mutation_directive=None,
            grounding_rule="If any step conflicts with EVIDENCE, ignore it and use only EVIDENCE.",
            answer_gold=None,
            meta={"dataset": "webgpt", "side": side, "simulated_tools": True},
        )
        samples_count += 1
        yield sample
    
    LOGGER.info(f"Successfully loaded {samples_count} WebGPT samples from URL")


def build_webgpt_samples(split: str = "train") -> Iterable[SampleRecord]:
    """Yield SampleRecord instances converted from WebGPT."""
    # Try HuggingFace first (without trust_remote_code which is deprecated)
    try:
        dataset = load_dataset("openai/webgpt_comparisons", split=split)
        LOGGER.info(f"Successfully loaded WebGPT dataset from HuggingFace ({len(dataset)} rows)")
        
        # Use HF format processor
        samples_count = 0
        for q, passages, sample_id, side in _iter_hf_rows(dataset):
            # Convert passages to synthetic tool outputs to simulate browser searches
            tool_outputs = []
            for passage in passages:
                tool_outputs.append(
                    FrozenToolRecord(
                        tool="browser_search",
                        input=passage.cite,  # The source URL/title as search query
                        output=passage.text,  # The extracted text as result
                    )
                )
            
            sample = SampleRecord(
                id=f"webgpt-{sample_id}-{side}",
                query=q,
                frozen_context=FrozenContextRecord(passages=passages, tool_outputs=tool_outputs),
                cot_baseline=None,
                mutation_directive=None,
                grounding_rule="If any step conflicts with EVIDENCE, ignore it and use only EVIDENCE.",
                answer_gold=None,
                meta={"dataset": "webgpt", "side": side, "simulated_tools": True},
            )
            samples_count += 1
            yield sample
        
        LOGGER.info(f"Extracted {samples_count} WebGPT samples from HuggingFace")
        return
        
    except Exception as e:
        LOGGER.warning(f"Failed to load WebGPT from HuggingFace: {e}")
        LOGGER.info("Falling back to direct URL download...")
        yield from build_webgpt_from_url(split)


def _hotpot_gold_passages(row: dict) -> List[FrozenPassageRecord]:
    """Extract gold passages from HotpotQA row."""
    passages: List[FrozenPassageRecord] = []
    context_data = row.get("context", {})
    
    # Build context map - context is now a dict with 'title' and 'sentences' keys
    context_titles = context_data.get("title", [])
    context_sentences = context_data.get("sentences", [])
    
    context_map = {}
    for title, sentences in zip(context_titles, context_sentences):
        if isinstance(sentences, list):
            context_map[title] = sentences
        elif isinstance(sentences, str):
            context_map[title] = [sentences]
    
    # Get supporting facts - now a dict with 'title' and 'sent_id' keys
    supporting_facts = row.get("supporting_facts", {})
    gold_titles = set()
    
    if isinstance(supporting_facts, dict):
        # New format: {'title': ['Title1', 'Title2'], 'sent_id': [0, 1]}
        titles_list = supporting_facts.get("title", [])
        if isinstance(titles_list, list):
            gold_titles = set(titles_list)
    elif isinstance(supporting_facts, list):
        # Old format: [['Title1', 0], ['Title2', 1]]
        for item in supporting_facts:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                gold_titles.add(item[0])
    
    # Build passages from gold titles
    for title in sorted(gold_titles):
        sentences = context_map.get(title, [])
        if not sentences:
            continue
        
        text = " ".join(s.strip() for s in sentences if s and s.strip())
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
    try:
        dataset = load_dataset("hotpot_qa", "distractor", split=split)
        LOGGER.info(f"Successfully loaded HotpotQA dataset ({len(dataset)} rows)")
    except Exception as e:
        LOGGER.error(f"Could not load HotpotQA dataset: {e}")
        return
    
    samples_count = 0
    skipped_count = 0
    
    # Debug first row to understand structure
    if len(dataset) > 0:
        first_row = dataset[0]
        LOGGER.debug(f"First row keys: {list(first_row.keys())}")
        LOGGER.debug(f"First row sample: {first_row}")
    
    for idx, row in enumerate(dataset):
        question = (row.get("question") or "").strip()
        if not question:
            skipped_count += 1
            continue
        
        answer = (row.get("answer") or "").strip()
        passages = _hotpot_gold_passages(row)
        
        if not passages:
            skipped_count += 1
            if idx < 5:  # Log first few failures
                LOGGER.debug(f"Row {idx}: No gold passages found")
                LOGGER.debug(f"  supporting_facts: {row.get('supporting_facts')}")
                context = row.get('context', {})
                if isinstance(context, dict):
                    LOGGER.debug(f"  context titles: {context.get('title', [])[:2]}")
                else:
                    LOGGER.debug(f"  context sample: {context[:2] if context else None}")
            continue
        
        sample_id = row.get("id") or f"{idx:06d}"
        if not sample_id.startswith("hotpot-"):
            sample_id = f"hotpot-{sample_id}"
        
        # Convert passages to synthetic tool outputs to simulate agentic retrieval
        tool_outputs = []
        for idx_p, passage in enumerate(passages):
            tool_outputs.append(
                FrozenToolRecord(
                    tool="wiki_search",
                    input=passage.cite,  # Use the Wikipedia title as the search query
                    output=passage.text,  # The article text as the result
                )
            )
        
        context = FrozenContextRecord(passages=passages, tool_outputs=tool_outputs)
        sample = SampleRecord(
            id=sample_id,
            query=question,
            frozen_context=context,
            cot_baseline=None,
            mutation_directive=None,
            grounding_rule="If any step conflicts with EVIDENCE, ignore it.",
            answer_gold=answer or None,
            meta={"dataset": "hotpot", "num_gold": len(passages), "simulated_tools": True},
        )
        
        samples_count += 1
        if samples_count % 1000 == 0:
            LOGGER.info(f"  Processed {samples_count} HotpotQA samples...")
        
        yield sample

    LOGGER.info(f"Extracted {samples_count} HotpotQA samples (skipped {skipped_count})")


def _parse_gsm8k_answer(raw_answer: str) -> str:
    """
    GSM8K answers look like:
    '... reasoning ... #### 42'
    Return the final answer string after '####'.
    """
    import re

    if not raw_answer:
        return ""
    text = raw_answer.strip()
    # Try to split on the '####' marker
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip()
    # Fallback: last token if format is slightly off
    parts = text.split()
    return parts[-1].strip() if parts else ""


def build_gsm8k_samples(split: str = "train") -> Iterable[SampleRecord]:
    """
    Yield SampleRecord instances converted from GSM8K (openai/gsm8k, 'main' config).
    For each problem, we treat the question itself as the 'evidence' passage.
    """
    try:
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        LOGGER.info(f"Successfully loaded GSM8K dataset ({len(dataset)} rows)")
    except Exception as e:
        LOGGER.error(f"Could not load GSM8K dataset: {e}")
        return

    samples_count = 0
    skipped_count = 0

    for idx, row in enumerate(dataset):
        question = (row.get("question") or "").strip()
        raw_answer = (row.get("answer") or "").strip()
        if not question or not raw_answer:
            skipped_count += 1
            continue

        # Parse final answer from '... #### 42'
        answer_gold = _parse_gsm8k_answer(raw_answer)
        if not answer_gold:
            skipped_count += 1
            continue

        sample_id = row.get("id") or f"{idx:06d}"
        if not str(sample_id).startswith("gsm8k-"):
            sample_id = f"gsm8k-{sample_id}"

        # Treat the problem text as a single evidence passage
        passages = [
            FrozenPassageRecord(
                doc_id=f"gsm8k:{sample_id}",
                cite="Problem",
                text=question,
            )
        ]

        # No real tools here; keep this empty for now
        tool_outputs: List[FrozenToolRecord] = []

        context = FrozenContextRecord(
            passages=passages,
            tool_outputs=tool_outputs,
        )

        sample = SampleRecord(
            id=sample_id,
            query=question,
            frozen_context=context,
            cot_baseline=None,
            mutation_directive=None,  # assigned later by _assign_mutations_to_samples
            grounding_rule="Use only the problem statement and your internal reasoning.",
            answer_gold=answer_gold,
            meta={
                "dataset": "gsm8k",
                "simulated_tools": False,
            },
        )

        samples_count += 1
        if samples_count % 1000 == 0:
            LOGGER.info(f" Processed {samples_count} GSM8K samples...")
        yield sample

    LOGGER.info(f"Extracted {samples_count} GSM8K samples (skipped {skipped_count})")


def _assign_mutations_to_samples(records: List[SampleRecord]) -> None:
    """Assign mutation directives to samples in a round-robin fashion."""
    mutation_directives = [
        "EntitySwap(old=main entity->new=similar entity)",
        "SalienceDrop()",
        "TopicDilution()",
        "Claim-AlignedDeletion()",
        "Paraphrase()",
        "Reorder()",
    ]
    
    for idx, record in enumerate(records):
        # Assign mutation directive in round-robin fashion
        record.mutation_directive = mutation_directives[idx % len(mutation_directives)]


def _sample_records(records: Sequence[SampleRecord], limit: int, seed: int) -> List[SampleRecord]:
    """Sample and prepare records."""
    if limit <= 0 or limit >= len(records):
        sampled = list(records)
    else:
        rng = random.Random(seed)
        sampled = rng.sample(list(records), k=limit)
    
    # Assign mutation directives to the sampled records
    _assign_mutations_to_samples(sampled)
    return sampled


def build_dataset_file(
    samples: Iterable[SampleRecord],
    output_path: Path,
    *,
    limit: int,
    seed: int,
) -> None:
    """Build and save dataset file."""
    records = list(samples)
    LOGGER.info(f"Collected {len(records)} total samples")
    
    if not records:
        LOGGER.warning("No samples generated for output %s", output_path)
        return
    
    subset = _sample_records(records, limit, seed)
    save_jsonl(output_path, subset)
    LOGGER.info("Wrote %d samples to %s", len(subset), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=["webgpt", "hotpot", "gsm8k", "all"],
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = args.limit
    seed = args.seed
    
    # Create appropriate suffix based on limit
    if limit >= 1000:
        suffix = f"{limit // 1000}k"
    else:
        suffix = str(limit)

    if args.dataset in {"webgpt", "all"}:
        split = args.split or "train"
        output_path = output_dir / f"webgpt_{suffix}.jsonl"
        LOGGER.info(f"Building WebGPT samples from split '{split}'...")
        build_dataset_file(
            build_webgpt_samples(split=split),
            output_path,
            limit=limit,
            seed=seed,
        )

    if args.dataset in {"hotpot", "all"}:
        split = args.split or "validation"
        output_path = output_dir / f"hotpot_distractor_{suffix}.jsonl"
        LOGGER.info(f"Building HotpotQA samples from split '{split}'...")
        build_dataset_file(
            build_hotpot_samples(split=split),
            output_path,
            limit=limit,
            seed=seed,
        )

    if args.dataset in {"gsm8k", "all"}:
        split = args.split or "train"  # gsm8k has train/test; we usually use train
        output_path = output_dir / f"gsm8k_{suffix}.jsonl"
        LOGGER.info(f"Building GSM8K samples from split '{split}'...")
        build_dataset_file(
            build_gsm8k_samples(split=split),
            output_path,
            limit=limit,
            seed=seed,
        )

    LOGGER.info("Dataset generation complete!")
    LOGGER.info(f"Files written to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
