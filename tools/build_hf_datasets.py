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

try:
    from tools.fever_evidence_retriever import (
        FEVERExactEvidenceRetriever,
        FEVERDistractorRetriever,
    )
    _FEVER_RETRIEVER_AVAILABLE = True
except ImportError:
    _FEVER_RETRIEVER_AVAILABLE = False


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


def _normalize_fever_label(label: str) -> str:
    """Normalize FEVER labels to uppercase."""
    if not label:
        return ""
    label = label.strip().upper()
    # Map common variations
    if label in ["SUPPORTS", "SUPPORT"]:
        return "SUPPORTS"
    elif label in ["REFUTES", "REFUTE"]:
        return "REFUTES"
    elif label in ["NOT ENOUGH INFO", "NOT_ENOUGH_INFO", "NEI"]:
        return "NOT ENOUGH INFO"
    return label


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


def build_fever_samples(
    split: str = "train",
    evidence_lookup: Optional[Dict[str, str]] = None,
    local_file: Optional[Path] = None,
    evidence_retriever = None,
    retrieval_variant: str = "none",
    k_total: int = 8,
    limit: Optional[int] = None,
) -> Iterable[SampleRecord]:
    """
    Yield SampleRecord instances converted from FEVER (Fact Extraction and VERification).
    
    Args:
        split: Dataset split to use (only relevant if loading from HuggingFace, not from local file)
        evidence_lookup: Optional dict mapping sample_id -> retrieved evidence text.
                        If provided, this evidence will be used instead of empty passages.
                        This allows integration with external dense retrieval systems.
        local_file: Path to local FEVER JSONL file. If provided, loads from local file instead of HuggingFace.
        evidence_retriever: FEVERExactEvidenceRetriever or FEVERDistractorRetriever instance
        retrieval_variant: 'exact' for golden only, 'distractor' for golden+distractors, 'none' for placeholder
        k_total: Total number of documents to retrieve (for distractor variant)
        limit: Maximum number of samples to process (stops early if provided)
    
    Returns:
        Iterator of SampleRecord instances
    """
    # Try loading from local file first
    if local_file and local_file.exists():
        LOGGER.info(f"Loading FEVER dataset from local file: {local_file}")
        try:
            samples_yielded = 0
            with local_file.open("r", encoding="utf-8") as f:
                dataset = []
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        dataset.append(obj)
                    except json.JSONDecodeError as e:
                        LOGGER.debug(f"Line {line_num}: Failed to parse JSON: {e}")
                        continue
            LOGGER.info(f"Successfully loaded {len(dataset)} FEVER samples from local file")
        except Exception as e:
            LOGGER.error(f"Could not load FEVER dataset from local file: {e}")
            return
    else:
        # Fallback to HuggingFace (will likely fail with current datasets)
        try:
            # Use the tomhosking/fever-nli dataset which is in standard format
            # This contains the FEVER claims with labels
            dataset = load_dataset("tomhosking/fever-nli", split=split)
            LOGGER.info(f"Successfully loaded FEVER dataset ({len(dataset)} rows)")
        except Exception as e:
            LOGGER.error(f"Could not load FEVER dataset: {e}")
            LOGGER.info("Trying alternative dataset source...")
            try:
                # Fallback: try loading from fever shared task format
                dataset = load_dataset("fever", "v1.0", split=split)
                LOGGER.info(f"Successfully loaded FEVER dataset from alternative source ({len(dataset)} rows)")
            except Exception as e2:
                LOGGER.error(f"All FEVER dataset sources failed: {e2}")
                return

    samples_count = 0
    skipped_count = 0

    for idx, row in enumerate(dataset):
        # Extract fields - handle both dict (local file) and HF dataset formats
        if isinstance(row, dict):
            claim = (row.get("claim") or "").strip()
            label_raw = row.get("label")
            sample_id = row.get("id")
        else:
            claim = (getattr(row, "claim", None) or "").strip()
            label_raw = getattr(row, "label", None)
            sample_id = getattr(row, "id", None)
            
        if not claim:
            skipped_count += 1
            continue
        
        # Get label - FEVER uses numeric labels in some versions, string in others
        label_raw = row.get("label")
        if isinstance(label_raw, int):
            # Map numeric to string: 0=SUPPORTS, 1=REFUTES, 2=NOT ENOUGH INFO
            label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
            label = label_map.get(label_raw, "")
        else:
            label = str(label_raw or "").strip()
        
        # Normalize label
        label = _normalize_fever_label(label)
        if not label or label not in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
            skipped_count += 1
            if idx < 5:  # Log first few failures
                LOGGER.debug(f"Row {idx}: Invalid label '{label_raw}'")
            continue
        
        # Skip "NOT ENOUGH INFO" samples - not relevant for evidence-based experiments
        if label == "NOT ENOUGH INFO":
            skipped_count += 1
            continue
        
        # Get sample ID - preserve as string for alignment with external retrieval
        sample_id = row.get("id")
        if sample_id is None:
            sample_id = f"{idx:06d}"
        sample_id = str(sample_id)  # Ensure string type
        
        if not sample_id.startswith("fever-"):
            sample_id = f"fever-{sample_id}"
        
        # Handle evidence - use retriever if available, otherwise fallback to lookup or placeholder
        # Following HotPotQA pattern: evidence appears in both passages AND tool_outputs
        passages = []
        tool_outputs = []
        evidence_docs = []
        
        # Try retriever first (if configured)
        if evidence_retriever is not None and retrieval_variant != "none":
            # Get evidence annotation from raw data
            evidence_annotation = row.get("evidence", []) if isinstance(row, dict) else getattr(row, "evidence", [])
            
            if retrieval_variant == "exact":
                evidence_docs = evidence_retriever.retrieve_golden_evidence(evidence_annotation)
            elif retrieval_variant == "distractor":
                evidence_docs = evidence_retriever.retrieve_with_distractors(
                    claim, evidence_annotation, k_total=k_total
                )
            
            # Convert evidence documents to passages and tool outputs
            for doc in evidence_docs:
                passages.append(
                    FrozenPassageRecord(
                        doc_id=doc.doc_id,
                        cite=f"{doc.title} (sent {doc.sentence_id})" if doc.sentence_id is not None else doc.title,
                        text=doc.text,
                    )
                )
            
            # Add all evidence to tool_outputs as a single retrieval result
            if evidence_docs:
                combined_evidence = "\n\n".join([
                    f"[{i+1}] {doc.title}: {doc.text}"
                    for i, doc in enumerate(evidence_docs)
                ])
                tool_outputs.append(
                    FrozenToolRecord(
                        tool="dense_retrieval",
                        input=claim,
                        output=combined_evidence,
                    )
                )
        
        # Fallback to evidence_lookup if no retriever
        elif evidence_lookup and sample_id in evidence_lookup:
            # Use externally retrieved evidence
            evidence_text = evidence_lookup[sample_id]
            if evidence_text and evidence_text.strip():
                passages.append(
                    FrozenPassageRecord(
                        doc_id=f"fever:{sample_id}",
                        cite="Retrieved Evidence",
                        text=evidence_text.strip(),
                    )
                )
                tool_outputs.append(
                    FrozenToolRecord(
                        tool="dense_retrieval",
                        input=claim,
                        output=evidence_text.strip(),
                    )
                )
        
        # Final fallback: placeholder
        if not passages:
            placeholder_text = "[Evidence will be retrieved externally based on sample_id]"
            passages.append(
                FrozenPassageRecord(
                    doc_id=f"fever:{sample_id}",
                    cite="Evidence Placeholder",
                    text=placeholder_text,
                )
            )
            tool_outputs.append(
                FrozenToolRecord(
                    tool="dense_retrieval",
                    input=claim,
                    output=placeholder_text,
                )
            )
        
        context = FrozenContextRecord(
            passages=passages,
            tool_outputs=tool_outputs,
        )
        
        # Format the query to include the claim and instructions
        query = f"""Claim: {claim}

Based on the evidence provided, determine if the claim is:
- SUPPORTS: The evidence supports the claim
- REFUTES: The evidence refutes the claim
- NOT ENOUGH INFO: The evidence does not provide enough information

Answer with only one of: SUPPORTS, REFUTES, or NOT ENOUGH INFO"""
        
        sample = SampleRecord(
            id=sample_id,
            query=query,
            frozen_context=context,
            cot_baseline=None,
            mutation_directive=None,
            grounding_rule="Base your answer only on the provided evidence. Do not use external knowledge.",
            answer_gold=label,
            meta={
                "dataset": "fever",
                "claim": claim,
                "retrieval_variant": retrieval_variant,
                "num_evidence_docs": len(evidence_docs) if evidence_docs else 0,
                "num_golden_docs": sum(1 for d in evidence_docs if d.is_golden) if evidence_docs else 0,
                "has_external_evidence": bool(evidence_lookup and sample_id in evidence_lookup),
                "simulated_tools": True,
            },
        )
        
        samples_count += 1
        if samples_count % 1000 == 0:
            LOGGER.info(f"  Processed {samples_count} FEVER samples...")
        
        yield sample
        
        # Stop early if limit reached
        if limit and samples_yielded >= limit:
            LOGGER.info(f"Reached limit of {limit} samples, stopping early")
            break
        samples_yielded += 1

    LOGGER.info(f"Extracted {samples_count} FEVER samples (skipped {skipped_count})")


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
        choices=["webgpt", "hotpot", "gsm8k", "fever", "all"],
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
        "--evidence-file",
        type=Path,
        default=None,
        help="Optional JSONL file mapping sample_id to evidence text (for FEVER)",
    )
    parser.add_argument(
        "--fever-file",
        type=Path,
        default=None,
        help="Optional local FEVER JSONL file (for FEVER dataset)",
    )
    parser.add_argument(
        "--fever-retrieval",
        choices=["none", "exact", "distractor"],
        default="none",
        help="FEVER evidence retrieval variant: 'none' (placeholder), 'exact' (golden only), 'distractor' (golden+distractors)",
    )
    parser.add_argument(
        "--fever-k",
        type=int,
        default=8,
        help="Total number of evidence documents for FEVER distractor retrieval (default: 8)",
    )
    parser.add_argument(
        "--fever-wiki-pages",
        type=Path,
        default=Path("data/fever_raw/wiki-pages/wiki-pages"),
        help="Path to FEVER wiki-pages directory (default: data/fever_raw/wiki-pages/wiki-pages)",
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
    
    # Load evidence lookup if provided (for FEVER)
    evidence_lookup = None
    if args.evidence_file and args.evidence_file.exists():
        LOGGER.info(f"Loading evidence from {args.evidence_file}")
        evidence_lookup = {}
        with args.evidence_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        sample_id = obj.get("id") or obj.get("sample_id")
                        evidence = obj.get("evidence") or obj.get("text")
                        if sample_id and evidence:
                            # Normalize sample_id format
                            sample_id = str(sample_id)
                            if not sample_id.startswith("fever-"):
                                sample_id = f"fever-{sample_id}"
                            evidence_lookup[sample_id] = evidence
                    except json.JSONDecodeError:
                        continue
        LOGGER.info(f"Loaded evidence for {len(evidence_lookup)} samples")

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

    if args.dataset in {"fever", "all"}:
        split = args.split or "train"  # Default to train since that's the most common split
        output_path = output_dir / f"fever_{suffix}.jsonl"
        LOGGER.info(f"Building FEVER samples from split '{split}'...")
        
        # Use local file if provided, otherwise try to find it in data/fever_raw/
        fever_file = args.fever_file
        if not fever_file:
            # Try default location
            default_fever = Path("data/fever_raw") / f"{split}.jsonl"
            if default_fever.exists():
                fever_file = default_fever
                LOGGER.info(f"Using default FEVER file: {fever_file}")
        
        # Initialize evidence retriever if requested
        evidence_retriever = None
        retrieval_variant = args.fever_retrieval
        
        if retrieval_variant != "none":
            if not _FEVER_RETRIEVER_AVAILABLE:
                LOGGER.error("FEVER retriever not available. Install sentence-transformers and faiss.")
                LOGGER.info("Falling back to 'none' (placeholder) mode.")
                retrieval_variant = "none"
            elif not args.fever_wiki_pages.exists():
                LOGGER.error(f"Wiki-pages directory not found: {args.fever_wiki_pages}")
                LOGGER.info("Falling back to 'none' (placeholder) mode.")
                retrieval_variant = "none"
            else:
                try:
                    if retrieval_variant == "exact":
                        LOGGER.info("Initializing exact evidence retriever...")
                        evidence_retriever = FEVERExactEvidenceRetriever(args.fever_wiki_pages)
                    elif retrieval_variant == "distractor":
                        LOGGER.info("Initializing distractor retriever (this may take a while)...")
                        
                        # Extract relevant articles from dataset to limit index size
                        LOGGER.info("Extracting article titles from FEVER dataset...")
                        relevant_articles = set()
                        with open(fever_file, encoding='utf-8') as f:
                            for line_idx, line in enumerate(f):
                                if limit and line_idx >= limit * 2:  # Read more to get enough articles
                                    break
                                item = json.loads(line)
                                if item.get('label') == 'NOT ENOUGH INFO':
                                    continue
                                evidence = item.get('evidence', [])
                                for evidence_set in evidence:
                                    for evidence_item in evidence_set:
                                        if len(evidence_item) >= 3:
                                            wiki_title = evidence_item[2]
                                            if wiki_title:
                                                relevant_articles.add(wiki_title)
                        LOGGER.info(f"Found {len(relevant_articles)} unique article titles")
                        
                        evidence_retriever = FEVERDistractorRetriever(
                            args.fever_wiki_pages,
                            limit_to_articles=relevant_articles if len(relevant_articles) > 0 else None
                        )
                    LOGGER.info(f"Retriever initialized successfully (variant: {retrieval_variant})")
                except Exception as e:
                    LOGGER.error(f"Failed to initialize retriever: {e}")
                    LOGGER.info("Falling back to 'none' (placeholder) mode.")
                    retrieval_variant = "none"
                    evidence_retriever = None
        
        build_dataset_file(
            build_fever_samples(
                split=split,
                evidence_lookup=evidence_lookup,
                local_file=fever_file,
                evidence_retriever=evidence_retriever,
                retrieval_variant=retrieval_variant,
                k_total=args.fever_k,
                limit=limit,
            ),
            output_path,
            limit=limit,
            seed=seed,
        )

    LOGGER.info("Dataset generation complete!")
    LOGGER.info(f"Files written to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
