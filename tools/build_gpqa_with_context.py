"""Build GPQA with frozen retrieval context.

This script attaches retrieved evidence passages from external corpora (Wikipedia
DPR and, optionally, other corpora) to each GPQA question and writes the result
as a JSONL file consumable by the chat-cot-mutator pipeline.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer

_FAISS_AVAILABLE = importlib.util.find_spec("faiss") is not None
if _FAISS_AVAILABLE:
    import faiss


LOGGER = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    doc_id: str
    source: str
    title: Optional[str]
    text: str

    def as_json(self) -> Dict[str, str]:
        payload = {
            "doc_id": self.doc_id,
            "source": self.source,
            "text": self.text,
        }
        if self.title:
            payload["title"] = self.title
        return payload


@dataclass
class RetrievalResult:
    documents: List[RetrievedDocument]
    scores: List[float]


class WikiDPRRetriever:
    """Simple dense retriever over the `facebook/wiki_dpr` corpus."""

    def __init__(self, k: int, use_faiss: bool = True) -> None:
        self.k = k
        self.use_faiss = use_faiss and _FAISS_AVAILABLE
        LOGGER.info("Loading wiki_dpr corpus (psg split)…")
        self.dataset: Dataset = load_dataset("facebook/wiki_dpr", "psg", split="train")
        LOGGER.info("wiki_dpr corpus size: %d", len(self.dataset))
        LOGGER.info("Stacking embeddings…")
        self.embeddings = np.vstack(self.dataset["emb"]).astype("float32")
        self.embeddings = self._normalize(self.embeddings)
        self.index = self._build_index(self.embeddings)

    def _build_index(self, embeddings: np.ndarray):
        dim = embeddings.shape[1]
        if self.use_faiss:
            LOGGER.info("Building FAISS index (IndexFlatIP, dim=%d)…", dim)
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            return index
        LOGGER.info("FAISS unavailable; falling back to brute-force cosine similarity.")
        return embeddings

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    def retrieve(self, query_vec: np.ndarray) -> RetrievalResult:
        query_vec = query_vec.astype("float32")
        query_vec = self._normalize(query_vec[None, :])
        if self.use_faiss:
            scores, indices = self.index.search(query_vec, self.k)
            scores = scores[0].tolist()
            indices = indices[0].tolist()
        else:
            scores_np = query_vec @ self.index.T
            scores = scores_np[0].tolist()
            indices = np.argpartition(-scores, kth=min(self.k, len(scores) - 1))[: self.k]
            sorted_pairs = sorted(zip(indices, [scores[i] for i in indices]), key=lambda p: -p[1])
            indices, scores = zip(*sorted_pairs)
            indices = list(indices)
            scores = list(scores)
        documents = [self._format_document(idx) for idx in indices]
        return RetrievalResult(documents=documents, scores=scores)

    def _format_document(self, idx: int) -> RetrievedDocument:
        record = self.dataset[int(idx)]
        return RetrievedDocument(
            doc_id=f"wiki_dpr:{idx}",
            source="wiki_dpr",
            title=record.get("title"),
            text=record.get("text", ""),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach evidence to GPQA examples.")
    parser.add_argument("--split", choices=["diamond", "main"], default="diamond", help="GPQA split to process")
    parser.add_argument("--k", type=int, default=8, help="Total number of evidence passages to retrieve")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N questions (for debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use-faiss", dest="use_faiss", action="store_true", help="Use FAISS for retrieval if available")
    parser.add_argument(
        "--no-faiss",
        dest="use_faiss",
        action="store_false",
        help="Disable FAISS even if installed",
    )
    parser.set_defaults(use_faiss=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def load_gpqa(split: str, limit: Optional[int] = None) -> Dataset:
    LOGGER.info("Loading GPQA split '%s'…", split)
    dataset = load_dataset("Idavidrein/gpqa", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    LOGGER.info("Loaded %d GPQA examples", len(dataset))
    return dataset


def get_choice_mapping(example: Dict[str, str]) -> Dict[str, str]:
    if "options" in example and isinstance(example["options"], (dict, list)):
        if isinstance(example["options"], dict):
            options = example["options"]
        else:
            letters = ["A", "B", "C", "D"]
            options = {letters[i]: text for i, text in enumerate(example["options"]) if i < len(letters)}
    else:
        options = {letter: example[letter] for letter in ["A", "B", "C", "D"] if letter in example}
    return options


def build_query_text(question: str, choices: Dict[str, str]) -> str:
    formatted_choices = " ".join([f"({key}) {value}" for key, value in sorted(choices.items())])
    return f"Q: {question} Options: {formatted_choices}".strip()


def embed_queries(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    embeddings = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def format_sample_id(split: str, index: int, example: Dict[str, str]) -> str:
    if "id" in example:
        return f"gpqa-{split}-{example['id']}"
    return f"gpqa-{split}-{index:05d}"


def process_dataset(
    dataset: Dataset,
    retriever: WikiDPRRetriever,
    model: SentenceTransformer,
    split: str,
    k: int,
) -> Iterable[str]:
    queries: List[str] = []
    choice_mappings: List[Dict[str, str]] = []
    for example in dataset:
        choices = get_choice_mapping(example)
        query_text = build_query_text(example["question"], choices)
        queries.append(query_text)
        choice_mappings.append(choices)

    LOGGER.info("Encoding %d queries…", len(queries))
    query_embeddings = embed_queries(model, queries)

    for idx, (example, choices, query_vec) in enumerate(zip(dataset, choice_mappings, query_embeddings)):
        retrieval = retriever.retrieve(query_vec)
        sample = {
            "sample_id": format_sample_id(split, idx, example),
            "dataset": f"gpqa_{split}",
            "question": example["question"],
            "choices": choices,
            "correct_choice": example.get("answer", example.get("correct_choice")),
            "evidence": [doc.as_json() for doc in retrieval.documents[:k]],
        }
        yield json.dumps(sample, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    set_seed(args.seed)

    output_path = args.output or os.path.join("data", f"gpqa_{args.split}_with_context.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gpqa_dataset = load_gpqa(args.split, args.limit)
    retriever = WikiDPRRetriever(k=args.k, use_faiss=args.use_faiss)
    LOGGER.info("Loading query encoder (facebook-dpr-question_encoder-single-nq-base)…")
    query_model = SentenceTransformer("facebook-dpr-question_encoder-single-nq-base")

    LOGGER.info("Writing output to %s", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in process_dataset(gpqa_dataset, retriever, query_model, args.split, args.k):
            f.write(line + "\n")
    LOGGER.info("Finished writing %s", output_path)


if __name__ == "__main__":
    main()
