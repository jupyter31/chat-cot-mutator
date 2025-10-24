"""Data schema and helpers for frozen samples."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict, Union


class FrozenPassage(TypedDict):
    doc_id: Optional[str]
    text: str
    cite: Optional[str]


class FrozenTool(TypedDict):
    tool: str
    input: Union[str, Dict[str, Any]]
    output: Union[str, Dict[str, Any]]


class FrozenContext(TypedDict):
    passages: List[FrozenPassage]
    tool_outputs: List[FrozenTool]


class Sample(TypedDict):
    id: str
    query: str
    frozen_context: FrozenContext
    cot_baseline: Optional[str]
    mutation_directive: Optional[str]
    grounding_rule: Optional[str]
    answer_gold: Optional[str]
    meta: Optional[Dict[str, Any]]


@dataclass
class FrozenPassageRecord:
    text: str
    doc_id: Optional[str] = None
    cite: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrozenPassageRecord":
        if not isinstance(data, dict):
            raise TypeError("FrozenPassage must be a dict")
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("FrozenPassage.text must be a non-empty string")
        doc_id = data.get("doc_id")
        if doc_id is not None and not isinstance(doc_id, str):
            raise ValueError("FrozenPassage.doc_id must be a string or None")
        cite = data.get("cite")
        if cite is not None and not isinstance(cite, str):
            raise ValueError("FrozenPassage.cite must be a string or None")
        return cls(text=text, doc_id=doc_id, cite=cite)

    def to_dict(self) -> FrozenPassage:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "cite": self.cite,
        }


@dataclass
class FrozenToolRecord:
    tool: str
    input: Union[str, Dict[str, Any]]
    output: Union[str, Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrozenToolRecord":
        if not isinstance(data, dict):
            raise TypeError("FrozenTool must be a dict")
        tool = data.get("tool")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError("FrozenTool.tool must be a non-empty string")
        if "input" not in data or "output" not in data:
            raise ValueError("FrozenTool requires 'input' and 'output' fields")
        input_value = data["input"]
        output_value = data["output"]
        if not isinstance(input_value, (str, dict)):
            raise ValueError("FrozenTool.input must be a string or dict")
        if not isinstance(output_value, (str, dict)):
            raise ValueError("FrozenTool.output must be a string or dict")
        return cls(tool=tool, input=input_value, output=output_value)

    def to_dict(self) -> FrozenTool:
        return {
            "tool": self.tool,
            "input": self.input,
            "output": self.output,
        }


@dataclass
class FrozenContextRecord:
    passages: List[FrozenPassageRecord] = field(default_factory=list)
    tool_outputs: List[FrozenToolRecord] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrozenContextRecord":
        if not isinstance(data, dict):
            raise TypeError("FrozenContext must be a dict")
        passages_raw = data.get("passages", [])
        if not isinstance(passages_raw, list):
            raise ValueError("FrozenContext.passages must be a list")
        passages = [FrozenPassageRecord.from_dict(p) for p in passages_raw]
        tool_outputs_raw = data.get("tool_outputs", [])
        if not isinstance(tool_outputs_raw, list):
            raise ValueError("FrozenContext.tool_outputs must be a list")
        tool_outputs = [FrozenToolRecord.from_dict(t) for t in tool_outputs_raw]
        return cls(passages=passages, tool_outputs=tool_outputs)

    def to_dict(self) -> FrozenContext:
        return {
            "passages": [p.to_dict() for p in self.passages],
            "tool_outputs": [t.to_dict() for t in self.tool_outputs],
        }


@dataclass
class SampleRecord:
    id: str
    query: str
    frozen_context: FrozenContextRecord
    cot_baseline: Optional[str] = None
    mutation_directive: Optional[str] = None
    grounding_rule: Optional[str] = None
    answer_gold: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleRecord":
        if not isinstance(data, dict):
            raise TypeError("Sample must be a dict")
        id_value = data.get("id")
        query = data.get("query")
        if not isinstance(id_value, str) or not id_value.strip():
            raise ValueError("Sample.id must be a non-empty string")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Sample.query must be a non-empty string")
        frozen_context = FrozenContextRecord.from_dict(data.get("frozen_context", {}))
        cot_baseline = data.get("cot_baseline")
        mutation_directive = data.get("mutation_directive")
        grounding_rule = data.get("grounding_rule")
        answer_gold = data.get("answer_gold")
        meta = data.get("meta")

        for field_name, value in [
            ("cot_baseline", cot_baseline),
            ("mutation_directive", mutation_directive),
            ("grounding_rule", grounding_rule),
            ("answer_gold", answer_gold),
        ]:
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Sample.{field_name} must be a string or None")

        if meta is not None and not isinstance(meta, dict):
            raise ValueError("Sample.meta must be a dict or None")

        return cls(
            id=id_value,
            query=query,
            frozen_context=frozen_context,
            cot_baseline=cot_baseline,
            mutation_directive=mutation_directive,
            grounding_rule=grounding_rule,
            answer_gold=answer_gold,
            meta=meta,
        )

    def to_dict(self) -> Sample:
        return {
            "id": self.id,
            "query": self.query,
            "frozen_context": self.frozen_context.to_dict(),
            "cot_baseline": self.cot_baseline,
            "mutation_directive": self.mutation_directive,
            "grounding_rule": self.grounding_rule,
            "answer_gold": self.answer_gold,
            "meta": self.meta,
        }


def load_jsonl(path: Union[str, Path]) -> List[SampleRecord]:
    """Load and validate a JSONL file of samples."""
    path = Path(path)
    samples: List[SampleRecord] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            samples.append(SampleRecord.from_dict(payload))
    return samples


def save_jsonl(path: Union[str, Path], samples: Iterable[SampleRecord]) -> None:
    """Persist the provided samples as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            if isinstance(sample, SampleRecord):
                record = sample
            elif isinstance(sample, dict):
                record = SampleRecord.from_dict(sample)
            else:
                raise TypeError("save_jsonl expects SampleRecord or dict instances")
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


__all__ = [
    "FrozenPassage",
    "FrozenTool",
    "FrozenContext",
    "Sample",
    "FrozenPassageRecord",
    "FrozenToolRecord",
    "FrozenContextRecord",
    "SampleRecord",
    "load_jsonl",
    "save_jsonl",
]
