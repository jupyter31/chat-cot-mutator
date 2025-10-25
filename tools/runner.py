"""Headless batch runner for Chain-of-Thought mutation experiments."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for environments without PyYAML
    yaml = None

from clients.replay_tool_client import ReplayToolClient
from core.pipeline import (
    PromptTemplates,
    _extract_text_from_mutation,
    assemble_prompt,
    cache_key_for_A,
    extract_reasoning_block,
    generate_trace_A,
    run_condition,
    try_load_cached_A,
    try_use_sample_A,
)
from core.schema import SampleRecord, load_jsonl
from eval.metrics import (
    compute_metrics_by_mutation,
    compute_overall_metrics,
    token_latency_rows,
)
from mutations.registry import mutate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

PROMPT_CONDITIONS = ["A", "B", "C", "D"]


def _load_prompts(root: Path) -> PromptTemplates:
    templates: Dict[str, str] = {}
    for condition in PROMPT_CONDITIONS:
        prompt_path = root / "prompts" / "conditions" / f"{condition}.txt"
        templates[condition] = prompt_path.read_text(encoding="utf-8")
    return PromptTemplates(templates)


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
        return data or {}
    config: Dict[str, Any] = {}
    for line in text.splitlines():
        if not line or line.strip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        config[key.strip()] = value.strip()
    return config


def _resolve_conditions(value: Optional[str]) -> List[str]:
    if value is None:
        return PROMPT_CONDITIONS
    if isinstance(value, list):
        return [str(v) for v in value]
    return [part.strip() for part in value.split(",") if part.strip()]


def _determine_mutation(sample: SampleRecord, policy: str, idx: int) -> Optional[str]:
    policy = policy or "pivotal"
    if policy == "pivotal":
        return sample.mutation_directive
    if policy == "control":
        options = ["Paraphrase()", "Reorder()"]
        return options[idx % len(options)]
    if policy == "none":
        return None
    raise ValueError(f"Unknown mutation policy: {policy}")


def _create_model_client(model_spec: str):
    provider = model_spec
    model_name = model_spec
    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)

    client = None
    try:
        import client_config

        getter_name = f"get_{provider}_client"
        if hasattr(client_config, getter_name):
            client = getattr(client_config, getter_name)()
    except ImportError:
        client = None

    if client is None:
        from clients.client_factory import create_llm_client

        client = create_llm_client(provider, endpoint=None)
    return client, model_name


@dataclass
class RunnerConfig:
    input_path: Path
    output_dir: Path
    conditions: List[str]
    model_spec: str
    resolved_model_name: str
    temperature: float
    seed_value: Optional[int]
    judge_mode: str
    judge_model_spec: Optional[str]
    mutation_policy: str
    baseline_cot_source: str = "generate"
    reuse_cached_A_cots: bool = True
    cot_cache_dir: Path | None = None
    run_id: str = ""


def _infer_run_id(output_dir: Path) -> str:
    parts = [p for p in output_dir.parts if p]
    return parts[-1] if parts else "run"


def _resolve_cache_dir(raw_value: Optional[str], run_id: str) -> Path:
    if raw_value:
        return Path(raw_value.replace("${RUN}", run_id))
    return Path("results") / run_id / "cot_cache"


def _validate_baseline_source(value: str) -> str:
    allowed = {"generate", "sample", "auto"}
    if value not in allowed:
        raise ValueError(f"baseline_cot_source must be one of {sorted(allowed)}")
    return value


def _infer_mutation_type(directive: Optional[str]) -> str:
    if not directive:
        return "none"
    if re.search(r"paraphrase|reorder", directive or "", re.IGNORECASE):
        return "control"
    return "pivotal"


def _resolve_judge_clients(cfg: RunnerConfig, model_client) -> Tuple[Any, Optional[str]]:
    if cfg.judge_mode != "llm":
        return None, None
    if cfg.judge_model_spec:
        return _create_model_client(cfg.judge_model_spec)
    return model_client, cfg.resolved_model_name


def _cache_file_paths(cfg: RunnerConfig, model_name: str, sample_id: str) -> Tuple[Path, Path]:
    cache_path = cfg.cot_cache_dir / cache_key_for_A(cfg.run_id, model_name, sample_id)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".json")
    return cache_path, meta_path


def _save_to_cache(
    cfg: RunnerConfig, model_name: str, sample_id: str, payload: Dict[str, Any]
) -> None:
    if not cfg.reuse_cached_A_cots or cfg.cot_cache_dir is None:
        return
    cfg.cot_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path, meta_path = _cache_file_paths(cfg, model_name, sample_id)

    trace = payload.get("trace_A") or ""
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(trace, encoding="utf-8")
    tmp_path.replace(cache_path)

    serializable = {
        "trace_A": trace,
        "final_answer_A": payload.get("final_answer_A"),
        "final_answer_text_A": payload.get("final_answer_text_A"),
        "raw_A": payload.get("raw_A"),
        "usage": payload.get("usage"),
        "citations_A": payload.get("citations_A"),
        "judge_A": payload.get("judge_A"),
        "latency_s": payload.get("latency_s"),
        "record": payload.get("record"),
    }
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
    meta_tmp.replace(meta_path)


def _load_cached_a_result(
    cfg: RunnerConfig, model_name: str, sample_id: str
) -> Optional[Dict[str, Any]]:
    if not cfg.reuse_cached_A_cots or cfg.cot_cache_dir is None:
        return None
    cache_path, meta_path = _cache_file_paths(cfg, model_name, sample_id)
    if not meta_path.exists():
        if cache_path.exists():
            trace = cache_path.read_text(encoding="utf-8")
            return {"trace_A": trace or ""}
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if "trace_A" not in data:
        data["trace_A"] = cache_path.read_text(encoding="utf-8") if cache_path.exists() else ""
    return data


_CITE_BLOCK = re.compile(r"\[\[[^\]]+\]\]")


def _extract_final_answer_from_text(text: str) -> str:
    stripped = (text or "").strip()
    for line in reversed(stripped.splitlines()):
        if line.lower().startswith("final answer"):
            return line.split(":", 1)[-1].strip()
    return stripped


def _strip_citations(text: str) -> str:
    return _CITE_BLOCK.sub("", text or "").strip()


def _build_sample_a_result(
    sample: SampleRecord, baseline_text: str, prompts: PromptTemplates
) -> Dict[str, Any]:
    response_text = baseline_text or ""
    reasoning = extract_reasoning_block(response_text) or response_text
    final_answer = _extract_final_answer_from_text(response_text)
    if "Final Answer:" not in response_text:
        final_answer = final_answer or (sample.answer_gold or "")
        formatted = reasoning or response_text
        response_text = f"Reasoning: {formatted}\nFinal Answer: {final_answer}".strip()
    citations = [match.strip() for match in re.findall(r"\[\[([^\]]+)\]\]", response_text)]
    record = {
        "sample_id": sample.id,
        "condition": "A",
        "prompt": assemble_prompt("A", sample, mutated_cot=None, prompts=prompts),
        "response": response_text,
        "final_answer": f"Final Answer: {final_answer}" if "Final Answer:" not in final_answer else final_answer,
        "final_answer_text": _strip_citations(final_answer),
        "citations": citations,
        "judge": {
            "method": "sample_baseline",
            "is_grounded": None,
            "citations_provided": citations,
            "answer_correct": None,
        },
        "latency_s": 0.0,
        "usage": {},
    }
    return {
        "record": record,
        "trace_A": reasoning,
        "final_answer_A": record["final_answer"],
        "final_answer_text_A": record["final_answer_text"],
        "raw_A": {"role": "assistant", "content": response_text},
        "usage": record["usage"],
        "citations_A": citations,
        "judge_A": record["judge"],
        "latency_s": record["latency_s"],
    }


def _append_common_metadata(record: Dict[str, Any], cfg: RunnerConfig) -> None:
    record["run_seed"] = cfg.seed_value
    record["temperature"] = cfg.temperature
    record["judge_mode"] = cfg.judge_mode


def run_sample(
    sample: SampleRecord,
    model_client,
    prompts: PromptTemplates,
    cfg: RunnerConfig,
    replay_client: ReplayToolClient,
    *,
    mutation_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    model_name = cfg.resolved_model_name
    directive = mutation_override if mutation_override is not None else sample.mutation_directive

    baseline_cot: Optional[str] = None
    source: Optional[str] = None
    a_result: Optional[Dict[str, Any]] = None
    if cfg.reuse_cached_A_cots:
        cached, hit = try_load_cached_A(cfg, model_name, sample.id)
        if hit:
            baseline_cot = cached or ""
            source = "cache"
            cached_payload = _load_cached_a_result(cfg, model_name, sample.id)
            if cached_payload:
                a_result = cached_payload
    if baseline_cot is None:
        sample_cot, hit = try_use_sample_A(cfg, sample)
        if hit:
            baseline_cot = sample_cot or ""
            source = "sample"
            a_result = _build_sample_a_result(sample, baseline_cot, prompts)

    judge_client, judge_model = _resolve_judge_clients(cfg, model_client)

    if a_result is None or "record" not in a_result:
        a_result = generate_trace_A(
            sample,
            model_client,
            prompts,
            model_name=model_name,
            temperature=cfg.temperature,
            seed=cfg.seed_value,
            judge_client=judge_client,
            judge_model=judge_model,
        )
        baseline_cot = a_result.get("trace_A") or ""
        source = "generated"
        _save_to_cache(cfg, model_name, sample.id, a_result)
    else:
        if baseline_cot is None:
            baseline_cot = a_result.get("trace_A") or ""

    source = source or "generated"
    a_result["trace_A"] = baseline_cot or ""

    results: List[Dict[str, Any]] = []
    mutated_bundle: Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]] = None

    for condition in cfg.conditions:
        replay_client.reset()
        if condition == "A":
            record = dict(a_result["record"])
            record["trace_A"] = a_result["trace_A"] or ""
            record["baseline_cot_used"] = source
            record["baseline_cot"] = baseline_cot or ""
            record["mutation_type"] = "baseline"
            record["directive"] = directive
            record["raw_response"] = a_result["raw_A"]
            results.append(record)
        elif condition == "B":
            record = run_condition(
                sample,
                "B",
                model_client,
                replay_client,
                prompts,
                model_name=model_name,
                temperature=cfg.temperature,
                seed=cfg.seed_value,
                judge_client=judge_client,
                judge_model=judge_model,
                baseline_cot=None,
                baseline_cot_used=source,
                mutation_type="answer_only",
                directive=directive,
            )
            results.append(record)
        elif condition in {"C", "D"}:
            if mutated_bundle is None:
                meta_and_spec, mutated_text = mutate(
                    directive,
                    baseline_cot or "",
                    model_client=model_client,
                    model_name=model_name,
                    temperature=cfg.temperature,
                    seed=cfg.seed_value,
                )
                # Extract plain text from the mutation result
                mutated_text = _extract_text_from_mutation(mutated_text)
                
                # Unpack the tuple properly
                meta, spec = meta_and_spec
                mutated_bundle = (mutated_text, meta, spec)
            mutated_cot, meta, spec = mutated_bundle
            record = run_condition(
                sample,
                condition,
                model_client,
                replay_client,
                prompts,
                model_name=model_name,
                temperature=cfg.temperature,
                seed=cfg.seed_value,
                judge_client=judge_client,
                judge_model=judge_model,
                baseline_cot=mutated_cot,
                baseline_cot_used=source,
                mutation_meta=(meta, spec),
                mutation_type=_infer_mutation_type(directive),
                directive=directive,
            )
            results.append(record)
        else:
            raise ValueError(f"Unsupported condition: {condition}")

    return results


def run_experiment(config: Dict[str, Any], *, model_client=None) -> Dict[str, Any]:
    logger.info("="*60)
    logger.info("Starting experiment with configuration:")
    logger.info(f"  Input: {config.get('input')}")
    logger.info(f"  Model: {config.get('model')}")
    logger.info(f"  Judge mode: {config.get('judge')}")
    logger.info(f"  Judge model: {config.get('judge_model')}")
    logger.info(f"  Max samples: {config.get('max_samples')}")
    logger.info("="*60)
    
    logger.info("Loading prompt templates...")
    repo_root = Path(__file__).resolve().parent.parent
    prompts = _load_prompts(repo_root)
    logger.info(f"✓ Loaded {len(prompts.condition_to_template)} prompt templates")
    
    logger.info(f"Loading samples from {config['input']}...")
    input_path = Path(config["input"])
    samples = load_jsonl(input_path)
    max_samples = int(config.get("max_samples", 0) or 0)
    if max_samples:
        samples = samples[:max_samples]
        logger.info(f"✓ Loaded {len(samples)} samples (limited to max_samples={max_samples})")
    else:
        logger.info(f"✓ Loaded {len(samples)} samples")
    
    conditions = config.get("conditions")
    if isinstance(conditions, str):
        conditions = _resolve_conditions(conditions)
    elif not conditions:
        conditions = PROMPT_CONDITIONS
    logger.info(f"Running conditions: {', '.join(conditions)}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    model_spec = config.get("model") or "openai:gpt-4o"
    if model_client is None:
        model_client, resolved_model_name = _create_model_client(model_spec)
        logger.info(f"✓ Created model client: {resolved_model_name}")
    else:
        resolved_model_name = model_spec.split(":", 1)[1] if ":" in model_spec else model_spec
        logger.info(f"✓ Using provided model client: {resolved_model_name}")

    temperature = float(config.get("temperature", 0.0))
    seed = config.get("seed")
    seed_value = int(seed) if seed is not None else None
    judge_mode = config.get("judge", "prog")
    judge_model_spec = config.get("judge_model")
    mutation_policy = config.get("mutation_policy", "pivotal")
    logger.info(f"Settings: temperature={temperature}, seed={seed_value}, mutation_policy={mutation_policy}")

    run_id = _infer_run_id(output_dir)
    baseline_source = _validate_baseline_source(config.get("baseline_cot_source", "generate"))
    reuse_cached = bool(config.get("reuse_cached_A_cots", True))
    cache_dir = _resolve_cache_dir(config.get("cot_cache_dir"), run_id)

    cfg = RunnerConfig(
        input_path=input_path,
        output_dir=output_dir,
        conditions=conditions,
        model_spec=model_spec,
        resolved_model_name=resolved_model_name,
        temperature=temperature,
        seed_value=seed_value,
        judge_mode=judge_mode,
        judge_model_spec=judge_model_spec,
        mutation_policy=mutation_policy,
        baseline_cot_source=baseline_source,
        reuse_cached_A_cots=reuse_cached,
        cot_cache_dir=cache_dir,
        run_id=run_id,
    )

    all_results: List[Dict[str, Any]] = []
    total_runs = len(samples) * len(conditions)
    logger.info("")
    logger.info(f"Starting {total_runs} condition runs ({len(samples)} samples × {len(conditions)} conditions)")
    logger.info("="*60)

    run_count = 0
    for idx, sample in enumerate(samples):
        logger.info("")
        logger.info(f"Sample {idx+1}/{len(samples)}: {sample.id}")
        replay_client = ReplayToolClient(sample.frozen_context.tool_outputs)
        mutation_override = _determine_mutation(sample, mutation_policy, idx)
        sample_results = run_sample(
            sample,
            model_client,
            prompts,
            cfg,
            replay_client,
            mutation_override=mutation_override,
        )
        for record in sample_results:
            _append_common_metadata(record, cfg)
            all_results.append(record)

    logger.info("")
    logger.info("="*60)
    logger.info("All condition runs completed. Writing outputs...")
    
    samples_jsonl = output_dir / "samples.jsonl"
    logger.info(f"Writing samples to {samples_jsonl}...")
    with samples_jsonl.open("w", encoding="utf-8") as f:
        for record in all_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"✓ Wrote {len(all_results)} results")

    logger.info("Computing overall metrics...")
    metrics_overall = compute_overall_metrics(all_results)
    (output_dir / "metrics_overall.json").write_text(
        json.dumps(metrics_overall, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"✓ Wrote metrics_overall.json")

    logger.info("Computing metrics by mutation type...")
    metrics_mutation = compute_metrics_by_mutation(all_results)
    (output_dir / "metrics_by_mutation.json").write_text(
        json.dumps(metrics_mutation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"✓ Wrote metrics_by_mutation.json")

    logger.info("Computing token usage and latency stats...")
    token_rows = token_latency_rows(all_results)
    with (output_dir / "tokens_latency.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "sample_id",
                "condition",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "latency_s",
            ],
        )
        writer.writeheader()
        writer.writerows(token_rows)
    logger.info(f"✓ Wrote tokens_latency.csv")
    
    logger.info("="*60)
    logger.info("Experiment completed successfully!")
    logger.info(f"Results written to: {output_dir}")
    logger.info("="*60)

    return {
        "results": all_results,
        "metrics_overall": metrics_overall,
        "metrics_by_mutation": metrics_mutation,
        "output_dir": str(output_dir),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless runner for chat-cot-mutator experiments.")
    parser.add_argument("--config", help="Path to YAML config file", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = _load_config(args.config)
    
    print(f"[Runner] Config file: {args.config}")
    print(f"[Runner] Loaded config keys: {list(config.keys())}")

    # Validate required config keys
    missing = [key for key in ["input", "output_dir"] if not config.get(key)]
    if missing:
        raise SystemExit(f"Missing required configuration values: {', '.join(missing)}")

    run_experiment(config)


if __name__ == "__main__":
    main()
