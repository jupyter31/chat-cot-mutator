"""Headless batch runner for Chain-of-Thought mutation experiments."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for environments without PyYAML
    yaml = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from clients.replay_tool_client import ReplayToolClient
from core.pipeline import PromptTemplates, run_condition
from core.schema import SampleRecord, load_jsonl
from eval.metrics import (
    compute_metrics_by_mutation,
    compute_overall_metrics,
    token_latency_rows,
)
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
    # very small YAML subset fallback: key: value per line
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

    logger.info("Initializing model client...")
    model_spec = config.get("model") or "openai:gpt-4o"  # Default to openai if not specified
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
    judge_model_spec = config.get("judge_model")  # Can be None
    mutation_policy = config.get("mutation_policy", "pivotal")
    logger.info(f"Settings: temperature={temperature}, seed={seed_value}, mutation_policy={mutation_policy}")

    all_results: List[Dict[str, Any]] = []
    total_runs = len(samples) * len(conditions)
    logger.info("")
    logger.info(f"Starting {total_runs} condition runs ({len(samples)} samples × {len(conditions)} conditions)")
    logger.info("="*60)
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
        for condition in conditions:
            run_count += 1
            logger.info(f"  [{run_count}/{total_runs}] Running condition {condition}...")
            replay_client.reset()
            judge_client = None
            judge_model = None
            if judge_mode == "llm":
                if judge_model_spec:
                    # Use separate model for judge
                    logger.info(f"    • Initializing separate judge model: {judge_model_spec}")
                    judge_client, judge_model = _create_model_client(judge_model_spec)
                else:
                    # Use same model as main inference
                    judge_client = model_client
                    judge_model = resolved_model_name
            result = run_condition(
                sample,
                condition,
                model_client,
                replay_client,
                prompts,
                model_name=resolved_model_name,
                temperature=temperature,
                seed=seed_value,
                mutation_override=mutation_override,
                judge_client=judge_client,
                judge_model=judge_model,
            )
            result["run_seed"] = seed_value
            result["temperature"] = temperature
            result["judge_mode"] = judge_mode
            all_results.append(result)
            logger.info(f"    ✓ Completed condition {condition} (grounded: {result['judge']['is_grounded']})")
            logger.info(f"    ✓ Completed condition {condition} (grounded: {result['judge']['is_grounded']})")

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


if __name__ == "__main__":  # pragma: no cover
    main()
