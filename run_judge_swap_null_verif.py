#!/usr/bin/env python3
# run_judge_swap_null_verif.py: Run judge-swap null experiment using verified proxy sets.
# Writes analyzer-compatible cache JSONLs and only performs vLLM inference for J(K vs R).
# Written by: Dani
# Created: 2026-01-15 07:11 EST
# Last Modified: 2026-01-15 07:25 EST

import argparse
import gc
import getpass
import json
import logging
import os
import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from reproduce_paper_experiments import (
    CONFIG,
    MODEL_FAMILIES,
    build_verdict_prompt,
    extract_abt_probs_from_logprobs,
    extract_cot_verdict_with_logprobs,
    generate_verdict_vllm,
    is_reasoning_model,
    setup_vllm_engine,
)


# ----------------------
# --- FILE UTILITIES ---
# ----------------------


def _now_est_str() -> str:
    """Return current timestamp formatted in EST.

    Args:
        None

    Returns:
        str: Timestamp string like YYYY-MM-DD HH:MM EST.

    Raises:
        None
    """
    # Prefer the system tz database if available.
    try:
        old_tz = os.environ.get("TZ")
        os.environ["TZ"] = "America/New_York"
        if hasattr(time, "tzset"):
            time.tzset()
        return datetime.now().strftime("%Y-%m-%d %H:%M EST")
    finally:
        if "old_tz" in locals():
            if old_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = old_tz
            if hasattr(time, "tzset"):
                time.tzset()


def setup_logging() -> logging.Logger:
    """Set up logging to both console and timestamped file.

    Args:
        None

    Returns:
        logging.Logger: Configured logger.

    Raises:
        None
    """
    log_dir = Path("file_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_judge_swap_null_verif_{ts}.log"

    logger = logging.getLogger("judge_swap_null_verif")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("RUN_JUDGE_SWAP_NULL_VERIF START")
    logger.info("=" * 80)
    logger.info(f"Timestamp (EST approx): {_now_est_str()}")
    logger.info(f"User: {getpass.getuser()}")

    try:
        cuda_available = torch.cuda.is_available()
        n_gpus = torch.cuda.device_count() if cuda_available else 0
        logger.info(f"CUDA available: {cuda_available} | num_gpus: {n_gpus}")
        if cuda_available and n_gpus > 0:
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU[{i}]: name={props.name} | total_mem_gb={props.total_memory/1e9:.2f}")
    except Exception as e:
        logger.warning(f"Failed to query CUDA info: {e}")

    logger.info(f"CWD: {os.getcwd()}")
    logger.info(f"Log file: {log_path}")
    return logger


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file.

    Args:
        path (Path): File path.

    Returns:
        Dict[str, Any]: Parsed JSON.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file cannot be parsed.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_proxy_json_files(verif_dir: Path, datasets: Optional[Sequence[str]] = None) -> Iterable[Tuple[str, str, str, Path]]:
    """Yield proxy json files under a verification directory.

    Expected structure:
      verif_dir/<dataset>/<family>/proxies/<judge>.json

    Args:
        verif_dir (Path): Base verification directory.
        datasets (Optional[Sequence[str]]): If provided, only these datasets.

    Yields:
        Tuple[str, str, str, Path]: (dataset, family, judge_short, proxy_json_path)

    Raises:
        None
    """
    if datasets is None:
        dataset_dirs = [p for p in verif_dir.iterdir() if p.is_dir()]
    else:
        dataset_dirs = [verif_dir / d for d in datasets]

    for dataset_dir in sorted(dataset_dirs, key=lambda p: p.name):
        if not dataset_dir.exists():
            continue
        dataset = dataset_dir.name
        for family_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            family = family_dir.name
            proxies_dir = family_dir / "proxies"
            if not proxies_dir.exists():
                continue
            for proxy_json in sorted(list(proxies_dir.glob("*.json")), key=lambda p: p.name):
                judge_short = proxy_json.stem
                yield dataset, family, judge_short, proxy_json


def build_short_to_hf_id_map() -> Dict[str, str]:
    """Build mapping from short model names to HF model IDs from paper config.

    Args:
        None

    Returns:
        Dict[str, str]: Mapping like {"qwen-2.5-3b": "Qwen/Qwen2.5-3B-Instruct", ...}

    Raises:
        None
    """
    out: Dict[str, str] = {}

    def _norm_short(s: str) -> str:
        return s.strip().lower()

    for family, model_ids in MODEL_FAMILIES.items():
        for mid in model_ids:
            # Llama
            if mid.startswith("meta-llama/") and "Llama-" in mid:
                m = re.search(r"Llama-(\d+(?:\.\d+)?)-(\d+)B", mid)
                if m:
                    ver = m.group(1)
                    size = m.group(2).lower()
                    out[_norm_short(f"llama-{ver}-{size}b")] = mid
                    continue

            # Gemma
            if mid.startswith("google/") and "gemma-" in mid:
                m = re.search(r"gemma-2-(\d+)b", mid.lower())
                if m:
                    out[_norm_short(f"gemma-2-{m.group(1)}b")] = mid
                    continue

            # Qwen2.5
            if mid.lower().startswith("qwen/") and "qwen2.5-" in mid.lower():
                m = re.search(r"qwen2\.5-(\d+)b", mid.lower())
                if m:
                    out[_norm_short(f"qwen-2.5-{m.group(1)}b")] = mid
                    continue

            # DeepSeek
            if mid.lower().startswith("deepseek-ai/") and "deepseek-r1" in mid.lower():
                mm = re.search(r"distill-(llama|qwen)-(\d+)b", mid.lower())
                if mm:
                    fam = mm.group(1)
                    size = mm.group(2)
                    out[_norm_short(f"deepseek-r1-distill-{fam}-{size}b")] = mid
                    continue

            # Mistral
            if mid.lower().startswith("mistralai/") and "mistral-7b-instruct" in mid.lower():
                out[_norm_short("mistral-7b-v0.3")] = mid
                continue
            if mid.lower().startswith("mistralai/") and "mistral-small" in mid.lower():
                out[_norm_short("mistral-small")] = mid
                continue

            # Phi
            if mid.lower().startswith("microsoft/") and "phi-3.5-mini" in mid.lower():
                out[_norm_short("phi-3.5-mini")] = mid
                continue

    return out


def model_size_key(model_short: str) -> Tuple[float, str]:
    """Sort key for model names by parameter count (ascending), fallback by name.

    Args:
        model_short (str): Short model name.

    Returns:
        Tuple[float, str]: (size_in_b, model_short)

    Raises:
        None
    """
    s = model_short.lower()
    m = re.search(r"(\d+(?:\.\d+)?)b", s)
    if m:
        try:
            return float(m.group(1)), model_short
        except Exception:
            pass
    # Put unknown-size models at the end.
    return float("inf"), model_short


def read_existing_cache_keys(path: Path) -> Set[str]:
    """Read cache_key values from an existing JSONL cache file.

    Args:
        path (Path): JSONL path.

    Returns:
        Set[str]: Existing cache keys.

    Raises:
        None
    """
    if not path.exists():
        return set()

    keys: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ck = obj.get("cache_key")
                if ck is not None:
                    keys.add(str(ck))
            except Exception:
                continue
    return keys


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Append rows to a JSONL file.

    Args:
        path (Path): JSONL path.
        rows (List[Dict[str, Any]]): Rows to append.

    Returns:
        None

    Raises:
        OSError: If writing fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def chunked(seq: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    """Yield chunks of size n.

    Args:
        seq (Sequence[Any]): Sequence.
        n (int): Chunk size.

    Yields:
        Sequence[Any]: Chunk.

    Raises:
        ValueError: If n <= 0.
    """
    if n <= 0:
        raise ValueError("chunk size must be > 0")
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# ---------------------
# --- CORE LOGIC    ---
# ---------------------


def write_j_vs_r_cache(
    out_dir: Path,
    dataset: str,
    judge: str,
    reference: str,
    j_rows_by_id: Dict[int, Dict[str, Any]],
    overwrite: bool,
    logger: logging.Logger,
) -> None:
    """Write J_vs_R_game1/game2 cache JSONLs from existing verified probabilities.

    Args:
        out_dir (Path): Base output directory.
        dataset (str): Dataset name.
        judge (str): Judge short name.
        reference (str): Reference short name.
        j_rows_by_id (Dict[int, Dict[str, Any]]): Map to_eval_id -> J_v_R row (verified row).
        overwrite (bool): Whether to overwrite existing cache files.
        logger (logging.Logger): Logger.

    Returns:
        None

    Raises:
        KeyError: If required fields are missing in a row.
    """
    cache_base = out_dir / dataset / "cache" / judge / reference
    g1_path = cache_base / "J_vs_R_game1.jsonl"
    g2_path = cache_base / "J_vs_R_game2.jsonl"

    if overwrite:
        if g1_path.exists():
            g1_path.unlink()
        if g2_path.exists():
            g2_path.unlink()

    existing_g1 = read_existing_cache_keys(g1_path)
    existing_g2 = read_existing_cache_keys(g2_path)

    to_write_g1: List[Dict[str, Any]] = []
    to_write_g2: List[Dict[str, Any]] = []

    for eid, row in sorted(j_rows_by_id.items(), key=lambda x: x[0]):
        cache_key = str(eid)
        category = row.get("gold_label") if row.get("gold_label") in ("lsp", "ilsp") else "unknown"

        g1 = row["game_1"]
        probs1 = g1["generated_probs"]
        raw1 = [float(probs1["A"]), float(probs1["B"]), float(probs1["T"])]
        if cache_key not in existing_g1:
            to_write_g1.append(
                {
                    "cache_key": cache_key,
                    "example_id": int(eid),
                    "judge": judge,
                    "reference": reference,
                    "comparison_type": "J_vs_R_game1",
                    "response1_key": "judge",
                    "response2_key": "reference",
                    "prob_response1": raw1[0],
                    "prob_response2": raw1[1],
                    "generated_text": g1.get("generated_text", ""),
                    "raw_probs": raw1,
                    "normalized_sum": float(sum(raw1)),
                    "category": category,
                    "dataset": dataset,
                }
            )

        g2 = row["game_2"]
        probs2 = g2["generated_probs"]
        raw2 = [float(probs2["A"]), float(probs2["B"]), float(probs2["T"])]
        if cache_key not in existing_g2:
            to_write_g2.append(
                {
                    "cache_key": cache_key,
                    "example_id": int(eid),
                    "judge": judge,
                    "reference": reference,
                    "comparison_type": "J_vs_R_game2",
                    "response1_key": "reference",
                    "response2_key": "judge",
                    "prob_response1": raw2[0],
                    "prob_response2": raw2[1],
                    "generated_text": g2.get("generated_text", ""),
                    "raw_probs": raw2,
                    "normalized_sum": float(sum(raw2)),
                    "category": category,
                    "dataset": dataset,
                }
            )

    if to_write_g1:
        append_jsonl(g1_path, to_write_g1)
    if to_write_g2:
        append_jsonl(g2_path, to_write_g2)

    logger.info(
        f"Wrote/kept J_vs_R cache for dataset={dataset} judge={judge} ref={reference}: "
        f"game1 +{len(to_write_g1)} (total_known={len(j_rows_by_id)}), "
        f"game2 +{len(to_write_g2)}"
    )


def infer_k_vs_r_cache(
    llm,
    tokenizer,
    out_dir: Path,
    dataset: str,
    judge: str,
    reference: str,
    proxy: str,
    examples: List[Dict[str, Any]],
    reasoning_mode: str,
    temperature: float,
    max_tokens: int,
    config: Dict[str, Any],
    batch_size: int,
    overwrite: bool,
    logger: logging.Logger,
) -> None:
    """Run vLLM inference for J judging K vs R and write K_vs_R_game1/game2 cache JSONLs.

    Args:
        llm: vLLM LLM engine.
        tokenizer: Tokenizer.
        out_dir (Path): Base output dir.
        dataset (str): Dataset.
        judge (str): Judge short name.
        reference (str): Reference short name.
        proxy (str): Proxy short name.
        examples (List[Dict[str, Any]]): Each item includes question, proxy_answer, reference_answer, example_id, category.
        reasoning_mode (str): One of none/cot/long_cot.
        temperature (float): Sampling temperature.
        max_tokens (int): Max tokens for generation.
        config (Dict[str, Any]): Reproduction config.
        batch_size (int): Inference batch size.
        overwrite (bool): Whether to overwrite existing cache files.
        logger (logging.Logger): Logger.

    Returns:
        None

    Raises:
        ValueError: If required example fields missing.
    """
    cache_base = out_dir / dataset / "cache" / judge / reference
    g1_path = cache_base / "K_vs_R_game1.jsonl"
    g2_path = cache_base / "K_vs_R_game2.jsonl"

    if overwrite:
        if g1_path.exists():
            g1_path.unlink()
        if g2_path.exists():
            g2_path.unlink()

    existing_g1 = read_existing_cache_keys(g1_path)
    existing_g2 = read_existing_cache_keys(g2_path)

    # Filter per-game by missing cache key (example||proxy).
    def _cache_key(ex: Dict[str, Any]) -> str:
        return f"{int(ex['example_id'])}||{proxy}"

    ex_game1 = [ex for ex in examples if _cache_key(ex) not in existing_g1]
    ex_game2 = [ex for ex in examples if _cache_key(ex) not in existing_g2]

    if not ex_game1 and not ex_game2:
        logger.info(f"K_vs_R already cached (skip): dataset={dataset} judge={judge} ref={reference} proxy={proxy}")
        return

    logger.info(
        f"Inferring K_vs_R for dataset={dataset} judge={judge} ref={reference} proxy={proxy}: "
        f"missing game1={len(ex_game1)}/{len(examples)} game2={len(ex_game2)}/{len(examples)}"
    )

    # GAME 1: A=proxy, B=reference
    new_rows_g1: List[Dict[str, Any]] = []
    if ex_game1:
        for chunk in chunked(ex_game1, batch_size):
            prompts = [
                build_verdict_prompt(
                    question=ex["question"],
                    response_a=ex["proxy_answer"],
                    response_b=ex["reference_answer"],
                    benchmark=dataset,
                    reasoning_mode=reasoning_mode,
                )
                for ex in chunk
            ]
            outputs = generate_verdict_vllm(llm, tokenizer, prompts, temperature, max_tokens, config, logger)
            for ex, out in zip(chunk, outputs):
                if reasoning_mode == "none":
                    probs, tok_pos = extract_abt_probs_from_logprobs(out, tokenizer, logger)
                else:
                    probs, tok_pos = extract_cot_verdict_with_logprobs(out, tokenizer, logger)

                raw = [float(probs["A"]), float(probs["B"]), float(probs["T"])]
                ck = _cache_key(ex)
                new_rows_g1.append(
                    {
                        "cache_key": ck,
                        "example_id": int(ex["example_id"]),
                        "judge": judge,
                        "reference": reference,
                        "proxy": proxy,
                        "comparison_type": "K_vs_R_game1",
                        "response1_key": "proxy",
                        "response2_key": "reference",
                        "prob_response1": raw[0],
                        "prob_response2": raw[1],
                        "generated_text": (out.outputs[0].text if out.outputs else "").strip(),
                        "raw_probs": raw,
                        "normalized_sum": float(sum(raw)),
                        "category": ex.get("category", "unknown"),
                        "dataset": dataset,
                        "verdict_token_position": int(tok_pos),
                    }
                )

    # GAME 2: A=reference, B=proxy
    new_rows_g2: List[Dict[str, Any]] = []
    if ex_game2:
        for chunk in chunked(ex_game2, batch_size):
            prompts = [
                build_verdict_prompt(
                    question=ex["question"],
                    response_a=ex["reference_answer"],
                    response_b=ex["proxy_answer"],
                    benchmark=dataset,
                    reasoning_mode=reasoning_mode,
                )
                for ex in chunk
            ]
            outputs = generate_verdict_vllm(llm, tokenizer, prompts, temperature, max_tokens, config, logger)
            for ex, out in zip(chunk, outputs):
                if reasoning_mode == "none":
                    probs, tok_pos = extract_abt_probs_from_logprobs(out, tokenizer, logger)
                else:
                    probs, tok_pos = extract_cot_verdict_with_logprobs(out, tokenizer, logger)

                raw = [float(probs["A"]), float(probs["B"]), float(probs["T"])]
                ck = _cache_key(ex)
                new_rows_g2.append(
                    {
                        "cache_key": ck,
                        "example_id": int(ex["example_id"]),
                        "judge": judge,
                        "reference": reference,
                        "proxy": proxy,
                        "comparison_type": "K_vs_R_game2",
                        "response1_key": "reference",
                        "response2_key": "proxy",
                        "prob_response1": raw[0],
                        "prob_response2": raw[1],
                        "generated_text": (out.outputs[0].text if out.outputs else "").strip(),
                        "raw_probs": raw,
                        "normalized_sum": float(sum(raw)),
                        "category": ex.get("category", "unknown"),
                        "dataset": dataset,
                        "verdict_token_position": int(tok_pos),
                    }
                )

    if new_rows_g1:
        append_jsonl(g1_path, new_rows_g1)
    if new_rows_g2:
        append_jsonl(g2_path, new_rows_g2)

    logger.info(
        f"Wrote K_vs_R cache for dataset={dataset} judge={judge} ref={reference} proxy={proxy}: "
        f"game1 +{len(new_rows_g1)}, game2 +{len(new_rows_g2)}"
    )


def run(verif_dir: Path, out_dir: Path, datasets: Optional[List[str]], judges: Optional[List[str]], references: Optional[List[str]], reasoning_mode: str, batch_size: int, tensor_parallel: Optional[int], max_proxies_per_ref: Optional[int], max_examples_per_proxy: Optional[int], overwrite: bool, dry_run: bool, logger: logging.Logger) -> None:
    """Main pipeline: load proxy sets, reuse JvsR probs, infer only KvsR, write caches.

    Execution order is: judge models -> datasets -> references -> proxies.

    Args:
        verif_dir (Path): Base verification directory.
        out_dir (Path): Base output directory.
        datasets (Optional[List[str]]): Datasets to include.
        judges (Optional[List[str]]): Judges to include.
        references (Optional[List[str]]): References to include.
        reasoning_mode (str): Reasoning mode.
        batch_size (int): Inference batch size.
        tensor_parallel (Optional[int]): vLLM tensor parallel size override.
        max_proxies_per_ref (Optional[int]): Optional cap for proxies per (judge, ref).
        max_examples_per_proxy (Optional[int]): Optional cap for examples per (judge, ref, proxy).
        overwrite (bool): Overwrite caches.
        dry_run (bool): If True, do not run inference or write.
        logger (logging.Logger): Logger.

    Returns:
        None

    Raises:
        ValueError: If verif_dir is missing.
    """
    if not verif_dir.exists():
        raise ValueError(f"Missing verif_dir: {verif_dir}")

    short_to_hf = build_short_to_hf_id_map()

    # Collect jobs grouped by judge.
    jobs_by_judge: Dict[str, List[Tuple[str, str, Path]]] = {}
    for dataset, family, judge_short, proxy_json_path in iter_proxy_json_files(verif_dir, datasets=datasets):
        if judges and judge_short not in set(judges):
            continue
        jobs_by_judge.setdefault(judge_short, []).append((dataset, family, proxy_json_path))

    if not jobs_by_judge:
        logger.warning("No proxy JSON jobs found; nothing to do")
        return

    judge_list = sorted(list(jobs_by_judge.keys()), key=model_size_key)

    logger.info(f"Found {len(judge_list)} judges with proxy sets")
    logger.info(f"Output dir: {out_dir}")

    for judge_short in judge_list:
        judge_key = judge_short.lower()
        if judge_key not in short_to_hf:
            logger.warning(f"No HF mapping for judge={judge_short}; skipping")
            continue

        judge_hf_id = short_to_hf[judge_key]
        logger.info("=" * 80)
        logger.info(f"JUDGE: {judge_short} -> {judge_hf_id}")
        logger.info("=" * 80)

        # Setup vLLM for this judge once.
        cfg = deepcopy(CONFIG)
        if tensor_parallel is not None:
            cfg["vllm_tensor_parallel_size"] = tensor_parallel


        temperature = 0.6 if is_reasoning_model(judge_hf_id) else 0.0
        max_tokens = cfg.get("verdict_max_tokens", 1) if reasoning_mode == "none" else cfg.get("cot_max_tokens", 2048)

        if dry_run:
            llm = None
            tokenizer = None
        else:
            llm = setup_vllm_engine(judge_hf_id, cfg, logger)
            tokenizer = llm.get_tokenizer()

        # Process datasets (sorted) for this judge.
        judge_jobs = sorted(jobs_by_judge[judge_short], key=lambda t: (t[0], t[1], str(t[2])))

        for dataset, family, proxy_json_path in judge_jobs:
            logger.info(f"Dataset={dataset} | judge_family={family} | proxies={proxy_json_path}")

            proxy_blob = load_json(proxy_json_path)

            # proxy_blob: { reference: { proxy: {data:{lsp:[{J_v_R,K_v_R},...], ilsp:[...]}, ...} } }
            for reference_name in sorted(proxy_blob.keys()):
                if references and reference_name not in set(references):
                    continue

                ref_block = proxy_blob[reference_name]
                proxies = sorted(list(ref_block.keys()))
                if max_proxies_per_ref is not None:
                    proxies = proxies[: max_proxies_per_ref]

                # Build J rows union for this (judge, reference).
                j_rows_by_id: Dict[int, Dict[str, Any]] = {}

                # Prepare KvsR example lists per proxy.
                kvs_by_proxy: Dict[str, List[Dict[str, Any]]] = {}

                for proxy_name in proxies:
                    entry = ref_block[proxy_name]
                    data = entry.get("data", {})

                    ex_list: List[Dict[str, Any]] = []
                    for cat in ("lsp", "ilsp"):
                        for pair in data.get(cat, []):
                            j_row = pair["J_v_R"]
                            k_row = pair["K_v_R"]

                            eid = int(j_row["to_eval_id"])
                            if eid not in j_rows_by_id:
                                j_rows_by_id[eid] = j_row

                            ex = {
                                "example_id": eid,
                                "question": k_row["question"],
                                "proxy_answer": k_row["response_a"],
                                "reference_answer": k_row["response_b"],
                                "category": cat,
                            }
                            ex_list.append(ex)

                    # Deduplicate by example_id within this proxy.
                    seen_ids: Set[int] = set()
                    deduped: List[Dict[str, Any]] = []
                    for ex in ex_list:
                        if int(ex["example_id"]) in seen_ids:
                            continue
                        seen_ids.add(int(ex["example_id"]))
                        deduped.append(ex)

                    if max_examples_per_proxy is not None:
                        deduped = deduped[: max_examples_per_proxy]

                    kvs_by_proxy[proxy_name] = deduped

                if dry_run:
                    logger.info(
                        f"DRY RUN: judge={judge_short} dataset={dataset} ref={reference_name} "
                        f"proxies={len(kvs_by_proxy)} j_examples={len(j_rows_by_id)}"
                    )
                    continue

                # Write JvsR caches from existing probabilities.
                write_j_vs_r_cache(
                    out_dir=out_dir,
                    dataset=dataset,
                    judge=judge_short,
                    reference=reference_name,
                    j_rows_by_id=j_rows_by_id,
                    overwrite=overwrite,
                    logger=logger,
                )

                # Infer KvsR for each proxy.
                for proxy_name in proxies:
                    examples = kvs_by_proxy.get(proxy_name, [])
                    if not examples:
                        continue

                    infer_k_vs_r_cache(
                        llm=llm,
                        tokenizer=tokenizer,
                        out_dir=out_dir,
                        dataset=dataset,
                        judge=judge_short,
                        reference=reference_name,
                        proxy=proxy_name,
                        examples=examples,
                        reasoning_mode=reasoning_mode,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        config=cfg,
                        batch_size=batch_size,
                        overwrite=overwrite,
                        logger=logger,
                    )

        # IMPORTANT: vLLM/torch can hold onto significant GPU memory.
        # When we move to the next judge model, explicitly free resources.
        if not dry_run:
            try:
                del llm
                del tokenizer
            except Exception:
                pass
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Failed to fully clear CUDA cache between judges: {e}")


# --------------------
# --- CLI / MAIN   ---
# --------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed args.

    Raises:
        SystemExit: For invalid arguments.
    """
    p = argparse.ArgumentParser(description="Run judge-swap null verif (only infer J(K vs R), reuse existing J vs R probs)")
    p.add_argument("--verif_dir", type=str, default="llm-sp-verif", help="Verification directory containing proxies/")
    p.add_argument("--out_dir", type=str, default="judge_swap_null_verif", help="Output directory for analyzer-compatible caches")
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated datasets (default: all found)")
    p.add_argument("--judges", type=str, default=None, help="Comma-separated judge short names (default: all)")
    p.add_argument("--references", type=str, default=None, help="Comma-separated reference short names (default: all)")
    p.add_argument("--reasoning_mode", type=str, default="none", choices=["none", "cot", "long_cot"], help="Prompt/verdict mode")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size for vLLM generation")
    p.add_argument("--tensor_parallel", type=int, default=None, help="Override vLLM tensor_parallel_size")
    p.add_argument("--max_proxies_per_ref", type=int, default=None, help="Optional cap for proxies per (judge, ref)")
    p.add_argument("--max_examples_per_proxy", type=int, default=None, help="Optional cap for examples per (judge, ref, proxy)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    p.add_argument("--dry_run", action="store_true", help="Do not write or run inference; just report work")
    return p.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    logger = setup_logging()

    datasets = args.datasets.split(",") if args.datasets else None
    judges = args.judges.split(",") if args.judges else None
    references = args.references.split(",") if args.references else None

    run(
        verif_dir=Path(args.verif_dir),
        out_dir=Path(args.out_dir),
        datasets=datasets,
        judges=judges,
        references=references,
        reasoning_mode=args.reasoning_mode,
        batch_size=args.batch_size,
        tensor_parallel=args.tensor_parallel,
        max_proxies_per_ref=args.max_proxies_per_ref,
        max_examples_per_proxy=args.max_examples_per_proxy,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        logger=logger,
    )


if __name__ == "__main__":
    main()
