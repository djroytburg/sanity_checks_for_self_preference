"""verify_results.py

Verify reproduction JSONLs against original llm-sp results and generate proxy datasets.

This script performs two main tasks:
1) Create *_reprod_verified.jsonl files by attaching `to_eval_id` and oracle-derived `gold_label`.
2) For each (dataset, judge, reference), find proxy models K (possibly from other families) where
   gold labels agree on shared `to_eval_id` examples, then emit DBG-style table stats and a
   per-judge JSON proxy dataset used by downstream judge-swap tests.
"""

# verify_results.py: Verify llm-sp reproduction mapping and generate proxy datasets
# Written by: Dani
# Created: Jan 15, 2026, 03:05 EST
# Last Modified: Jan 15, 2026, 03:05 EST

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch



# ----------------------
# --- FILE UTILITIES ---
# ----------------------

def _now_est_timestamp_str() -> str:
    """Return a stable timestamp string used for log filenames.

    Args:
        None

    Returns:
        str: Timestamp string in YYYYMMDD_HHMMSS format.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_text(text: str) -> str:
    """Normalize text for strict-but-tolerant comparisons.

    Args:
        text (str): Input text.

    Returns:
        str: Stripped text.
    """
    return (text or "").strip()


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load a JSONL file.

    Args:
        filepath (Path): Path to the JSONL.

    Returns:
        List[Dict]: List of parsed objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(filepath: Path, rows: Iterable[Dict]) -> None:
    """Write a JSONL file.

    Args:
        filepath (Path): Destination path.
        rows (Iterable[Dict]): Rows to write.

    Returns:
        None
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup logging to console and to file_logs/ with timestamps.

    Args:
        debug (bool): If True, emit DEBUG logs to console.

    Returns:
        logging.Logger: Configured logger.
    """
    Path("file_logs").mkdir(parents=True, exist_ok=True)
    timestamp = _now_est_timestamp_str()
    log_path = Path("file_logs") / f"verify_results_{timestamp}.log"

    logger = logging.getLogger("verify_results")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("VERIFY RESULTS + PROXY GENERATION")
    logger.info("=" * 80)
    logger.info(f"User: {getpass.getuser()}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        except Exception:
            logger.info("CUDA device count: (unavailable)")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 80)
    return logger


def parse_judge_ref_from_stem(stem: str) -> Tuple[str, str]:
    """Parse (judge, reference) from a result filename stem.

    Supports stems like:
      - "{judge}_{ref}_reprod"
      - "{judge}_{ref}_reprod_verified"
      - "{judge}_{ref}_eval"

    Args:
        stem (str): Filename without extension.

    Returns:
        Tuple[str, str]: (judge, reference)

    Raises:
        ValueError: If parsing fails.
    """
    for suffix in ("_reprod_verified", "_reprod", "_eval", "_average", "_merge"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(f"Could not parse judge/ref from stem: {stem}")
    return parts[0], parts[1]



# ---------------------------
# --- VERIFIED MAP OUTPUT ---
# ---------------------------

def map_results(
    repro_dir: str,
    orig_dir: str,
    verif_dir: str,
    datasets: Optional[List[str]],
    families: Optional[List[str]],
    overwrite: bool,
    strict_response_match: bool,
    logger: logging.Logger,
) -> None:
    """Map reproduction results to original results, verify mapping, and add oracle labels.

    This creates files named *_reprod_verified.jsonl under verif_dir.

    Args:
        repro_dir (str): Directory containing reproduction results JSONLs.
        orig_dir (str): Directory containing original llm-sp results JSONLs.
        verif_dir (str): Directory to write verified JSONLs.
        datasets (Optional[List[str]]): Datasets to process.
        families (Optional[List[str]]): Model families to process.
        overwrite (bool): Overwrite verified files if they exist.
        strict_response_match (bool): If True, require response_a/response_b match original answers.
        logger (logging.Logger): Logger.

    Returns:
        None

    Raises:
        ValueError: If required directories/files are missing.
        AssertionError: If strict validation fails.
    """
    repro_path = Path(repro_dir)
    orig_path = Path(orig_dir)
    save_path = Path(verif_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    datasets = datasets or ["math500", "mbpp-plus", "mmlu"]
    families = families or ["qwen", "gemma", "llama"]

    for dataset in datasets:
        og_sdir = orig_path / dataset
        repro_sdir = repro_path / dataset

        if not og_sdir.exists():
            raise ValueError(f"Original dataset directory does not exist: {og_sdir}")
        if not repro_sdir.exists():
            raise ValueError(f"Reproduction dataset directory does not exist: {repro_sdir}")

        logger.info(f"Mapping dataset: {dataset}")

        # These dictionaries are shared across families so mapping is consistent within dataset.
        question_to_eval_id: Dict[str, int] = {}
        response_by_question_and_model: Dict[Tuple[str, str], str] = {}

        for family in families:
            og_fdir = og_sdir / family
            repro_fdir = repro_sdir / family
            if not repro_fdir.exists():
                logger.warning(f"Reproduction family directory missing (skipping): {repro_fdir}")
                continue
            if not og_fdir.exists():
                logger.warning(f"Original family directory missing (skipping): {og_fdir}")
                continue

            for og_file in og_fdir.glob("*.jsonl"):
                judge, ref = parse_judge_ref_from_stem(og_file.stem)
                repro_stem = og_file.stem.replace("_eval", "_reprod")
                repro_file = repro_fdir / f"{repro_stem}.jsonl"
                if not repro_file.exists():
                    raise ValueError(f"Missing reproduction file for original: {og_file} -> {repro_file}")

                verified_stem = repro_stem + "_verified"
                out_dir = save_path / dataset / family
                out_file = out_dir / f"{verified_stem}.jsonl"
                if out_file.exists() and not overwrite:
                    logger.info(f"Verified exists (skip): {out_file}")
                    continue

                og_data = load_jsonl(og_file)
                repro_data = load_jsonl(repro_file)

                # Build quick lookup from question->original row index (only within this file).
                og_question_to_idx = {row["problem"]: idx for idx, row in enumerate(og_data)}

                verified_rows: List[Dict] = []
                for row in repro_data:
                    q = row.get("question")
                    if q is None:
                        raise ValueError(f"Missing 'question' in repro row in {repro_file}")
                    if q not in og_question_to_idx:
                        raise ValueError(f"Question not found in original data: {q}")
                    og_row = og_data[og_question_to_idx[q]]

                    # Stable dataset-level eval id.
                    if q not in question_to_eval_id:
                        question_to_eval_id[q] = og_question_to_idx[q]
                    row["to_eval_id"] = question_to_eval_id[q]

                    # Validate response identity per model across the dataset.
                    j_key = (q, judge)
                    r_key = (q, ref)
                    j_resp = og_row.get("assistent_1_answer", "")
                    r_resp = og_row.get("assistent_2_answer", "")
                    if j_key in response_by_question_and_model:
                        assert response_by_question_and_model[j_key] == j_resp, (
                            f"J response mismatch for question and judge: {judge}\n"
                            f"question={q}"
                        )
                    else:
                        response_by_question_and_model[j_key] = j_resp
                    if r_key in response_by_question_and_model:
                        assert response_by_question_and_model[r_key] == r_resp, (
                            f"R response mismatch for question and ref: {ref}\n"
                            f"question={q}"
                        )
                    else:
                        response_by_question_and_model[r_key] = r_resp

                    # Optionally enforce the reproduction file truly contains those same responses.
                    if strict_response_match:
                        assert _normalize_text(row.get("response_a", "")) == _normalize_text(j_resp), (
                            f"response_a mismatch vs original for {judge} vs {ref} in {repro_file.name}\n"
                            f"question={q}"
                        )
                        assert _normalize_text(row.get("response_b", "")) == _normalize_text(r_resp), (
                            f"response_b mismatch vs original for {judge} vs {ref} in {repro_file.name}\n"
                            f"question={q}"
                        )

                    # Oracle correctness + gold label.
                    row["judge_model"] = judge
                    row["reference_model"] = ref
                    row["dataset"] = dataset
                    row["family"] = family
                    row["j_right"] = bool(og_row.get("assistent_1_is_correct"))
                    row["r_right"] = bool(og_row.get("assistent_2_is_correct"))

                    if row["j_right"] and not row["r_right"]:
                        row["gold_label"] = "lsp"
                    elif (not row["j_right"]) and row["r_right"]:
                        row["gold_label"] = "ilsp"
                    elif row["j_right"] and row["r_right"]:
                        row["gold_label"] = "both"
                    elif (not row["j_right"]) and (not row["r_right"]):
                        row["gold_label"] = "neither"
                    else:
                        raise ValueError(f"Invalid correctness flags: {row['j_right']}, {row['r_right']}")

                    verified_rows.append(row)

                write_jsonl(out_file, verified_rows)
                logger.info(
                    f"Saved verified: {out_file} (n={len(verified_rows)}) for judge={judge}, ref={ref}"
                )



# -----------------------------
# --- PROXY GENERATION LOGIC ---
# -----------------------------

def _build_id_map(rows: List[Dict]) -> Dict[int, Dict]:
    """Index rows by to_eval_id.

    Args:
        rows (List[Dict]): Verified rows.

    Returns:
        Dict[int, Dict]: Mapping from to_eval_id -> row.
    """
    out: Dict[int, Dict] = {}
    for row in rows:
        if "to_eval_id" not in row:
            continue
        out[int(row["to_eval_id"])] = row
    return out


def _eligible_gold_label(row: Dict) -> Optional[str]:
    """Return the gold label if eligible for proxy matching.

    Args:
        row (Dict): Verified row.

    Returns:
        Optional[str]: 'lsp' or 'ilsp' if eligible, else None.
    """
    gl = row.get("gold_label")
    if gl in ("lsp", "ilsp"):
        return gl
    return None


def compute_proxy_agreement(
    j_rows: List[Dict],
    k_rows: List[Dict],
    judge_name: str,
    proxy_name: str,
    reference_name: str,
    logger: logging.Logger,
    assert_question_and_ref_response: bool = True,
) -> Tuple[int, int, float, int, int, float, Dict[str, List[Dict]]]:
    """Compute DBG-style agreement metrics and construct matched example payloads.

    Agreement is computed on the intersection of `to_eval_id` between JvR and KvR.
    We only consider rows with `gold_label` in {'lsp','ilsp'}.

    Args:
        j_rows (List[Dict]): Verified rows for J vs R.
        k_rows (List[Dict]): Verified rows for K vs R.
        judge_name (str): Judge model J.
        proxy_name (str): Proxy model K.
        reference_name (str): Reference model R.
        logger (logging.Logger): Logger.
        assert_question_and_ref_response (bool): If True, assert question and response_b match.

    Returns:
        Tuple[int, int, float, int, int, float, Dict[str, List[Dict]]]:
            (n_agree, n_total, agreement_rate, n_right, n_wrong, proxy_winrate, data)
    """
    j_by_id = _build_id_map(j_rows)
    k_by_id = _build_id_map(k_rows)

    j_ids = {eid for eid, row in j_by_id.items() if _eligible_gold_label(row) is not None}
    k_ids = {eid for eid, row in k_by_id.items() if _eligible_gold_label(row) is not None}
    common_ids = sorted(list(j_ids & k_ids))

    n_total = len(common_ids)
    n_agree = 0
    n_right = 0
    n_wrong = 0
    data: Dict[str, List[Dict]] = {"lsp": [], "ilsp": []}

    # Proxy winrate (K beats R) over its eligible examples.
    k_eligible = [k_by_id[eid] for eid in k_ids]
    proxy_winrate = (
        sum(1 for r in k_eligible if r.get("gold_label") == "lsp") / len(k_eligible)
        if k_eligible
        else 0.0
    )

    for eid in common_ids:
        j_row = j_by_id[eid]
        k_row = k_by_id[eid]

        if assert_question_and_ref_response:
            assert _normalize_text(j_row.get("question", "")) == _normalize_text(k_row.get("question", "")), (
                f"Question mismatch for to_eval_id={eid} (J={judge_name}, K={proxy_name}, R={reference_name})"
            )
            assert _normalize_text(j_row.get("response_b", "")) == _normalize_text(k_row.get("response_b", "")), (
                f"Reference response mismatch for to_eval_id={eid} (J={judge_name}, K={proxy_name}, R={reference_name})"
            )

        j_gl = _eligible_gold_label(j_row)
        k_gl = _eligible_gold_label(k_row)
        if j_gl is None or k_gl is None:
            continue

        if j_gl == k_gl:
            n_agree += 1
            if j_gl == "lsp":
                n_right += 1
                data["lsp"].append({"J_v_R": j_row, "K_v_R": k_row})
            elif j_gl == "ilsp":
                n_wrong += 1
                data["ilsp"].append({"J_v_R": j_row, "K_v_R": k_row})

    agreement_rate = (n_agree / n_total) if n_total > 0 else 0.0

    return n_agree, n_total, agreement_rate, n_right, n_wrong, proxy_winrate, data


def generate_proxies(
    verif_dir: str,
    datasets: Optional[List[str]],
    min_per_class: int,
    max_print: int,
    logger: logging.Logger,
) -> None:
    """Generate proxies across families (same reference name) and save per-judge JSON outputs.

    Output format:
        verif_dir/dataset/{judge_family}/proxies/{judge}.json
    Top-level JSON:
        { reference_model: { proxy_model: { ...metadata..., data: {lsp: [...], ilsp: [...]}}}}

    Args:
        verif_dir (str): Directory containing verified JSONLs.
        datasets (Optional[List[str]]): Datasets to process.
        min_per_class (int): Minimum required matches in each of LSP/ILSP to keep a proxy.
        max_print (int): Max proxies to print per judge-reference in the table.
        logger (logging.Logger): Logger.

    Returns:
        None
    """
    base = Path(verif_dir)
    datasets = datasets or ["math500", "mbpp-plus", "mmlu"]

    for dataset in datasets:
        dataset_dir = base / dataset
        if not dataset_dir.exists():
            logger.warning(f"Verified dataset dir missing (skip): {dataset_dir}")
            continue

        # Collect all result files under this dataset (across families).
        # Prefer *_reprod_verified.jsonl when present; otherwise fall back per-family to *_reprod.jsonl.
        all_files: List[Path] = []
        for family_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            verified = sorted(list(family_dir.glob("*_reprod_verified.jsonl")))
            if verified:
                all_files.extend(verified)
                continue

            # Backward compat: fall back to *_reprod.jsonl if verified not present in this family.
            reprod = sorted(list(family_dir.glob("*_reprod.jsonl")))
            all_files.extend(reprod)

        if not all_files:
            logger.warning(f"No verified JSONLs found for dataset={dataset} under {dataset_dir}")
            continue

        # Index files by (judge, ref) and also group candidates by ref.
        file_by_pair: Dict[Tuple[str, str], Path] = {}
        family_by_pair: Dict[Tuple[str, str], str] = {}
        files_by_ref: Dict[str, List[Tuple[str, Path]]] = {}

        for fp in all_files:
            judge, ref = parse_judge_ref_from_stem(fp.stem)
            file_by_pair[(judge, ref)] = fp
            family_by_pair[(judge, ref)] = fp.parent.name
            files_by_ref.setdefault(ref, []).append((judge, fp))

        logger.info(f"Dataset={dataset}: found {len(file_by_pair)} (judge,ref) files")

        # For each base family, we build outputs per judge for the judge files living in that family.
        judge_pairs_by_family: Dict[str, List[Tuple[str, str]]] = {}
        for (judge, ref), fam in family_by_pair.items():
            judge_pairs_by_family.setdefault(fam, []).append((judge, ref))

        for family, pairs in judge_pairs_by_family.items():
            logger.info(f"Generating proxies for dataset={dataset}, family={family} (pairs={len(pairs)})")

            # Group by judge so we can emit one JSON per judge with multiple references.
            refs_by_judge: Dict[str, List[str]] = {}
            for judge, ref in pairs:
                refs_by_judge.setdefault(judge, []).append(ref)

            for judge_name, references in refs_by_judge.items():
                output_by_reference: Dict[str, Dict[str, Dict]] = {}

                for reference_name in sorted(set(references)):
                    j_file = file_by_pair.get((judge_name, reference_name))
                    if j_file is None:
                        continue

                    j_rows = load_jsonl(j_file)
                    j_eligible = [r for r in j_rows if _eligible_gold_label(r) is not None]
                    judge_winrate = (
                        sum(1 for r in j_eligible if r.get("gold_label") == "lsp") / len(j_eligible)
                        if j_eligible
                        else 0.0
                    )

                    # Candidate proxies are all models K where a (K, R) file exists anywhere.
                    candidates = [k for (k, _p) in files_by_ref.get(reference_name, [])]
                    candidates = [c for c in sorted(set(candidates)) if c not in (judge_name, reference_name)]

                    results_for_table: List[Dict] = []
                    proxies_for_ref: Dict[str, Dict] = {}
                    for proxy_name in candidates:
                        k_file = file_by_pair.get((proxy_name, reference_name))
                        if k_file is None:
                            continue

                        k_rows = load_jsonl(k_file)
                        n_agree, n_total, rate, n_right, n_wrong, proxy_winrate, data = compute_proxy_agreement(
                            j_rows=j_rows,
                            k_rows=k_rows,
                            judge_name=judge_name,
                            proxy_name=proxy_name,
                            reference_name=reference_name,
                            logger=logger,
                            assert_question_and_ref_response=True,
                        )

                        if n_right < min_per_class or n_wrong < min_per_class:
                            continue

                        proxies_for_ref[proxy_name] = {
                            "n_agree": n_agree,
                            "n_total": n_total,
                            "n_right": n_right,
                            "n_wrong": n_wrong,
                            "agreement_rate": rate,
                            "judge_winrate": judge_winrate,
                            "proxy_winrate": proxy_winrate,
                            "metadata": {
                                "judge_name": judge_name,
                                "proxy_name": proxy_name,
                                "reference_name": reference_name,
                                "dataset": dataset,
                                "judge_file": str(j_file),
                                "proxy_file": str(k_file),
                            },
                            "data": data,
                        }
                        results_for_table.append(
                            {
                                "proxy": proxy_name,
                                "n_agree": n_agree,
                                "n_total": n_total,
                                "rate": rate,
                                "n_right": n_right,
                                "n_wrong": n_wrong,
                                "judge_winrate": judge_winrate,
                                "proxy_winrate": proxy_winrate,
                            }
                        )

                    # Sort proxies by ILSP count (DBG-style) for reporting.
                    results_for_table.sort(key=lambda x: x["n_wrong"], reverse=True)

                    logger.info("")
                    logger.info("=" * 80)
                    logger.info(f"PROXY MODELS for dataset={dataset} | Judge={judge_name} | Ref={reference_name}")
                    logger.info("=" * 80)
                    winrate_text = f"J.WR {judge_winrate:.1%}"
                    logger.info(
                        f"{'Proxy Model':<35} {'Agree':>6} {'Total':>6} {'Rate':>6} {'LSP':>6} {'ILSP':>6} {winrate_text:>12} {'K.WR':>8}"
                    )
                    logger.info("-" * 80)
                    for r in results_for_table[:max_print]:
                        logger.info(
                            f"{r['proxy']:<35} {r['n_agree']:>6} {r['n_total']:>6} {r['rate']:>6.1%} {r['n_right']:>6} {r['n_wrong']:>6} {r['judge_winrate']:>12.1%} {r['proxy_winrate']:>8.1%}"
                        )
                    logger.info("-" * 80)
                    logger.info(f"Found {len(proxies_for_ref)} suitable proxy models")

                    if proxies_for_ref:
                        # Sort JSON by proxy name for stability; reporting sort is separate.
                        output_by_reference[reference_name] = {
                            k: proxies_for_ref[k] for k in sorted(proxies_for_ref.keys())
                        }

                if not output_by_reference:
                    continue

                out_dir = base / dataset / family / "proxies"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{judge_name}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(output_by_reference, f, indent=2)
                logger.info(f"Saved per-judge proxy JSON: {out_file}")



# ----------------------
# --- MAIN EXECUTION ---
# ----------------------

def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify reproduction results against original llm-sp results, attach oracle labels, "
            "and generate cross-family proxy datasets."
        )
    )
    parser.add_argument("--repro_dir", type=str, default="./", help="Directory containing reproduction results.")
    parser.add_argument("--orig_dir", type=str, default="../llm-sp/sp/", help="Directory containing original results.")
    parser.add_argument(
        "--verif_dir",
        type=str,
        default="../llm-repro-verif",
        help="Directory to save verified JSONLs and proxy JSON outputs.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to process (default: math500 mbpp-plus mmlu).",
    )
    parser.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=None,
        help="Families to process (default: qwen gemma llama).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite verified outputs if present.")
    parser.add_argument(
        "--no_strict_response_match",
        action="store_true",
        help="Disable strict response_a/response_b matching to original answers.",
    )
    parser.add_argument(
        "--map_only",
        action="store_true",
        help="Only write *_reprod_verified.jsonl files (no proxy generation).",
    )
    parser.add_argument(
        "--proxies_only",
        action="store_true",
        help="Only generate proxies from existing verified files (no mapping step).",
    )
    parser.add_argument(
        "--min_per_class",
        type=int,
        default=5,
        help="Minimum required matched examples in each of LSP and ILSP to keep a proxy.",
    )
    parser.add_argument(
        "--max_print",
        type=int,
        default=30,
        help="Max proxies to print per judge-reference table.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG console logging.")
    args = parser.parse_args()

    logger = setup_logging(debug=args.debug)
    logger.info("Configuration:")
    logger.info(f"  repro_dir: {args.repro_dir}")
    logger.info(f"  orig_dir: {args.orig_dir}")
    logger.info(f"  verif_dir: {args.verif_dir}")
    logger.info(f"  datasets: {args.datasets}")
    logger.info(f"  families: {args.families}")
    logger.info(f"  overwrite: {args.overwrite}")
    logger.info(f"  strict_response_match: {not args.no_strict_response_match}")
    logger.info(f"  map_only: {args.map_only}")
    logger.info(f"  proxies_only: {args.proxies_only}")
    logger.info(f"  min_per_class: {args.min_per_class}")
    logger.info(f"  max_print: {args.max_print}")

    if not args.proxies_only:
        map_results(
            repro_dir=args.repro_dir,
            orig_dir=args.orig_dir,
            verif_dir=args.verif_dir,
            datasets=args.datasets,
            families=args.families,
            overwrite=args.overwrite,
            strict_response_match=not args.no_strict_response_match,
            logger=logger,
        )
        logger.info("Mapping step complete.")

    if not args.map_only:
        generate_proxies(
            verif_dir=args.verif_dir,
            datasets=args.datasets,
            min_per_class=args.min_per_class,
            max_print=args.max_print,
            logger=logger,
        )
        logger.info("Proxy generation step complete.")

    logger.info("Done.")


if __name__ == "__main__":
    main()