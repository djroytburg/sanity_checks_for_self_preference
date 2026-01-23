# run_judge_swap_null_author_obfuscation.py: Judge swap null hypothesis test using author_obfuscation harness
# Tests H0: P(J chooses J over R) = P(J chooses K over R) where K is a proxy with similar capability
# Adapted from run_judge_swap_null_dbg.py to work with author_obfuscation infrastructure
# Created: Jan 4, 2026, 19:00 EST
# Last Modified: Jan 14 2026, 09:30 EST

import sys
import os
import json
import torch
import torch.nn.functional as F
import asyncio
import argparse
import logging
import getpass
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
import csv
import random

# Add author_obfuscation to path
sys.path.insert(0, str(Path(__file__).parent / "author_obfuscation"))

from transformers import AutoTokenizer, AutoModelForCausalLM

# Import existing author_obfuscation components
# from author_obfuscation.code.common.clients import format_model_name_together  # Removed to avoid Together client
from author_obfuscation.code.common.prompts import (
    QA_COMPARISON_SYSTEM_PROMPT_QUALITY,
    QA_COMPARISON_PROMPT_TEMPLATE_QUALITY,
)


# --------------------------
# --- FORMATTING UTILITY ---
# --------------------------

def get_hf_model_path(model_name: str) -> str:
    """
    Normalizes friendly model names to Together-hosted slugs.
    """
    model_name = model_name.replace("-Turbo", "")
    if model_name.startswith("Meta-Llama") or model_name.startswith("Llama"):
        return f"meta-llama/{model_name}"
    if model_name.startswith("Qwen"):
        return f"Qwen/{model_name}"
    if model_name.startswith("DeepSeek"):
        return f"deepseek-ai/{model_name}"
    if model_name.startswith("Mistral") or model_name.startswith("Mixtral"):
        return f"mistralai/{model_name}"
    if model_name.startswith("Kimi"):
        return f"moonshotai/{model_name}"
    if model_name.startswith("gpt"):
        return f"openai/{model_name}"
    if model_name.startswith("GLM"):
        return f"zai-org/{model_name}"
    return model_name


# --------------------------
# --- PROMPT FORMATTING ---
# --------------------------

def format_preference_prompt(article: str, query: str, response1, response2, dataset: str = 'quality') -> List[Dict]:
    """
    Format preference prompt for smoke test.
    response1, response2 are tuples of form (reason, label).
    """
    if dataset == 'quality':
        if type(response1) != type(response2):
            raise ValueError(f"Response types must match for preference prompt formatting. Types: {type(response1)}, {type(response2)}")
        elif isinstance(response1, tuple):
            response1 = response1[1] + ". " + response1[0]
            response2 = response2[1] + ". " + response2[0]
        elif isinstance(response1, str):
            pass
        else:
            raise ValueError(f"Unknown response type for preference prompt formatting: {type(response1)}")
        prompt = QA_COMPARISON_PROMPT_TEMPLATE_QUALITY.format(
            article=article,
            question=query,
            answer1=response1,
            answer2=response2
        )
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "system", "content": QA_COMPARISON_SYSTEM_PROMPT_QUALITY},
        ]
        
        return messages
    else:
        raise ValueError(f"Unknown dataset for prompt formatting: {dataset}")
    
# --------------------------
# ---- HF MODEL MANAGER ----
# --------------------------
class AuthorObfuscationHFManager:
    """
    Adapted HF manager for author_obfuscation framework.
    Directly borrows from BatchHFManager (DBG Scoring) with custom model mapping.
    """

    def __init__(self, model_path: str, logger: logging.Logger):
        self.model_path = model_path
        self.logger = logger
                
        logger.info(f"Loading model: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if "gemma-2" in model_path.lower():
            self.tokenizer.padding_side = "right"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=os.environ.get("HF_TOKEN"),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.model.device}")
    
    def format_messages(self, messages: List[List[Dict]]) -> List[str]:
        """
        Format chat messages into prompts.
        For instruct models, use chat template. For base models, concatenate content.
        """
        prompts = []
        for msg in messages:
            prompt = self.tokenizer.apply_chat_template(
                msg, tokenize=False, skip_special_tokens=False, add_generation_prompt=True
            )
            prompts.append(prompt)
        return prompts
    
    def prefer_generate_batch(
        self,
        messages: List[List[Dict]],
        max_tokens: int = 1,
        temperature: float = 0.0,
        logger=None,
        batch_size: int = 1,
    ) -> List[Dict]:
        """
                Generate preferences for a batch of messages.

                This uses a single `model.generate(..., return_dict_in_generate=True, output_scores=True)`
                call per batch chunk to obtain both:
                    1) the generated text, and
                    2) the logits for the generated token(s) (via `output.scores`).

                Preference probabilities are computed from the *first generated token* distribution.
                In the author_obfuscation prompt, the judge should answer with a choice like "1" or "2".
        
        Returns list of dicts with:
            - prob_A: Normalized probability of choosing A
            - prob_B: Normalized probability of choosing B  
            - generated_text: The actual output sequence
            - raw_probs: Unnormalized [prob_A, prob_B]
        """
        if logger is None:
            logger = self.logger

        device = self.model.device
        results: List[Dict] = []

        def _choice_token_ids(text_variants: List[str]) -> List[int]:
            ids: List[int] = []
            for t in text_variants:
                enc = self.tokenizer.encode(t, add_special_tokens=False)
                if enc:
                    ids.append(enc[0])
            # de-dupe while preserving order
            seen = set()
            uniq = []
            for tid in ids:
                if tid not in seen:
                    uniq.append(tid)
                    seen.add(tid)
            return uniq

        # Some tokenizers prefer leading-space tokens for standalone digits.
        token_A_ids = _choice_token_ids(["1", " 1", "\n1"])
        token_B_ids = _choice_token_ids(["2", " 2", "\n2"])
        if not token_A_ids or not token_B_ids:
            logger.warning(
                f"Could not reliably tokenize choice digits; token_A_ids={token_A_ids}, token_B_ids={token_B_ids}"
            )

        prompts = self.format_messages(messages)

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_msgs = messages[start : start + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = output.sequences
            scores = output.scores
            if not scores:
                logger.error("model.generate returned no scores; falling back to uniform probabilities")
                scores0 = None
            else:
                scores0 = scores[0]

            # Prompt lengths differ with padding; use attention_mask to slice generated tokens.
            prompt_lens = inputs.attention_mask.sum(dim=1).tolist()

            if scores0 is not None:
                probs0 = F.softmax(scores0, dim=-1)

            for i in range(len(batch_prompts)):
                if scores0 is None:
                    prob_A = 0.5
                    prob_B = 0.5
                    raw_A = 0.0
                    raw_B = 0.0
                else:
                    # Sum probabilities across a few common tokenizations.
                    raw_A = float(probs0[i, token_A_ids].sum().item()) if token_A_ids else 0.0
                    raw_B = float(probs0[i, token_B_ids].sum().item()) if token_B_ids else 0.0
                    total = raw_A + raw_B
                    if total > 0:
                        prob_A = raw_A / total
                        prob_B = raw_B / total
                    else:
                        prob_A = 0.5
                        prob_B = 0.5

                # Decode only newly-generated tokens for this row.
                prompt_len = int(prompt_lens[i])
                generated_tokens = sequences[i, prompt_len:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                # Optional debug diagnostics.
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Prompt (idx={start + i}) length={prompt_len}")
                    logger.debug(f"Choice token ids: A={token_A_ids} B={token_B_ids}")
                    logger.debug(f"Generated text: {generated_text!r}")
                    logger.debug(f"Raw probs: A={raw_A} B={raw_B} (norm sum={prob_A + prob_B})")
                    try:
                        if scores0 is not None:
                            k = min(10, probs0.shape[-1])
                            topv, topi = torch.topk(probs0[i], k)
                            top_tokens = []
                            for tid, tv in zip(topi.tolist(), topv.tolist()):
                                try:
                                    token_text = self.tokenizer.convert_ids_to_tokens(tid)
                                except Exception:
                                    token_text = self.tokenizer.decode([tid], skip_special_tokens=True)
                                top_tokens.append((tid, token_text, float(tv)))
                            logger.debug(f"Top-{k} first-token probs: {top_tokens}")
                    except Exception as e:
                        logger.debug(f"Failed to compute top-k tokens: {e}")

                results.append(
                    {
                        "prob_A": prob_A,
                        "prob_B": prob_B,
                        "generated_text": generated_text,
                        "raw_probs": [raw_A, raw_B],
                        "normalized_sum": prob_A + prob_B,
                    }
                )

        return results
    
    def __del__(self):
        """Cleanup GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()


# ----------------------------
# ---- PROXY DATA LOADING ----
# ----------------------------

def load_author_obfuscation_proxy_data(proxy_dir: Path, logger: logging.Logger, response_dir: Path = None, judge_models: List[str] = None) -> Dict:
    """
    Load proxy data from author_obfuscation format.
    Converts to format expected by judge swap test.

    Returns:
        Dict mapping (judge, proxy, reference) -> {
            'lsp': [...examples...],
            'ilsp': [...examples...],
            'metadata': {...}
        }
    """
    data_by_triplet = {}

    # Load the quality preference data to get examples
    csv_file = response_dir or Path(proxy_dir).parent.parent / "quality_responses.csv"
    csv_data = {}
    if csv_file.exists():
        logger.info(f"Loading question text from {csv_file}")
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_data[row['pid']] = row
    else:
        logger.warning(f"CSV file not found: {csv_file}, using JSON questions where available")

    # # Load quality preference data from both ben and harmful JSONs
    # preference_dir = Path(proxy_dir).parent / "preference_results"
    # ben_file = preference_dir / "clean_pref_quality_ben.json"
    # harmful_file = preference_dir / "clean_pref_quality_harmful.json"
    
    # quality_data = []
    # for json_file, dataset_tag in [(ben_file, "ben"), (harmful_file, "harmful")]:
    #     if json_file.exists():
    #         logger.info(f"Loading {dataset_tag} data from {json_file}")
    #         with open(json_file) as f:
    #             data = json.load(f)
    #             for ex in data:
    #                 ex['dataset'] = dataset_tag
    #             quality_data.extend(data)
    #     else:
    #         raise ValueError(f"Required data file not found: {json_file}")
    
    # logger.info(f"Loaded {len(quality_data)} examples from JSON files")
    
    # # Debug: print keys of first example
    # if len(quality_data) > 0:
    #     logger.info(f"First example keys: {list(quality_data[0].keys())}")
    # else:
    #     raise ValueError("No quality data loaded; cannot proceed.")

    # Process each proxy file
    for proxy_file in proxy_dir.glob("*.json"):
        logger.info(f"Processing proxy file: {proxy_file.name}")

        with open(proxy_file) as f:
            proxy_data = json.load(f)

        judge = proxy_data["reference_evaluator"]
        reference = proxy_data["reference_evaluatee"]

        if judge_models is not None and judge not in judge_models:
            logger.info(f"Skipping judge not in specified models: {judge}")
            continue
        logger.info(f"Judge: {judge}")
        logger.info(f"Reference: {reference}")
        # Skip API models for judge
        if any(api in judge.lower() for api in ["claude", "gemini", "glm"]):
            logger.info(f"Skipping API judge: {judge}")
            continue

        for proxy_name, proxy_info in proxy_data["proxies"].items():
            # Skip API models for proxy
            if any(api in proxy_name.lower() for api in ["claude", "gpt", "gemini", "glm"]):
                continue

            # Get LSP and ILSP examples
            lsp_examples = []
            ilsp_examples = []

            for ex_id in proxy_info["lsp_ids"]:
                if ex_id in csv_data:
                    ex = csv_data[ex_id].copy()
                    # Add judge and reference responses
                    if any([f'{judge}_output_label' not in ex, f'{reference}_output_label' not in ex, f'{proxy_name}_output_label' not in ex]):
                        logger.warning(f"Missing output labels for example {ex_id}, skipping")
                        continue
                    ex['article'] = ex['text']
                    ex['question'] = ex['questions']                   
                    ex['judge'] = ex[f'{judge}_output_label'] + ". " + ex[f'{judge}_reason']
                    ex['reference'] = ex[f'{reference}_output_label'] + ". " + ex[f'{reference}_reason']
                    ex['proxy'] = ex[f'{proxy_name}_output_label'] + ". " + ex[f'{proxy_name}_reason']
                    lsp_examples.append(ex)
            for ex_id in proxy_info["ilsp_ids"]:
                if ex_id in csv_data:
                    ex = csv_data[ex_id].copy()
                    # Add judge and reference responses
                    if any([f'{judge}_output_label' not in ex, f'{reference}_output_label' not in ex, f'{proxy_name}_output_label' not in ex]):
                        logger.warning(f"Missing output labels for example {ex_id}, skipping")
                        continue
                    ex['article'] = ex['text']
                    ex['question'] = ex['questions']                   
                    ex['judge'] = ex[f'{judge}_output_label'] + ". " + ex[f'{judge}_reason']
                    ex['reference'] = ex[f'{reference}_output_label'] + ". " + ex[f'{reference}_reason']
                    ex['proxy'] = ex[f'{proxy_name}_output_label'] + ". " + ex[f'{proxy_name}_reason']
                    ilsp_examples.append(ex)

            if len(lsp_examples) > 0 and len(ilsp_examples) > 0:
                data_by_triplet[(judge, proxy_name, reference)] = {
                    'lsp': lsp_examples,
                    'ilsp': ilsp_examples,
                    'metadata': proxy_info
                }
    logger.info(f"Loaded {len(data_by_triplet)} (judge, proxy, reference) triplets")
    return data_by_triplet


# -------------------------
# --- MAIN EXECUTION ---
# -------------------------

def run_smoke_test(args, output_dir: Path, logger):
    """Run smoke test comparing HF vs Together API probabilities."""
    
    # Select small models (<10B parameters)
    if args.smoke_models:
        smoke_models = args.smoke_models
    else:
        # Auto-select small models from available data
        smoke_models = [
            "Qwen2.5-7B-Instruct-Turbo",  # ~7B
            "Meta-Llama-3.1-8B-Instruct-Turbo"  # ~8B
        ]
    
    logger.info(f"Running smoke test with models: {smoke_models}")
    
    # Load quality preference data from both ben and harmful JSONs
    preference_dir = Path(args.proxy_dir).parent / "preference_results"
    ben_file = preference_dir / "clean_pref_quality_ben.json"
    harmful_file = preference_dir / "clean_pref_quality_harmful.json"
    
    quality_data = []
    for json_file, dataset_tag in [(ben_file, "ben"), (harmful_file, "harmful")]:
        if json_file.exists():
            logger.info(f"Loading {dataset_tag} data from {json_file}")
            with open(json_file) as f:
                data = json.load(f)
                for ex in data:
                    ex['dataset'] = dataset_tag
                quality_data.extend(data)
        else:
            logger.warning(f"{dataset_tag} data file not found: {json_file}")
    
    logger.info(f"Loaded {len(quality_data)} examples from JSON files")
    
    # Debug: print keys of first example
    if quality_data:
        logger.info(f"First example keys: {list(quality_data[0].keys())}")
    
    # Cross-reference with quality_responses.csv for question text
    csv_file = Path(args.proxy_dir).parent.parent / "quality_responses.csv"
    csv_data = {}
    if csv_file.exists():
        logger.info(f"Loading question text from {csv_file}")
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_data[row['pid']] = row
    else:
        logger.warning(f"CSV file not found: {csv_file}, using JSON questions where available")

    # Build per-model example sets directly from quality_data where evaluator == model
    smoke_results = {}
    reproduction_validation = {}

    for model_name in smoke_models:
        logger.info(f"Collecting examples for evaluator model: {model_name}")
        # gather all examples (ben + harmful already concatenated) where evaluator==model_name
        model_examples = [ex.copy() for ex in quality_data if ex.get('evaluator') == model_name]
        logger.info(f"  Found {len(model_examples)} examples for {model_name}")

        # Attach question and responses from CSV (evaluator_response and evaluatee_response)
        enriched = []
        for ex in model_examples:
            pid = ex.get('pid')
            assert pid and pid in csv_data
            row = csv_data[pid]
            # Standard csv column names are like '{model}_reason'
            evaluator_resp = row.get(f"{ex.get('evaluator')}_reason") or row.get(f"{model_name}_reason") or ex.get(f"{ex.get('evaluator')}_response") or ''
            evaluatee_resp = row.get(f"{ex.get('evaluatee')}_reason") or ex.get(f"{ex.get('evaluatee')}_response") or ''
            ex['article'] = row['text'] if 'text' in row else ex['text']
            ex['question'] = row.get('questions') or ex.get('question')
            ex['evaluator_response'] = evaluator_resp
            ex['evaluatee_response'] = evaluatee_resp
            ex['evaluator_label'] = row[f"{ex.get('evaluator')}_output_label"]
            ex['evaluatee_label'] = row[f"{ex.get('evaluatee')}_output_label"]
            
            # only keep examples where both responses are present
            if ex['evaluator_response'] and ex['evaluatee_response'] and ex.get('question'):
                enriched.append(ex)

        model_examples = enriched
        logger.info(f"  {len(model_examples)} examples have both evaluator and evaluatee responses")

        if not model_examples:
            logger.warning(f"No usable examples for model {model_name}; skipping")
            continue

        # Select up to requested examples
        random.seed(42)
        selected = random.sample(model_examples, min(args.smoke_examples, len(model_examples)))
        logger.info(f"  Selected {len(selected)} examples for {model_name} smoke test")

        # Load HF model for this evaluator
        try:
            hf_path = get_hf_model_path(model_name)
            logger.info(f"Loading HF model from path: {hf_path}")
            hf_manager = AuthorObfuscationHFManager(hf_path, logger)
            logger.info(f"Model loaded: {hf_manager.model.config._name_or_path}")
        except Exception as e:
            logger.error(f"Failed to load HF model {model_name}: {e}")
            continue

        # Run examples and collect results; also write per-model JSONL
        model_results = []
        out_file = output_dir / f"{model_name.replace('/', '_')}_smoke_results.jsonl"
        with open(out_file, 'w', encoding='utf-8') as outf:
            for ex in selected:
                ex_id = ex.get('pid')
                judge_response = (ex.get('evaluator_response'), ex.get('evaluator_label'))
                reference_response = (ex.get('evaluatee_response'), ex.get('evaluatee_label'))

                # Forward
                messages_forward = format_preference_prompt(
                    article=ex['article'],
                    query=ex['question'],
                    response1=judge_response,
                    response2=reference_response,
                    dataset="quality",
                )
                hf_result_forward = hf_manager.prefer_generate_batch([messages_forward], temperature=0.0)[0]

                # Backward
                messages_backward = format_preference_prompt(
                    article=ex['article'],
                    query=ex['question'],
                    response1=reference_response,
                    response2=judge_response,
                    dataset="quality",
                )
                hf_result_backward = hf_manager.prefer_generate_batch([messages_backward], temperature=0.0)[0]

                avg_prob_J_over_R = (hf_result_forward['prob_A'] + hf_result_backward['prob_B']) / 2
                avg_prob_R_over_J = (hf_result_forward['prob_B'] + hf_result_backward['prob_A']) / 2

                result = {
                    'example_id': ex_id,
                    'dataset': ex.get('dataset'),
                    'evaluator': ex.get('evaluator'),
                    'evaluatee': ex.get('evaluatee'),
                    'hf_prob_J_over_R': avg_prob_J_over_R,
                    'hf_prob_R_over_J': avg_prob_R_over_J,
                    'forward_prob_A': hf_result_forward['prob_A'],
                    'forward_prob_B': hf_result_forward['prob_B'],
                    'backward_prob_A': hf_result_backward['prob_A'],
                    'backward_prob_B': hf_result_backward['prob_B'],
                    'forward_generated_text': hf_result_forward['generated_text'],
                    'backward_generated_text': hf_result_backward['generated_text'],
                    'forward_raw_probs': hf_result_forward['raw_probs'],
                    'backward_raw_probs': hf_result_backward['raw_probs'],
                    'json_forward_comparison': ex.get('forward_comparison'),
                    'json_forward_probability': ex.get('forward_probability'),
                    'json_backward_comparison': ex.get('backward_comparison'),
                    'json_backward_probability': ex.get('backward_probability'),
                    'json_self_preference': ex.get('self_preference'),
                    'judge_response': judge_response,
                    'reference_response': reference_response,
                    'question': ex.get('question')
                }

                model_results.append(result)
                outf.write(json.dumps(result) + "\n")

        # Store results and cleanup
        smoke_results[model_name] = [r.copy() for r in model_results]
        logger.info(f"Completed {len(model_results)} examples for {model_name}")
        logger.info(f"Saved per-model JSONL to {out_file}")
        if 'hf_manager' in locals():
            del hf_manager

        # Compute reproduction_validation for this model
        total_examples = len(model_results)
        forward_matches = sum(1 for r in model_results if str(r.get('forward_generated_text', '')) == str(r.get('json_forward_comparison', '')))
        backward_matches = sum(1 for r in model_results if str(r.get('backward_generated_text', '')) == str(r.get('json_backward_comparison', '')))
        both_match = sum(1 for r in model_results if (str(r.get('forward_generated_text', '')) == str(r.get('json_forward_comparison', '')) and str(r.get('backward_generated_text', '')) == str(r.get('json_backward_comparison', ''))))
        reproduction_validation[model_name] = {
            'total_examples': total_examples,
            'forward_matches': forward_matches,
            'backward_matches': backward_matches,
            'both_match': both_match,
            'forward_match_rate': forward_matches / total_examples if total_examples > 0 else 0,
            'backward_match_rate': backward_matches / total_examples if total_examples > 0 else 0,
            'both_match_rate': both_match / total_examples if total_examples > 0 else 0
        }
    
    # Save smoke test results with validation
    smoke_file = output_dir / "smoke_test_results.json"
    total_examples_all_models = sum(len(v) for v in smoke_results.values())
    with open(smoke_file, 'w') as f:
        json.dump({
            "metadata": {
                "models_tested": smoke_models,
                "n_examples": total_examples_all_models,
                "timestamp": datetime.now().isoformat(),
                "note": "Smoke test comparing HF probabilities with forward/backward averaging to reduce position bias"
            },
            "results": smoke_results,
            "reproduction_validation": reproduction_validation
        }, f, indent=2)
    
    logger.info(f"Smoke test results saved to {smoke_file}")
    
    # Log reproduction validation results
    logger.info("Reproduction validation (HF generated text vs JSON comparison):")
    for model_name, validation in reproduction_validation.items():
        logger.info(f"  {model_name}:")
        logger.info(f"    Forward matches: {validation['forward_matches']}/{validation['total_examples']} ({validation['forward_match_rate']:.1%})")
        logger.info(f"    Backward matches: {validation['backward_matches']}/{validation['total_examples']} ({validation['backward_match_rate']:.1%})")
        logger.info(f"    Both match: {validation['both_match']}/{validation['total_examples']} ({validation['both_match_rate']:.1%})")
    
    # Basic validation: check that probabilities are reasonable
    for model_name, results in smoke_results.items():
        if not results:
            continue
        
        probs_J_over_R = [r["hf_prob_J_over_R"] for r in results]
        probs_R_over_J = [r["hf_prob_R_over_J"] for r in results]
        
        logger.info(f"{model_name} stats:")
        logger.info(f"  Mean P(J>R): {sum(probs_J_over_R)/len(probs_J_over_R):.3f}")
        logger.info(f"  Mean P(R>J): {sum(probs_R_over_J)/len(probs_R_over_J):.3f}")
        logger.info(f"  Probabilities sum to ~1: {all(abs(a + b - 1.0) < 0.1 for a, b in zip(probs_J_over_R, probs_R_over_J))}")
    
    return smoke_results

def run_full_test(args, output_dir: Path, logger):
    """Run the full judge swap null hypothesis test."""
    # Load data
    proxy_dir = Path(args.proxy_dir)
    
    all_data = load_author_obfuscation_proxy_data(proxy_dir, logger, judge_models=args.judge_models)

    if not all_data:
        logger.error("No data loaded. Exiting.")
        return

    # Group by judge to minimize model switching
    by_judge = defaultdict(list)
    for (judge, proxy, ref), data in all_data.items():
        by_judge[judge].append((proxy, ref, data))

    logger.info(f"Processing {len(by_judge)} unique judges")

    # Process each judge
    for judge_name in sorted(by_judge.keys()):
        comparisons = by_judge[judge_name]
        logger.info(f"Processing Judge: {judge_name} ({len(comparisons)} proxy-reference pairs)")

        # Load judge model
        try:
            hf_path = get_hf_model_path(judge_name)
            judge_model = AuthorObfuscationHFManager(hf_path, logger)
        except Exception as e:
            logger.error(f"Failed to initialize judge model {judge_name}: {e}")
            continue

        # Process each proxy-reference pair
        for proxy_name, reference_name, data in comparisons:
            logger.info(f"  Processing: J={judge_name} | K={proxy_name} | R={reference_name}")

            # Combine LSP and ILSP examples
            all_examples = []
            for ex in data['lsp']:
                ex['category'] = 'lsp'
                all_examples.append(ex)
            for ex in data['ilsp']:
                ex['category'] = 'ilsp'
                all_examples.append(ex)

            logger.info(f"    Examples: {len(all_examples)} (LSP: {len(data['lsp'])}, ILSP: {len(data['ilsp'])})")

            # Run comparisons asynchronously
            asyncio.run(run_comparisons_async(
                judge_model, proxy_name, reference_name, all_examples,
                output_dir, judge_name, logger
            ))

        # Cleanup
        del judge_model

    logger.info("Judge swap null hypothesis test completed")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run judge swap null hypothesis test using author_obfuscation harness")
    parser.add_argument("--proxy_dir", type=str, default="author_obfuscation/data/quality/proxies",
                       help="Directory containing proxy JSON files")
    parser.add_argument("--output_dir", type=str, default="judge_swap_null_author_obfuscation",
                       help="Output directory for results")
    parser.add_argument("--judge_models", type=str, nargs="+",
                       help="Models to use for judge swap test (default: all available non-API models)")
    parser.add_argument("--smoke_test", action="store_true",
                       help="Run smoke test comparing HF vs Together API probabilities")
    parser.add_argument("--smoke_models", type=str, nargs="+",
                       help="Models to use for smoke test (default: small models <10B)")
    parser.add_argument("--smoke_examples", type=int, default=50,
                       help="Number of examples per model for smoke test")
    parser.add_argument("--debug", action="store_true",
                       help="Enable DEBUG level console logging and extra diagnostics")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, debug=args.debug)

    logger.info(f"Configuration:")
    logger.info(f"  Proxy dir: {args.proxy_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Smoke test: {args.smoke_test}")
    if args.smoke_test:
        logger.info(f"  Smoke models: {args.smoke_models or 'auto-select small models'}")
        logger.info(f"  Smoke examples: {args.smoke_examples}")

    if args.smoke_test:
        # Run smoke test
        run_smoke_test(args, output_dir, logger)
    else:
        # Run full judge swap test
        run_full_test(args, output_dir, logger)

    logger.info("Script completed")


async def run_comparisons_async(judge_model, proxy_name, reference_name, examples, output_dir, judge_name, logger):
    """Run the three comparison types asynchronously."""
    output_file = output_dir / "quality" / judge_name / f"judge_swap_{judge_name}_{proxy_name}_{reference_name}.json"
    if output_file.exists():
        logger.info(f"    Results already exist at {output_file}, skipping...")
        return
    # J vs R (Game 1: J=A, R=B)
    logger.info("    Running J vs R (J=A, R=B)...")
    j_vs_r_results = await run_comparison_batch(
        judge_model, examples, "judge", "reference", "J_vs_R_game1", judge_name
    )

    # J vs R (Game 2: R=A, J=B)
    logger.info("    Running J vs R (R=A, J=B)...")
    j_vs_r_g2_results = await run_comparison_batch(
        judge_model, examples, "reference", "judge", "J_vs_R_game2", judge_name
    )

    # K vs R (K=A, R=B)
    logger.info("    Running K vs R (K=A, R=B)...")
    k_vs_r_results = await run_comparison_batch(
        judge_model, examples, "proxy", "reference", "K_vs_R_game1", judge_name
    )

    logger.info("    Running K vs R (K=B, R=A)...")
    k_vs_r_g2_results = await run_comparison_batch(
        judge_model, examples, "reference", "proxy", "K_vs_R_game2", judge_name
    )

    # Save results
    results = {
        'metadata': {
            'judge': judge_name,
            'proxy': proxy_name,
            'reference': reference_name,
            'timestamp': datetime.now().isoformat(),
            'n_examples': len(examples)
        },
        'J_vs_R_game1': j_vs_r_results,
        'J_vs_R_game2': j_vs_r_g2_results,
        'K_vs_R_game1': k_vs_r_results,
        'K_vs_R_game2': k_vs_r_g2_results
    }

    output_file = output_dir / "quality" / judge_name / f"judge_swap_{judge_name}_{proxy_name}_{reference_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"    Results saved to {output_file}")


async def run_comparison_batch(judge_model, examples, response1_key, response2_key, comparison_type, judge_name):
    """Run batch comparison for a set of examples."""
    results = []

    for ex in examples:
        # Format messages for HF model
        messages = format_preference_prompt(
            article=ex['article'],
            query=ex['question'],
            response1=ex[response1_key],
            response2=ex[response2_key],
            dataset="quality"
        )
        
        # Run HF generation
        hf_result = judge_model.prefer_generate_batch([messages], temperature=0.0)[0]

        results.append({
            'example_id': ex['pid'],
            'category': ex.get('category', 'unknown'),
            'prob_A': hf_result['prob_A'],
            'prob_B': hf_result['prob_B'],
            'generated_text': hf_result['generated_text'],
            'raw_probs': hf_result['raw_probs'],
            'normalized_sum': hf_result['normalized_sum']
        })

    return results


def setup_logging(output_dir: Path, debug: bool = False) -> logging.Logger:
    """Setup logging (adapted from run_judge_swap_null_dbg.py)."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"judge_swap_author_obfuscation_{timestamp}.log"

    logger = logging.getLogger("judge_swap_author_obfuscation")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("JUDGE SWAP NULL HYPOTHESIS TEST - AUTHOR_OBFUSCATION HARNESS")
    logger.info("=" * 80)
    logger.info(f"User: {getpass.getuser()}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    return logger


if __name__ == "__main__":
    main()