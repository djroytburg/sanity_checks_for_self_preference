# run_self-rec_null_dbg.py: Generate self-recognition probabilities for correlation with self-preference
# Tests H0: Corr(J_pref, J_rec) = 0 - no correlation between preference and recognition
# Created: Dec 24, 2025, 17:15 EST
# Last Modified: Dec 24, 2025, 17:15 EST

import argparse
import json
import logging
import os
import sys
import getpass
import socket
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

# Add dbg-score-paper to path for imports
sys.path.insert(0, str(Path(__file__).parent / "dbg-score-paper"))
from prompts.self_recognition import (
    ALPACA_SELF_REC_SYSTEM_PROMPT,
    ALPACA_SELF_REC_USER_PROMPT,
    ALPACA_SELF_REC_USER_PROMPT_ONLY,
    TRANSLATION_SELF_REC_USER_PROMPT,
    TRUTHFULNESS_SELF_REC_USER_PROMPT
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F


# -------------------------
# --- MODEL NAME MAPPING ---
# -------------------------

def get_hf_model_path(model_name: str) -> str:
    """
    Map short model names to HuggingFace paths.
    
    Args:
        model_name (str): Short model name from proxy_preference_data files
    
    Returns:
        str: Full HuggingFace model path
    
    Raises:
        ValueError: If model name format is unknown
    """
    # Handle Llama models
    if "Llama" in model_name:
        return f"meta-llama/{model_name}"
    
    # Handle Qwen models (but not QwQ or DeepSeek)
    elif "Qwen2.5" in model_name or ("Qwen" in model_name and "QwQ" not in model_name and "DeepSeek" not in model_name):
        return f"Qwen/{model_name}"
    
    # Handle Gemma models
    elif "gemma" in model_name:
        return f"google/{model_name}"
    
    # Handle DeepSeek models (special case - uses deepseek-ai org)
    elif "DeepSeek-R1-Distill" in model_name:
        return f"deepseek-ai/{model_name}"
    
    # Handle QwQ models (special case - adds -Preview suffix)
    elif "QwQ" in model_name:
        return f"Qwen/{model_name}-Preview"
    
    # Skip API models (should be filtered earlier, but safety check)
    elif any(api in model_name.lower() for api in ["claude", "gpt", "gemini", "glm", "qwen-plus"]):
        raise ValueError(f"API model not supported for local loading: {model_name}")
    
    else:
        raise ValueError(f"Unknown model name format: {model_name}")


# -------------------------
# --- CUSTOM HF MANAGER ---
# -------------------------

class BatchHFManager:
    """
    Custom HuggingFace manager that supports batched self-recognition generation
    and returns both probabilities and generated output sequences.
    Handles both base and instruct models.
    """
    
    def __init__(self, model_path: str, logger: logging.Logger):
        """Initialize model and tokenizer."""
        self.model_path = model_path
        self.logger = logger
        
        # Determine if this is an instruct model
        self.is_instruct = "Instruct" in model_path or "-it" in model_path or "-chat" in model_path
        
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Model type: {'instruct' if self.is_instruct else 'base'}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        if "gemma-2" in model_path.lower():
            self.tokenizer.padding_side = "right"
        
        # Load model - use explicit device instead of device_map='auto' 
        # to avoid device mismatch issues with .to(device) calls
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Move to GPU explicitly
        if torch.cuda.is_available():
            self.model = self.model.to('cuda:0')
        
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.model.device}")
    
    def format_messages(self, messages: List[List[Dict]]) -> List[str]:
        """
        Format chat messages into prompts.
        For instruct models, use chat template. For base models, concatenate content.
        """
        prompts = []
        for msg in messages:
            if self.is_instruct:
                # Use chat template for instruct models
                prompt = self.tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # For base models, concatenate message content
                prompt = ""
                for m in msg:
                    if m["role"] == "system":
                        prompt += m["content"] + "\n\n"
                    elif m["role"] == "user":
                        prompt += m["content"]
            prompts.append(prompt)
        return prompts
    
    def prefer_generate_batch(
        self,
        messages: List[List[Dict]],
        max_tokens: int = 4,
        temperature: float = 0.0
    ) -> List[Dict]:
        """
        Generate self-recognition probabilities for a batch of messages.
        
        Returns list of dicts with:
            - prob_A: Normalized probability of choosing A
            - prob_B: Normalized probability of choosing B  
            - generated_text: The actual output sequence
            - raw_probs: Unnormalized [prob_A, prob_B]
        """
        device = self.model.device
        results = []
        
        # Process one at a time (no batching like original code)
        for msg in messages:
            # Format prompt
            prompts = self.format_messages([msg])
            prompt = prompts[0]
            
            # Tokenize - always use add_special_tokens=False because
            # apply_chat_template already adds them for instruct models
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            
            # Get logits for next token prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            
            # Get logits at last position (following judge swap pattern)
            last_pos = inputs.input_ids.shape[1] - 1
            last_logits = logits[0, last_pos, :]
            probs = F.softmax(last_logits, dim=-1)
            
            # Get probabilities for A and B tokens
            token_A = self.tokenizer.convert_tokens_to_ids("A")
            token_B = self.tokenizer.convert_tokens_to_ids("B")
            
            prob_a = probs[token_A].item()
            prob_b = probs[token_B].item()
            
            # Normalize
            total = prob_a + prob_b
            if total > 0:
                prob_a_norm = prob_a / total
                prob_b_norm = prob_b / total
            else:
                self.logger.warning(f"Zero probability sum")
                prob_a_norm, prob_b_norm = 0.5, 0.5
            
            # Generate actual output to verify (optional, for debugging)
            with torch.no_grad():
                gen_outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=5,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False
                )
            
            # Decode only the new tokens
            generated_tokens = gen_outputs[0][inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            results.append({
                "prob_A": prob_a_norm,
                "prob_B": prob_b_norm,
                "generated_text": generated_text,
                "raw_probs": [prob_a, prob_b],
                "normalized_sum": prob_a_norm + prob_b_norm
            })
        
        return results
    
    def __del__(self):
        """Cleanup GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()


# -------------------------
# --- LOGGING SETUP ---
# -------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up comprehensive logging with metadata."""
    log_dir = output_dir / "file_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"self_rec_null_dbg_{timestamp}.log"
    
    logger = logging.getLogger("self_rec_null_dbg")
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("=" * 80)
    logger.info("SELF-RECOGNITION NULL HYPOTHESIS TEST - RUN STARTED")
    logger.info("=" * 80)
    logger.info(f"User: {getpass.getuser()}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    
    return logger


# -------------------------
# --- DATA LOADING ---
# -------------------------

def load_proxy_data(
    proxy_data_dir: Path,
    dataset: str,
    logger: logging.Logger
) -> Dict:
    """
    Load proxy preference metadata from JSON files (to get examples with judge responses).
    We only need judge-reference pairs (not proxies) for self-recognition test.
    
    Returns:
        Dict mapping (judge, reference) -> {
            'lsp': [...examples...],
            'ilsp': [...examples...],
            'metadata': {...}
        }
    """
    dataset_dir = proxy_data_dir / dataset
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return {}
    
    data_by_pair = {}
    skipped_models = ["claude-3.5-haiku", "qwen-plus", "glm-4-plus"]
    
    for json_file in dataset_dir.glob("judge_*.json"):
        # Skip excluded models
        if any(skip in json_file.name for skip in skipped_models):
            logger.info(f"Skipping API model: {json_file.name}")
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            judge_name = data['judge_name']
            reference_name = data['reference_name']
            
            # Skip base models as judges (they can't follow instructions well)
            if not any(marker in judge_name for marker in ["Instruct", "-it", "-chat"]):
                logger.info(f"Skipping base model judge: {judge_name}")
                continue
            
            # Skip very large models (memory constraints)
            if any(large in judge_name for large in ["70B", "72B", "QwQ-32B", "DeepSeek-R1-Distill"]):
                logger.info(f"Skipping large model: {judge_name}")
                continue
            
            # Collect all examples from all proxies (we just need examples with judge responses)
            all_lsp = []
            all_ilsp = []
            seen_ids = set()
            
            for proxy_name, proxy_data in data['proxies'].items():
                for ex in proxy_data['data']['lsp']:
                    if ex['id'] not in seen_ids:
                        all_lsp.append(ex)
                        seen_ids.add(ex['id'])
                
                for ex in proxy_data['data']['ilsp']:
                    if ex['id'] not in seen_ids:
                        all_ilsp.append(ex)
                        seen_ids.add(ex['id'])
            
            if all_lsp or all_ilsp:
                data_by_pair[(judge_name, reference_name)] = {
                    'lsp': all_lsp,
                    'ilsp': all_ilsp,
                    'metadata': {
                        'judge': judge_name,
                        'reference': reference_name,
                        'dataset': dataset
                    }
                }
                logger.debug(f"Loaded {judge_name} vs {reference_name}: {len(all_lsp)} LSP, {len(all_ilsp)} ILSP")
        
        except Exception as e:
            logger.error(f"Error loading {json_file.name}: {e}")
            continue
    
    logger.info(f"Loaded {len(data_by_pair)} (judge, reference) pairs")
    return data_by_pair


# -------------------------
# --- RECOGNITION GENERATION ---
# -------------------------

def format_self_recognition_prompt(
    query: str,
    response1: str,
    response2: str,
    dataset: str,
    model_name: str
) -> List[Dict]:
    """
    Format a self-recognition prompt.
    
    Returns:
        List of message dicts for chat template
    """
    # Determine if model uses system prompt
    uses_system_prompt = not (
        "gemma-2-9b-it" in model_name.lower() or
        "-sft-UC" in model_name or
        "DeepSeek-R1-Distill" in model_name
    )
    
    # Select prompt template based on dataset
    if dataset == "alpaca_eval":
        if uses_system_prompt:
            system_prompt = ALPACA_SELF_REC_SYSTEM_PROMPT
            user_prompt = ALPACA_SELF_REC_USER_PROMPT.format(
                query=query,
                response1=response1,
                response2=response2
            )
        else:
            system_prompt = None
            user_prompt = ALPACA_SELF_REC_USER_PROMPT_ONLY.format(
                query=query,
                response1=response1,
                response2=response2
            )
    
    elif dataset == "translation":
        system_prompt = None
        user_prompt = TRANSLATION_SELF_REC_USER_PROMPT.format(
            german=query,
            english1=response1,
            english2=response2
        )
    
    elif dataset == "truthfulness":
        system_prompt = None
        user_prompt = TRUTHFULNESS_SELF_REC_USER_PROMPT.format(
            query=query,
            response1=response1,
            response2=response2
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Build message list
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        messages = [{"role": "user", "content": user_prompt}]
    
    return messages


def generate_recognition_for_comparison(
    examples: List[Dict],
    judge_model_manager: BatchHFManager,
    response1_key: str,
    response2_key: str,
    comparison_type: str,
    cache_file: Path,
    batch_size: int,
    dataset: str,
    judge_name: str,
    reference_name: str,
    logger: logging.Logger
) -> List[Dict]:
    """
    Generate self-recognition probabilities for a set of examples with comprehensive caching.
    
    Args:
        response1_key: Which model's response is in position 1 (e.g., 'judge' or 'reference')
        response2_key: Which model's response is in position 2
        comparison_type: 'J_rec_game1' or 'J_rec_game2' for cache identification
    """
    # Load existing cache
    cached_results = {}
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    cache_key = item.get('cache_key', item.get('example_id'))
                    cached_results[cache_key] = item
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded {len(cached_results)} cached results from {cache_file.name}")
    
    # Filter uncached examples
    uncached_examples = [ex for ex in examples if ex['id'] not in cached_results]
    
    if not uncached_examples:
        logger.info(f"All {len(examples)} examples already cached for {comparison_type}")
        return list(cached_results.values())
    
    logger.info(f"Generating {len(uncached_examples)} uncached examples for {comparison_type}")
    
    # Prepare batches
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    results = []
    
    num_batches = (len(uncached_examples) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"{comparison_type}"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(uncached_examples))
        batch_examples = uncached_examples[start_idx:end_idx]
        
        # Format prompts
        batch_messages = []
        for ex in batch_examples:
            # Get responses based on keys
            response1 = ex[response1_key]
            response2 = ex[response2_key]
            
            # Get query based on dataset
            if dataset == "translation":
                query = ex.get('german', '')
            else:
                query = ex.get('query', '')
            
            messages = format_self_recognition_prompt(
                query=query,
                response1=response1,
                response2=response2,
                dataset=dataset,
                model_name=judge_name
            )
            batch_messages.append(messages)
        
        # Generate probabilities
        batch_results = judge_model_manager.prefer_generate_batch(
            messages=batch_messages,
            max_tokens=4,
            temperature=0.0
        )
        
        # Save to cache and collect
        with open(cache_file, 'a', encoding='utf-8') as f:
            for ex, result in zip(batch_examples, batch_results):
                cache_entry = {
                    'cache_key': ex['id'],
                    'example_id': ex['id'],
                    'judge': judge_name,
                    'reference': reference_name,
                    'comparison_type': comparison_type,
                    'query': ex.get('query', ex.get('german', '')),
                    'response_judge': ex['judge'],
                    'response_reference': ex['reference'],
                    'response1_key': response1_key,
                    'response2_key': response2_key,
                    'prob_first': result['prob_A'],
                    'prob_second': result['prob_B'],
                    'generated_text': result['generated_text'],
                    'raw_probs': result['raw_probs'],
                    'normalized_sum': result['normalized_sum']
                }
                f.write(json.dumps(cache_entry, ensure_ascii=False) + '\n')
                results.append(cache_entry)
    
    # Combine with cached
    all_results = list(cached_results.values()) + results
    logger.info(f"Total results for {comparison_type}: {len(all_results)}")
    
    return all_results


# -------------------------
# --- MAIN EXECUTION ---
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Run self-recognition null hypothesis test")
    parser.add_argument("--proxy_data_dir", type=str, default="dbg-score-paper/proxy_preference_data",
                       help="Directory containing proxy preference metadata")
    parser.add_argument("--dataset", type=str, default="alpaca_eval",
                       choices=["alpaca_eval", "translation", "truthfulness"],
                       help="Dataset to process")
    parser.add_argument("--output_dir", type=str, default="self_rec_null_dbg_results",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for recognition generation")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum examples to process per comparison (for testing)")
    args = parser.parse_args()
    
    # Load HF token
    load_dotenv(Path(__file__).parent / ".env")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    # Setup
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info(f"Configuration:")
    logger.info(f"  Proxy data dir: {args.proxy_data_dir}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max examples: {args.max_examples or 'unlimited'}")
    
    # Load data
    proxy_data_dir = Path(args.proxy_data_dir)
    all_data = load_proxy_data(proxy_data_dir, args.dataset, logger)
    
    if not all_data:
        logger.error("No data loaded. Exiting.")
        return
    
    # Group by judge to minimize model switching
    by_judge = defaultdict(list)
    for (judge, ref), data in all_data.items():
        by_judge[judge].append((ref, data))
    
    logger.info(f"Processing {len(by_judge)} unique judges")
    
    # Process each judge
    for judge_name in sorted(by_judge.keys()):
        logger.info("=" * 80)
        logger.info(f"PROCESSING JUDGE: {judge_name}")
        logger.info("=" * 80)
        
        # Load judge model once
        try:
            judge_path = get_hf_model_path(judge_name)
            judge_manager = BatchHFManager(judge_path, logger)
        except Exception as e:
            logger.error(f"Failed to load judge {judge_name}: {e}")
            continue
        
        # Process each reference for this judge
        for reference_name, data in sorted(by_judge[judge_name]):
            logger.info(f"\n--- Judge: {judge_name} vs Reference: {reference_name} ---")
            
            # Combine LSP and ILSP examples
            all_examples = data['lsp'] + data['ilsp']
            
            if args.max_examples:
                all_examples = all_examples[:args.max_examples]
            
            logger.info(f"Total examples: {len(all_examples)} ({len(data['lsp'])} LSP, {len(data['ilsp'])} ILSP)")
            
            # Setup cache directory
            cache_dir = output_dir / "cache" / judge_name / reference_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Game 1: Judge's response is first (position A)
            # Correct recognition = model says "A"
            logger.info("\n>>> Generating Game 1: Judge response in position A")
            game1_cache = cache_dir / "J_rec_game1.jsonl"
            game1_results = generate_recognition_for_comparison(
                examples=all_examples,
                judge_model_manager=judge_manager,
                response1_key='judge',
                response2_key='reference',
                comparison_type='J_rec_game1',
                cache_file=game1_cache,
                batch_size=args.batch_size,
                dataset=args.dataset,
                judge_name=judge_name,
                reference_name=reference_name,
                logger=logger
            )
            
            # Game 2: Judge's response is second (position B)
            # Correct recognition = model says "B"
            logger.info("\n>>> Generating Game 2: Judge response in position B")
            game2_cache = cache_dir / "J_rec_game2.jsonl"
            game2_results = generate_recognition_for_comparison(
                examples=all_examples,
                judge_model_manager=judge_manager,
                response1_key='reference',
                response2_key='judge',
                comparison_type='J_rec_game2',
                cache_file=game2_cache,
                batch_size=args.batch_size,
                dataset=args.dataset,
                judge_name=judge_name,
                reference_name=reference_name,
                logger=logger
            )
            
            logger.info(f"\n✓ Completed {judge_name} vs {reference_name}")
            logger.info(f"  Game 1: {len(game1_results)} results")
            logger.info(f"  Game 2: {len(game2_results)} results")
        
        # Cleanup judge model
        del judge_manager
        torch.cuda.empty_cache()
        logger.info(f"\n✓ Completed all references for judge {judge_name}\n")
    
    logger.info("=" * 80)
    logger.info("SELF-RECOGNITION NULL HYPOTHESIS TEST COMPLETED")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
