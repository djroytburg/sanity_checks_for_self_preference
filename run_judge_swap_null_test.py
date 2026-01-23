# run_judge_swap_null_test.py: Generate preference probabilities for judge swap null hypothesis test
# Tests on DBG - H0: P(J chooses J over R) = P(J chooses K over R) where K is a proxy with similar capability
# Created: Dec 24, 2025, 03:30 EST
# Last Modified: Dec 24, 2025, 03:30 EST

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
from prompts.question_answering import (
    ALPACA_PREFERENCE_SYSTEM_PROMPT,
    ALPACA_PREFERENCE_USER_PROMPT,
    ALPACA_PREFERENCE_USER_PROMPT_ONLY
)
from prompts.translation import TRANSLATION_PREFERENCE_USER_PROMPT
from prompts.truthfulness import TRUTHFULNESS_PREFERENCE_USER_PROMPT
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
    Custom HuggingFace manager that supports batched preference generation
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
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if "gemma-2" in model_path.lower():
            self.tokenizer.padding_side = "right"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
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
            if self.is_instruct:
                # Use chat template for instruct models
                prompt = self.tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # For base models, just concatenate the message content
                # Typically just the user message for preference tasks
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
        Generate preferences for a batch of messages.
        
        Returns list of dicts with:
            - prob_A: Normalized probability of choosing A
            - prob_B: Normalized probability of choosing B  
            - generated_text: The actual output sequence
            - raw_probs: Unnormalized [prob_A, prob_B]
        """
        device = self.model.device
        
        # Format prompts
        prompts = self.format_messages(messages)
        
        # Tokenize with padding
        # For instruct models, don't add special tokens (already in chat template)
        # For base models, add special tokens
        # Don't truncate - we want the full prompt
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=(not self.is_instruct)
        ).to(device)
        
        # Get logits for next token prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        
        # Extract logits for the last position of each sequence
        batch_size = logits.shape[0]
        results = []
        
        for i in range(batch_size):
            # Find the last non-padding position
            # Handle both left and right padding by finding last non-pad token
            attention_mask_i = inputs.attention_mask[i]
            
            # Find the last position where attention_mask is 1
            non_padding_positions = torch.nonzero(attention_mask_i, as_tuple=False).squeeze()
            if non_padding_positions.dim() == 0:
                # Single non-padding token
                last_pos = non_padding_positions.item()
            else:
                # Multiple tokens - get the last one
                last_pos = non_padding_positions[-1].item()
            
            last_logits = logits[i, last_pos, :]
            
            # Debug logging for first batch
            if i < 2 and self.logger.level <= logging.DEBUG:
                self.logger.debug(f"Batch item {i}: last_pos={last_pos}, attention_mask sum={attention_mask_i.sum().item()}")
                last_token_id = inputs.input_ids[i, last_pos].item()
                self.logger.debug(f"  Last token ID: {last_token_id} = '{self.tokenizer.decode([last_token_id])}'")
            
            # Compute probabilities
            probs = F.softmax(last_logits, dim=-1)
            
            # Get probabilities for A and B tokens
            token_A = self.tokenizer.convert_tokens_to_ids("A")
            token_B = self.tokenizer.convert_tokens_to_ids("B")
            
            prob_A = probs[token_A].item()
            prob_B = probs[token_B].item()
            
            # Normalize
            total = prob_A + prob_B
            if total > 0:
                prob_A_norm = prob_A / total
                prob_B_norm = prob_B / total
            else:
                self.logger.warning(f"Zero probability sum for batch item {i}")
                prob_A_norm, prob_B_norm = 0.5, 0.5
            
            # Generate actual output to verify the model is doing the task correctly
            # This is a SMELL TEST - if it generates garbage, the prompt is wrong
            with torch.no_grad():
                gen_outputs = self.model.generate(
                    input_ids=inputs.input_ids[i:i+1],
                    attention_mask=inputs.attention_mask[i:i+1],
                    max_new_tokens=10,  # Allow some tokens to see what it generates
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False
                )
            
            # Decode ONLY the new tokens (not the prompt)
            seq_len_actual = inputs.input_ids[i].shape[0]
            generated_tokens = gen_outputs[0][seq_len_actual:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Check what the FIRST generated token actually is
            if len(generated_tokens) > 0:
                first_token_id = generated_tokens[0].item()
                first_token_decoded = self.tokenizer.decode([first_token_id])
                
                # Warn if the logits prediction doesn't match actual generation
                predicted_token_id = torch.argmax(probs).item()
                if first_token_id != predicted_token_id:
                    self.logger.debug(f"Logits predict token {predicted_token_id} ({self.tokenizer.decode([predicted_token_id])}), "
                                     f"but generation produced token {first_token_id} ({first_token_decoded})")
            
            # Sanity check: should start with A or B
            first_char = generated_text[0] if generated_text else ""
            if first_char not in ["A", "B"]:
                self.logger.warning(f"SMELL TEST FAILED for batch item {i}: Generated '{generated_text[:50]}...' instead of A or B")
                # Save full prompt to file for debugging
                debug_file = Path("judge_swap_null_results") / "debug_failed_prompts.txt"
                debug_file.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_file, "a") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"FAILED GENERATION\n")
                    f.write(f"Generated: {generated_text[:200]}\n")
                    f.write(f"Prob_A: {prob_A:.6e}, Prob_B: {prob_B:.6e}\n")
                    f.write(f"First token ID: {first_token_id if 'first_token_id' in locals() else 'N/A'}\n")
                    f.write(f"Predicted token ID: {predicted_token_id if 'predicted_token_id' in locals() else 'N/A'}\n")
                    f.write(f"\nFull prompt:\n{prompts[i]}\n")
                    f.write(f"{'='*80}\n")
                self.logger.warning(f"  Full prompt saved to {debug_file}")
            
            results.append({
                "prob_A": prob_A_norm,
                "prob_B": prob_B_norm,
                "generated_text": generated_text,
                "raw_probs": [prob_A, prob_B],
                "normalized_sum": prob_A_norm + prob_B_norm
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
    log_file = log_dir / f"judge_swap_null_test_{timestamp}.log"
    
    logger = logging.getLogger("judge_swap_null_test")
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
    logger.info("JUDGE SWAP NULL HYPOTHESIS TEST - RUN STARTED")
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
    Load proxy preference metadata from JSON files.
    New format: One file per judge-reference pair containing all proxies.
    
    Returns:
        Dict mapping (judge, proxy, reference) -> {
            'lsp': [...examples...],
            'ilsp': [...examples...],
            'metadata': {...}
        }
    """
    dataset_dir = proxy_data_dir / dataset
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return {}
    
    data_by_triplet = {}
    skipped_models = ["claude-3.5-haiku", "qwen-plus", "glm-4-plus"]
    
    for json_file in dataset_dir.glob("judge_*.json"):
        # Skip excluded models
        if any(skip in json_file.name for skip in skipped_models):
            logger.info(f"Skipping excluded file: {json_file.name}")
            continue
        
        try:
            with open(json_file) as f:
                file_data = json.load(f)
            
            judge = file_data["judge_name"]
            reference = file_data["reference_name"]
            
            # Skip API models
            if any(api in judge.lower() for api in ["gpt", "gemini", "deepseek", "claude", "qwen-plus", "glm"]):
                logger.info(f"Skipping API model judge: {judge}")
                continue
            
            # Skip base models (only instruct models can judge)
            is_base_model = not ("Instruct" in judge or "-it" in judge or "-chat" in judge)
            if is_base_model:
                logger.info(f"Skipping base model judge: {judge} (base models cannot judge)")
                continue
            
            # Skip very large models (>100B params) to save memory
            if any(large in judge for large in ["70B", "72B"]):
                logger.info(f"Skipping large model: {judge}")
                continue
            
            # Process all proxies in this file
            for proxy_name, proxy_data in file_data["proxies"].items():
                key = (judge, proxy_name, reference)
                data_by_triplet[key] = {
                    'lsp': proxy_data['data']['lsp'],
                    'ilsp': proxy_data['data']['ilsp'],
                    'metadata': {
                        'n_agree': proxy_data['n_agree'],
                        'n_total': proxy_data['n_total'],
                        'agreement_rate': proxy_data['agreement_rate'],
                        'dataset': file_data['dataset']
                    }
                }
                
                logger.debug(f"Loaded {judge} vs {proxy_name} vs {reference}: "
                            f"{len(proxy_data['data']['lsp'])} LSP, "
                            f"{len(proxy_data['data']['ilsp'])} ILSP")
        
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(data_by_triplet)} (judge, proxy, reference) triplets")
    return data_by_triplet


# -------------------------
# --- PREFERENCE GENERATION ---
# -------------------------

def format_preference_prompt(
    query: str,
    response1: str,
    response2: str,
    dataset: str,
    model_name: str
) -> List[Dict]:
    """
    Format a preference comparison prompt.
    
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
            system_prompt = ALPACA_PREFERENCE_SYSTEM_PROMPT
            user_prompt = ALPACA_PREFERENCE_USER_PROMPT.format(
                query=query,
                response1=response1,
                response2=response2
            )
        else:
            system_prompt = None
            user_prompt = ALPACA_PREFERENCE_USER_PROMPT_ONLY.format(
                query=query,
                response1=response1,
                response2=response2
            )
    
    elif dataset == "translation":
        system_prompt = None
        user_prompt = TRANSLATION_PREFERENCE_USER_PROMPT.format(
            german=query,  # For translation, query is German text
            english1=response1,
            english2=response2
        )
    
    elif dataset == "truthfulness":
        system_prompt = None
        user_prompt = TRUTHFULNESS_PREFERENCE_USER_PROMPT.format(
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


def generate_preferences_for_comparison(
    examples: List[Dict],
    judge_model_manager: BatchHFManager,
    response1_key: str,
    response2_key: str,
    comparison_type: str,
    cache_file: Path,
    batch_size: int,
    dataset: str,
    judge_name: str,
    proxy_name: str,
    reference_name: str,
    logger: logging.Logger,
    model_for_cache_key: str = None  # For K vs R, this is proxy_name to allow multiple K in same file
) -> List[Dict]:
    """
    Generate preference probabilities for a set of examples with comprehensive caching.
    Cache key is (example_id, model_for_cache_key) for K vs R to allow multiple proxies per file.
    Cache key is just example_id for J vs R (only one judge).
    """
    # Load existing cache
    cached_results = {}
    if cache_file.exists():
        logger.info(f"Loading cache from {cache_file}")
        with open(cache_file) as f:
            for line in f:
                item = json.loads(line)
                # Key by (example_id, model) if model_for_cache_key specified, else just example_id
                if model_for_cache_key:
                    cache_key = (item['example_id'], item.get('proxy', ''))
                else:
                    cache_key = item['example_id']
                cached_results[cache_key] = item
        logger.info(f"Loaded {len(cached_results)} cached results")
    
    # Filter uncached based on cache key structure
    if model_for_cache_key:
        uncached_examples = [ex for ex in examples if (ex['id'], model_for_cache_key) not in cached_results]
    else:
        uncached_examples = [ex for ex in examples if ex['id'] not in cached_results]
    
    if not uncached_examples:
        logger.info(f"All examples cached for {comparison_type}")
        # Return only results matching this model_for_cache_key
        if model_for_cache_key:
            return [v for k, v in cached_results.items() if isinstance(k, tuple) and k[1] == model_for_cache_key]
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
        
        # Format messages for batch
        batch_messages = []
        for ex in batch_examples:
            msg = format_preference_prompt(
                query=ex['query'],
                response1=ex[response1_key],
                response2=ex[response2_key],
                dataset=dataset,
                model_name=judge_name
            )
            batch_messages.append(msg)
        
        # Generate preferences
        batch_outputs = judge_model_manager.prefer_generate_batch(
            messages=batch_messages,
            max_tokens=4,
            temperature=0.0
        )
        
        # Save results with comprehensive metadata
        for ex, output in zip(batch_examples, batch_outputs):
            # Determine which category this example belongs to
            category = "lsp" if ex.get("is_lsp", True) else "ilsp"
            
            # Build cache key: (example_id, model_name) for K vs R, just example_id for J vs R
            if model_for_cache_key is not None:
                cache_key = f"{ex['id']}||{model_for_cache_key}"
            else:
                cache_key = ex['id']
            
            result = {
                "cache_key": cache_key,
                "example_id": ex['id'],
                "judge": judge_name,
                "reference": reference_name,
                "comparison_type": comparison_type,
                "response1_key": response1_key,
                "response2_key": response2_key,
                "prob_response1": output['prob_A'],
                "prob_response2": output['prob_B'],
                "generated_text": output['generated_text'],
                "raw_probs": output['raw_probs'],
                "normalized_sum": output['normalized_sum'],
                "category": category,
                "dataset": dataset
            }
            
            # Only include proxy name for K vs R comparisons (when model_for_cache_key is set)
            if model_for_cache_key is not None:
                result["proxy"] = model_for_cache_key
            
            results.append(result)
            
            # Append to cache immediately
            with open(cache_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
    
    # Combine with cached
    all_results = list(cached_results.values()) + results
    logger.info(f"Total results for {comparison_type}: {len(all_results)}")
    
    return all_results


# -------------------------
# --- MAIN EXECUTION ---
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Run judge swap null hypothesis test")
    parser.add_argument("--proxy_data_dir", type=str, default="dbg-score-paper/proxy_preference_data",
                       help="Directory containing proxy preference metadata")
    parser.add_argument("--dataset", type=str, default="alpaca_eval",
                       choices=["alpaca_eval", "translation", "truthfulness"],
                       help="Dataset to process")
    parser.add_argument("--output_dir", type=str, default="judge_swap_null_results",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for preference generation")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum examples to process per comparison (for testing)")
    args = parser.parse_args()
    
    # Load HF token
    load_dotenv(Path(__file__).parent / ".env")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    
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
    for (judge, proxy, ref), data in all_data.items():
        by_judge[judge].append((proxy, ref, data))
    
    logger.info(f"Processing {len(by_judge)} unique judges")
    
    # Process each judge
    for judge_name in sorted(by_judge.keys()):
        comparisons = by_judge[judge_name]
        logger.info("=" * 80)
        logger.info(f"Processing Judge: {judge_name}")
        logger.info(f"  {len(comparisons)} proxy-reference pairs")
        logger.info("=" * 80)
        
        # Load judge model once
        try:
            judge_hf_path = get_hf_model_path(judge_name)
            judge_model = BatchHFManager(judge_hf_path, logger)
        except Exception as e:
            logger.error(f"Failed to load judge model {judge_name}: {e}")
            continue
        
        # Group by reference to ensure consistent example sampling across proxies
        by_reference = defaultdict(list)
        for proxy_name, reference_name, data in comparisons:
            by_reference[reference_name].append((proxy_name, data))
        
        # Process each reference
        for reference_name, proxy_data_list in by_reference.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Reference: {reference_name} ({len(proxy_data_list)} proxies)")
            logger.info(f"{'='*80}")
            
            # Collect ALL unique example IDs across all proxies for this judge-reference pair
            all_example_ids = set()
            for proxy_name, data in proxy_data_list:
                for ex in data['lsp'] + data['ilsp']:
                    all_example_ids.add(ex['id'])
            
            # Apply max_examples limit to the GLOBAL set of example IDs
            if args.max_examples and len(all_example_ids) > args.max_examples:
                limited_ids = set(sorted(all_example_ids)[:args.max_examples])
                logger.info(f"  Limiting from {len(all_example_ids)} to {args.max_examples} unique examples (shared across all proxies)")
            else:
                limited_ids = all_example_ids
                logger.info(f"  Using all {len(limited_ids)} unique examples across proxies")
            
            # Process each proxy with the consistent example set
            for proxy_name, data in proxy_data_list:
                logger.info(f"\nProcessing: {judge_name} | Proxy={proxy_name} | Ref={reference_name}")
                
                # Combine LSP and ILSP, mark category
                all_examples = []
                for ex in data['lsp']:
                    ex['is_lsp'] = True
                    all_examples.append(ex)
                for ex in data['ilsp']:
                    ex['is_lsp'] = False
                    all_examples.append(ex)
                
                # Filter to only the limited_ids (consistent across all proxies for this judge-ref pair)
                all_examples = [ex for ex in all_examples if ex['id'] in limited_ids]
                
                logger.info(f"  Total examples for this proxy: {len(all_examples)} (after global limit)")
                logger.info(f"  LSP: {sum(1 for ex in all_examples if ex.get('is_lsp'))}, ILSP: {sum(1 for ex in all_examples if not ex.get('is_lsp'))}")
                
                # Cache directory: cache/Judge/Reference/ (same for all proxies of this judge-ref pair)
                cache_dir = output_dir / "cache" / judge_name / reference_name
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # J(J vs R) - Game 1: J=A, R=B (no model key - only one judge)
                logger.info("  Generating J(J vs R) - Game 1 (J=A, R=B)...")
                j_vs_r_g1 = generate_preferences_for_comparison(
                    examples=all_examples,
                    judge_model_manager=judge_model,
                    response1_key="judge",
                    response2_key="reference",
                    comparison_type="J_vs_R_game1",
                    cache_file=cache_dir / "J_vs_R_game1.jsonl",
                    batch_size=args.batch_size,
                    dataset=args.dataset,
                    judge_name=judge_name,
                    proxy_name=proxy_name,
                    reference_name=reference_name,
                    logger=logger,
                    model_for_cache_key=None  # No model key - only one judge
                )
                
                # J(J vs R) - Game 2: R=A, J=B (no model key - only one judge)
                logger.info("  Generating J(J vs R) - Game 2 (R=A, J=B)...")
                j_vs_r_g2 = generate_preferences_for_comparison(
                    examples=all_examples,
                    judge_model_manager=judge_model,
                    response1_key="reference",
                    response2_key="judge",
                    comparison_type="J_vs_R_game2",
                    cache_file=cache_dir / "J_vs_R_game2.jsonl",
                    batch_size=args.batch_size,
                    dataset=args.dataset,
                    judge_name=judge_name,
                    proxy_name=proxy_name,
                    reference_name=reference_name,
                    logger=logger,
                    model_for_cache_key=None  # No model key - only one judge
                )
                
                # J(K vs R) - Game 1: K=A, R=B (use proxy_name as cache key - multiple K in same file)
                logger.info("  Generating J(K vs R) - Game 1 (K=A, R=B)...")
                k_vs_r_g1 = generate_preferences_for_comparison(
                    examples=all_examples,
                    judge_model_manager=judge_model,
                    response1_key="proxy",
                    response2_key="reference",
                    comparison_type="K_vs_R_game1",
                    cache_file=cache_dir / "K_vs_R_game1.jsonl",
                    batch_size=args.batch_size,
                    dataset=args.dataset,
                    judge_name=judge_name,
                    proxy_name=proxy_name,
                    reference_name=reference_name,
                    logger=logger,
                    model_for_cache_key=proxy_name  # Use proxy name as key - multiple K in same file
                )
                
                # J(K vs R) - Game 2: R=A, K=B (use proxy_name as cache key - multiple K in same file)
                logger.info("  Generating J(K vs R) - Game 2 (R=A, K=B)...")
                k_vs_r_g2 = generate_preferences_for_comparison(
                    examples=all_examples,
                    judge_model_manager=judge_model,
                    response1_key="reference",
                    response2_key="proxy",
                    comparison_type="K_vs_R_game2",
                    cache_file=cache_dir / "K_vs_R_game2.jsonl",
                    batch_size=args.batch_size,
                    dataset=args.dataset,
                    judge_name=judge_name,
                    proxy_name=proxy_name,
                    reference_name=reference_name,
                    logger=logger,
                    model_for_cache_key=proxy_name  # Use proxy name as key - multiple K in same file
                )
                
                logger.info(f"  ✓ Completed: {proxy_name} vs {reference_name}")
        
        # Cleanup
        del judge_model
        torch.cuda.empty_cache()
        logger.info(f"✓ Completed all comparisons for judge: {judge_name}")
    
    logger.info("=" * 80)
    logger.info("JUDGE SWAP NULL HYPOTHESIS TEST COMPLETED")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
