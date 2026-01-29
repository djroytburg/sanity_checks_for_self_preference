

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
            
            # Get logits at last position (following original code)
            last_pos = inputs.input_ids.shape[1] - 1
            last_logits = logits[0, last_pos, :]
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
                self.logger.warning(f"Zero probability sum")
                prob_A_norm, prob_B_norm = 0.5, 0.5
            
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
    log_file = log_dir / f"judge_swap_null_dbg_results_{timestamp}.log"
    
    logger = logging.getLogger("judge_swap_null_dbg_results")
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
    parser.add_argument("--output_dir", type=str, default="judge_swap_null_dbg",
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
        
        # Process each proxy-reference pair
        for proxy_name, reference_name, data in comparisons:
            logger.info(f"\nProcessing: {judge_name} | Proxy={proxy_name} | Ref={reference_name}")
            
            # Combine LSP and ILSP, mark category
            all_examples = []
            for ex in data['lsp']:
                ex['is_lsp'] = True
                all_examples.append(ex)
            for ex in data['ilsp']:
                ex['is_lsp'] = False
                all_examples.append(ex)
            
            logger.info(f"  Total examples: {len(all_examples)} (LSP: {len(data['lsp'])}, ILSP: {len(data['ilsp'])})")
            
            # Limit if specified
            if args.max_examples:
                all_examples = all_examples[:args.max_examples]
                logger.info(f"  Limited to: {len(all_examples)} examples")
            
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
