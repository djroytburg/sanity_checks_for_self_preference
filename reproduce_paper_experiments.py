#!/usr/bin/env python3
# reproduce_paper_experiments.py: Faithful Reproduction of Paper Experiments
# Written by: Dani
# Created: 2025-12-21 10:30 EST
# Last Modified: 2026-01-12
"""
This module reproduces the self-preference experiments from the paper:
- Verdict generation using vLLM for efficient inference
- Temperature control (0 for non-reasoning, 0.6 for reasoning models)
- Token-level probability extraction for A/B/T labels
- Support for CoT reasoning with verdict parsing

Key paper details:
- All non-reasoning models: temperature=0 (greedy decoding)
- All reasoning models (DeepSeek-R1-Distill): temperature=0.6
- Verdict extraction: Extract A/B/T token logprobs (not string matching)
- For reasoning models: Remove <think>...</think> tokens from responses
- vLLM used for all inference except OpenAI API models

Robust verdict parsing:
- Uses token-level logprob extraction (like run_arena_self_preference.py)
- NOT brittle string matching - properly handles cases like "The answer is B"
- Extracts first A/B/T token from model output via logprobs
- Normalizes token probabilities to get A/B/T probabilities

UNKNOWN HYPERPARAMETERS (to be filled in):
- [ ] Verdict parsing format for CoT (what text before final verdict?)
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import random
import matplotlib.pyplot as plt

load_dotenv()


# ######################
# ### CONFIGURATION  ###
# ######################

# Model configurations from the paper
MODEL_FAMILIES = {
    "llama": [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ],
    "gemma": [
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
    ],
    "qwen": [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
    ],
    "deepseek": [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ],
    "mistral": [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-Small-Instruct-2409",
    ],
    "phi": [
        "microsoft/Phi-3.5-mini-instruct",
    ],
}

# Reasoning models (use temperature=0.6, preserve post-</think> content)
REASONING_MODELS = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}

# ====== UNKNOWN HYPERPARAMETERS - FILL THESE IN ======
CONFIG = {
    # Response generation
    "response_temperature": 0.0,  # 0.0 for non-reasoning, 0.6 for reasoning
    "response_max_tokens": None,  # UNKNOWN - likely 2048 or model_max_length
    "response_top_p": 1.0,  # UNKNOWN
    
    # Verdict generation
    "verdict_temperature": 0.0,  # 0.0 for non-reasoning, 0.6 for reasoning
    "verdict_max_tokens": 1,  # 1 token for non-reasoning (direct A/B/T verdict)
    # For cot/long_cot modes: max_tokens=None -> generate until EOS or model max context
    "verdict_top_k": 20,  # For logprobs extraction
    
    # System prompt for verdict generation
    "verdict_system_prompt": "",
    
    # User prompt template for verdict generation
    "verdict_user_prompt_template": """
You will be given answers from two assistants (Assistant A and Assistant B) to a user prompt. Please act as
an impartial judge and evaluate the quality of their answers.
You must output only one of the following choices:
- “A” if Assistant A’s answer is better.
- “T” if both answers are relatively the same in quality.
- “B” if Assistant B’s answer is better.
For example, if both answers are relatively the same in quality, output “T”.
<User Prompt>
{question}
<The Start of Assistant A’s Answer>
{answer_1}
<The End of Assistant A’s Answer>
<The Start of Assistant B’s Answer>
{answer_2}
<The End of Assistant B’s Answer>
    """,
    
    # Verdict parsing for CoT
    "cot_verdict_format": """UNKNOWN: Format for CoT verdict parsing
    Example: "My final verdict is A"
    """,
    
    # vLLM configuration
    "vllm_dtype": "float16",  # or "bfloat16"
    "vllm_tensor_parallel_size": 1,  # UNKNOWN - depends on GPU setup
    "vllm_gpu_memory_utilization": 0.9,  # UNKNOWN
    "vllm_seed": 42,  # UNKNOWN - check if paper specifies random seed
    "vllm_block_size": 32,  # Use 32 to avoid FlashInfer bug with head_size=256 (gemma-2-9b)
    
    # Dataset configuration
    "mmlu_sample_size": 1000,  # Paper: "1K instances from MMLU test set"
    "mmlu_random_seed": 42,  # UNKNOWN
    
    # Evaluation configuration
    "remove_think_tokens": True,  # Remove <think>...</think> from reasoning models
    "think_open_token": "<think>",
    "think_close_token": "</think>",
}

# ====== END UNKNOWN HYPERPARAMETERS ======


# ######################
# ### EVALUATOR PROMPTS (from Appendix B) ###
# ######################

# Valid benchmark and reasoning mode options
VALID_BENCHMARKS = {"math500", "mmlu", "mbpp-plus"}
VALID_REASONING_MODES = {"none", "cot", "long_cot"}

# =============================================================================
# MATH500 PROMPTS (Tables 1, 2, 3)
# =============================================================================

MATH500_NO_REASONING_PROMPT = """You will be given answers from two assistants (Assistant A and Assistant B) to a user prompt. Please act as an impartial judge and evaluate the quality of their answers.

You must output only one of the following choices:

- “A” if Assistant A’s answer is better.
- “T” if both answers are relatively the same in quality.
- “B” if Assistant B’s answer is better.

For example, if both answers are relatively the same in quality, output “T”.

<User Prompt>
{question}

<The Start of Assistant A’s Answer>
{answer_a}
<The End of Assistant A’s Answer>

<The Start of Assistant B’s Answer>
{answer_b}
<The End of Assistant B’s Answer>"""


MATH500_COT_PROMPT = """You will be given answers from two assistants (Assistant A and Assistant B) to a user prompt. Please act as an impartial judge and evaluate the quality of their answers.

You must end your response with: "My final verdict is $$...$$.". where $$...$$ must enclose one of the following:

- “A” if Assistant A’s answer is better.
- “T” if both answers are relatively the same in quality.
- “B” if Assistant B’s answer is better.

For example, if your final verdict is a tie, end your response with: "My final verdict is $$T$$."

<User Prompt>
{question}

<The Start of Assistant A’s Answer>
{answer_a}
<The End of Assistant A’s Answer>

<The Start of Assistant B’s Answer>
{answer_b}
<The End of Assistant B’s Answer>"""

# Long CoT adds <think> at the end to trigger reasoning
MATH500_LONG_COT_PROMPT = MATH500_COT_PROMPT  # Same prompt, but we append <think> token

# =============================================================================
# MMLU PROMPTS (Tables 4, 5, 6)
# =============================================================================

# MMLU no-reasoning is same as MATH500 no-reasoning
MMLU_NO_REASONING_PROMPT = MATH500_NO_REASONING_PROMPT

MMLU_COT_PROMPT = """You are given answers provided by two assistants, Assistant A and Assistant B, to the following user prompt.

<User Prompt>
{question}

<The Start of Assistant A’s Answer>
{answer_a}
<The End of Assistant A’s Answer>

<The Start of Assistant B’s Answer>
{answer_b}
<The End of Assistant B’s Answer>

Please act as an impartial judge and evaluate the quality of their answers.

You must end your response with: "My final verdict is $$...$$.". where $$...$$ must enclose one of the following:

- “A” if Assistant A’s answer is better.
- “T” if both answers are relatively the same in quality.
- “B” if Assistant B’s answer is better.

For example, if your final verdict is a tie, end your response with: "My final verdict is $$T$$."""

# Long CoT is same prompt, but we append <think> token
MMLU_LONG_COT_PROMPT = MMLU_COT_PROMPT

# =============================================================================
# MBPP+ PROMPTS (Tables 7, 8, 9)
# =============================================================================

MBPP_PLUS_NO_REASONING_PROMPT = """Please act as an impartial judge and assess the accuracy of the given Python functions. Provide your evaluation by outputting only one of the following

- “A” if Assistant A’s function is more accurate.
- “T” if both functions are similarly accurate.
- “B” if Assistant B’s function is more accurate.

Base your judgment on your understanding of the task described in the user prompt and whether the functions successfully pass the provided test cases.

<User Prompt>
{question}

<The Start of Assistant A’s Function>
{answer_a}
<The End of Assistant A’s Function>

<The Start of Assistant B’s Function>
{answer_b}
<The End of Assistant B’s Function>"""


MBPP_PLUS_COT_PROMPT = """Please act as an impartial judge and assess the accuracy of the given Python functions. You must end your response with: "My final verdict is $$...$$.". where $$...$$ must enclose one of the following:

- “A” if Assistant A’s function is more accurate.
- “T” if both functions are similarly accurate.
- “B” if Assistant B’s function is more accurate.

For example, if your final verdict is a tie, end your response with: "My final verdict is $$T$$."
Base your judgment on your understanding of the task described in the user prompt and whether the functions successfully pass the provided test cases.

<User Prompt>
{question}

<The Start of Assistant A’s Function>
{answer_a}
<The End of Assistant A’s Function>

<The Start of Assistant B’s Function>
{answer_b}
<The End of Assistant B’s Function>"""

# Long CoT is same prompt, but we append <think> token
MBPP_PLUS_LONG_COT_PROMPT = MBPP_PLUS_COT_PROMPT

# =============================================================================
# PROMPT REGISTRY
# =============================================================================

EVALUATOR_PROMPTS = {
    "math500": {
        "none": MATH500_NO_REASONING_PROMPT,
        "cot": MATH500_COT_PROMPT,
        "long_cot": MATH500_LONG_COT_PROMPT,
    },
    "mmlu": {
        "none": MMLU_NO_REASONING_PROMPT,
        "cot": MMLU_COT_PROMPT,
        "long_cot": MMLU_LONG_COT_PROMPT,
    },
    "mbpp-plus": {
        "none": MBPP_PLUS_NO_REASONING_PROMPT,
        "cot": MBPP_PLUS_COT_PROMPT,
        "long_cot": MBPP_PLUS_LONG_COT_PROMPT,
    },
}

def get_evaluator_prompt(benchmark: str, reasoning_mode: str) -> str:
    """
    Get the appropriate evaluator prompt for a benchmark and reasoning mode.
    
    Args:
        benchmark: One of "math500", "mmlu", "mbpp-plus"
        reasoning_mode: One of "none", "cot", "long_cot"
    
    Returns:
        The prompt template string with {question}, {answer_a}, {answer_b} placeholders.
    
    Raises:
        ValueError: If benchmark or reasoning_mode is invalid.
    """
    benchmark = benchmark.lower()
    reasoning_mode = reasoning_mode.lower()
    
    if benchmark not in VALID_BENCHMARKS:
        raise ValueError(f"Invalid benchmark: {benchmark}. Must be one of {VALID_BENCHMARKS}")
    if reasoning_mode not in VALID_REASONING_MODES:
        raise ValueError(f"Invalid reasoning_mode: {reasoning_mode}. Must be one of {VALID_REASONING_MODES}")
    
    return EVALUATOR_PROMPTS[benchmark][reasoning_mode]

# ######################
# ### LOGGING SETUP  ###
# ######################

def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Set up comprehensive logging with file and console handlers.

    Args:
        output_dir (Path): Base output directory for results.

    Returns:
        logging.Logger: Configured logger instance with file and console handlers.

    The logger writes to file_logs/ with timestamps per project standards.
    Console output shows INFO level and above.
    File output shows DEBUG level and above.
    """
    log_dir = Path("file_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"reproduce_paper_experiments_{timestamp}.log"
    
    logger = logging.getLogger("paper_reproduction")
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("=" * 80)
    logger.info("PAPER REPRODUCTION PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")
    
    return logger


# ######################
# ### UTIL FUNCTIONS ###
# ######################

def _truncate(s: Optional[str], n: int = 200) -> str:
    """
    Truncate string for logging purposes.

    Args:
        s (Optional[str]): Input string to truncate.
        n (int): Maximum length before truncation (default: 200).

    Returns:
        str: Truncated string with ellipsis and character count if truncated.
    """
    if s is None:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"...<{len(s)} chars>"


def is_reasoning_model(model_id: str) -> bool:
    """
    Check if model is a reasoning model.

    Args:
        model_id (str): Model identifier string.

    Returns:
        bool: True if model uses temperature=0.6 (reasoning), False if temperature=0.0.
    """
    return model_id in REASONING_MODELS


def remove_thinking_tokens(text: str, config: Dict) -> str:
    """
    Remove <think>...</think> tokens from reasoning model responses.

    Per paper specifications: "we preserve only the responses after the </think> token"
    This function extracts content following the closing think token for reasoning models.

    Args:
        text (str): Response text potentially containing thinking tokens.
        config (Dict): Configuration dict with think_open_token and think_close_token.

    Returns:
        str: Text with thinking tokens removed, or original text if no thinking tokens present.

    Raises:
        None: Returns original text gracefully if tokens not found.
    """
    open_token = config["think_open_token"]
    close_token = config["think_close_token"]
    
    if open_token not in text or close_token not in text:
        return text
    
    start_idx = text.find(open_token)
    end_idx = text.find(close_token)
    
    if start_idx >= 0 and end_idx > start_idx:
        # Keep text after </think>
        return text[end_idx + len(close_token):].strip()
    
    return text


def decode_token(tokenizer, tid: int) -> str:
    """
    Decode a single token ID to its string representation.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        tid (int): Token ID to decode.

    Returns:
        str: Decoded token string, or empty string if decode fails.

    Raises:
        None: Returns empty string on exception.
    """
    try:
        return tokenizer.decode([tid], skip_special_tokens=False)
    except Exception:
        return ""


def logits_to_prob_map(
    logits_tensor,
    tokenizer,
    topk: int = 200,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Convert logits tensor to {decoded_token_text -> probability}.

    Uses log-softmax and extracts top-k token probabilities.
    This is robust verdict parsing that matches Arena script approach.

    Args:
        logits_tensor: Logits tensor from model output (shape: [vocab_size])
        tokenizer: HuggingFace tokenizer for decoding tokens.
        topk (int): Number of top tokens to extract (default: 200).
        logger (Optional[logging.Logger]): Logger for debug output.

    Returns:
        Dict[str, float]: Mapping of token text to probability (sums to 1.0 for top-k).

    Raises:
        None: Processes gracefully even if logits are malformed.
    """
    if logger is None:
        logger = logging.getLogger("paper_reproduction")
    
    text_to_prob: Dict[str, float] = {}
    
    logger.debug(f"logits_to_prob_map: input shape={logits_tensor.shape}, dtype={logits_tensor.dtype}")
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits_tensor, dim=-1)
    
    # Get top-k indices
    k = min(topk, logits_tensor.shape[-1])
    topk_logprobs, topk_indices = torch.topk(log_probs, k, dim=-1)
    
    # Convert to CPU and numpy for processing
    topk_logprobs_np = topk_logprobs.detach().cpu().numpy()
    topk_indices_np = topk_indices.detach().cpu().numpy()
    
    logger.debug(f"Top-k extraction: k={k}, top logprobs range=[{topk_logprobs_np.min():.4f}, {topk_logprobs_np.max():.4f}]")
    
    for i, (token_id, logprob) in enumerate(zip(topk_indices_np, topk_logprobs_np)):
        token_id = int(token_id)
        logprob = float(logprob)
        prob = math.exp(logprob)
        ttext = decode_token(tokenizer, token_id)
        text_to_prob[str(ttext)] = prob
        
        if i < 10:  # Log first 10 for debugging
            logger.debug(f"topk[{i}]: token_id={token_id}, text='{_truncate(ttext, 40)}', logprob={logprob:.4f}, prob={prob:.6f}")
    
    return text_to_prob


def only_abt_probs(
    text_prob_map: Dict[str, float],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Collect probabilities for A/B/T tokens from logprob map.

    Handles common token variants: bare letters, space-prefixed (▁), newline-prefixed, etc.
    This robust approach avoids brittle string matching.

    Args:
        text_prob_map (Dict[str, float]): Map of token text to probability.
        logger (Optional[logging.Logger]): Logger for debug output.

    Returns:
        Dict[str, float]: Dictionary with keys "A", "B", "T" with aggregated probabilities.

    Raises:
        None: Returns zero probability for missing variants.
    """
    if logger is None:
        logger = logging.getLogger("paper_reproduction")
    
    # Token aliases: handle space-prefixed, newline-prefixed, tab-prefixed variants
    aliases = {
        "A": {"A", "▁A", " A", "\nA", "\tA", "Ċ"},  # Ċ is sometimes space representation
        "B": {"B", "▁B", " B", "\nB", "\tB"},
        "T": {"T", "▁T", " T", "\nT", "\tT"},
    }
    out = {"A": 0.0, "B": 0.0, "T": 0.0}
    
    for letter, alset in aliases.items():
        # Take maximum probability across all variants of this letter
        out[letter] = max((text_prob_map.get(a, 0.0) for a in alset), default=0.0)
        logger.debug(f"Letter {letter}: variants={alset}, max_prob={out[letter]:.6f}")
    
    logger.debug(f"only_abt_probs result: {out}")
    return out


def normalize_probs_abc(prob_map: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize A/B/T probabilities to sum to 1.0.

    Args:
        prob_map (Dict[str, float]): Dictionary with keys "A", "B", "T".

    Returns:
        Dict[str, float]: Normalized probabilities summing to 1.0.

    Raises:
        None: Falls back to uniform (1/3 each) if sum is zero.
    """
    total = prob_map["A"] + prob_map["B"] + prob_map["T"]
    if total <= 0:
        # Fall back to uniform so they sum to 1
        return {"A": 1/3, "B": 1/3, "T": 1/3}
    return {k: prob_map[k] / total for k in ["A", "B", "T"]}


# ######################
# ### VLLM HELPERS   ###
# ######################

def build_verdict_prompt(
    question: str,
    response_a: str,
    response_b: str,
    benchmark: str = "math500",
    reasoning_mode: str = "none",
) -> list:
    """
    Build verdict prompt for judge model using chat format.
    
    Per the paper: "we obtain the verdict by instructing the model to directly output
    a label: 'A', 'T', or 'B'". For non-CoT modes, we use a forced assistant prefix 
    to ensure the model outputs the verdict token at position 0.
    
    Args:
        question (str): Original user question/prompt.
        response_a (str): Response from Assistant A.
        response_b (str): Response from Assistant B.
        benchmark (str): Benchmark type - "math500", "mmlu", or "mbpp-plus".
        reasoning_mode (str): Reasoning mode - "none", "cot", or "long_cot".
    
    Returns:
        List of chat messages. For "none" mode, includes forced assistant prefix.
        For CoT modes, allows free generation. For "long_cot", appends <think> token.
    """
    # Get the appropriate prompt template
    prompt_template = get_evaluator_prompt(benchmark, reasoning_mode)
    
    # Format the prompt with the actual content
    user_content = prompt_template.format(
        question=question,
        answer_a=response_a,
        answer_b=response_b,
    )
    
    # Build chat messages based on reasoning mode
    if reasoning_mode == "none":
        # Remove token forcing (this works better)
        return [
            {"role": "user", "content": user_content},
        ]
    elif reasoning_mode == "cot":
        # Standard CoT: Let model reason freely, parse verdict from "My final verdict is $$X$$"
        return [
            {"role": "user", "content": user_content},
        ]
    elif reasoning_mode == "long_cot":
        # Long CoT: Force model into extended reasoning with <think> token
        return [
            {"role": "user", "content": user_content + "\n"},
            {"role": "assistant", "content": "<think>"},
        ]
    else:
        raise ValueError(f"Invalid reasoning_mode: {reasoning_mode}")


def extract_abt_probs_from_logprobs(
    output,
    tokenizer,
    logger: logging.Logger = None,
) -> tuple:
    """
    Extract A/B/T probabilities from vLLM output by finding first A/B/T token.
    
    This mirrors the approach in run_arena_self_preference.py:
    1. Scan generated tokens to find first A, B, or T
    2. Extract logprobs from that specific token position
    3. Aggregate probabilities for A/B/T variants
    
    Args:
        output: vLLM RequestOutput object.
        tokenizer: Tokenizer for decoding token IDs.
        logger: Logger for debug output.
    
    Returns:
        Tuple of (abt_probs dict, token_position) where abt_probs has keys A/B/T.
    """
    if logger is None:
        logger = logging.getLogger("paper_reproduction")
    
    if not output.outputs or not output.outputs[0].token_ids:
        logger.warning("No tokens generated")
        return {"A": 0.33, "B": 0.33, "T": 0.34}, -1
    
    completion = output.outputs[0]
    token_ids = completion.token_ids
    logprobs_list = completion.logprobs if hasattr(completion, 'logprobs') else []
    
    # Token variants for A, B, T (stripped versions to match)
    ABT_STRIPPED = {"A", "B", "T"}
    
    # SMARTER EXTRACTION: Parse generated text to find verdict
    # Use text-based matching to avoid catching "T" in words like "The" or "Tie"
    generated_text = completion.text.strip()
    
    # Initialize to None before trying patterns
    abt_token_idx = None
    
    # Pattern 1: Starts with standalone A, B, or T (with optional punctuation/whitespace)
    import re
    match = re.match(r'^([ABT])(?:\s|$|[.,!?:])', generated_text)
    if match:
        verdict_char = match.group(1)
        # Now find this verdict token in the token sequence
        for i, token_id in enumerate(token_ids):
            try:
                decoded = tokenizer.decode([token_id], skip_special_tokens=False).strip()
                if decoded == verdict_char:
                    abt_token_idx = i
                    logger.debug(f"Found verdict '{verdict_char}' at token position {i} via text pattern match")
                    break
            except Exception:
                continue
    
    # Pattern 2: "verdict is X" or similar patterns
    if abt_token_idx is None:
        patterns = [
            r'verdict\s*:?\s*is\s*([ABT])',
            r'verdict\s*:?\s*([ABT])',
            r'answer\s*:?\s*is\s*([ABT])',
        ]
        for pattern in patterns:
            match = re.search(pattern, generated_text, re.IGNORECASE)
            if match:
                verdict_char = match.group(1).upper()
                # Find this token
                for i, token_id in enumerate(token_ids):
                    try:
                        decoded = tokenizer.decode([token_id], skip_special_tokens=False).strip()
                        if decoded == verdict_char:
                            abt_token_idx = i
                            logger.debug(f"Found verdict '{verdict_char}' at token position {i} via pattern: {pattern}")
                            break
                    except Exception:
                        continue
                if abt_token_idx is not None:
                    break
    
    # Fallback: First A/B/T token (old behavior)
    if abt_token_idx is None:
        for i, token_id in enumerate(token_ids):
            try:
                decoded = tokenizer.decode([token_id], skip_special_tokens=False).strip()
                if decoded in ABT_STRIPPED:
                    abt_token_idx = i
                    logger.debug(f"Found A/B/T token '{decoded}' at position {i} (fallback method)")
                    break
            except Exception:
                continue
    
    if abt_token_idx is None:
        # If only 1 token generated, use position 0 for logprobs extraction
        if len(token_ids) == 1 and len(logprobs_list) >= 1:
            logger.debug(f"No A/B/T token found, but single token generated - using position 0 for logprobs")
            abt_token_idx = 0
        else:
            logger.warning(f"No A/B/T token found in generated sequence: {completion.text[:100]}")
            return {"A": 0.33, "B": 0.33, "T": 0.34}, -1
    
    # Get logprobs for that token position
    if abt_token_idx >= len(logprobs_list):
        logger.warning(f"Token index {abt_token_idx} out of range for logprobs list (len={len(logprobs_list)})")
        return {"A": 0.33, "B": 0.33, "T": 0.34}, abt_token_idx
    
    token_logprobs_dict = logprobs_list[abt_token_idx]
    
    # Token variants for aggregation (with space/unicode variants)
    aliases = {
        "A": {"A", "▁A", " A", "\nA", "\tA"},
        "B": {"B", "▁B", " B", "\nB", "\tB"},
        "T": {"T", "▁T", " T", "\nT", "\tT"},
    }
    
    # Aggregate probabilities
    abt_probs = {"A": 0.0, "B": 0.0, "T": 0.0}
    
    # vLLM logprobs structure: dict[int, float] or dict[int, Logprob object]
    for token_id, logprob_val in token_logprobs_dict.items():
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            # Extract logprob value (handle both float and Logprob object)
            if hasattr(logprob_val, 'logprob'):
                logprob = logprob_val.logprob
            else:
                logprob = float(logprob_val)
            prob = math.exp(logprob)
            
            # Check which label this token belongs to
            for label, variants in aliases.items():
                if decoded in variants:
                    abt_probs[label] += prob
                    logger.debug(f"Token '{decoded}' -> {label}, prob={prob:.6f}")
                    break
        except Exception as e:
            logger.debug(f"Error processing token {token_id}: {e}")
            continue
    
    # Normalize
    total = sum(abt_probs.values())
    if total > 0:
        abt_probs = {k: v / total for k, v in abt_probs.items()}
    else:
        logger.warning(f"No A/B/T probabilities found, using uniform")
        abt_probs = {"A": 0.33, "B": 0.33, "T": 0.34}
    
    logger.debug(f"Final A/B/T probs: {abt_probs}")
    return abt_probs, abt_token_idx


def extract_cot_verdict_with_logprobs(
    output,
    tokenizer,
    logger: logging.Logger = None,
) -> tuple:
    """
    Extract A/B/T probabilities from CoT response by finding the verdict token position.
    
    For CoT responses, the verdict appears at the END of the generated text in the format
    "My final verdict is $$X$$" where X is A, B, or T. We need to:
    1. Find the pattern in the generated text
    2. Map the character position of the verdict to a token position
    3. Extract logprobs at that specific token position
    
    Args:
        output: vLLM RequestOutput object.
        tokenizer: Tokenizer for decoding token IDs.
        logger: Logger for debug output.
    
    Returns:
        Tuple of (abt_probs dict, token_position) where abt_probs has keys A/B/T.
    """
    if logger is None:
        logger = logging.getLogger("paper_reproduction")
    
    if not output.outputs or not output.outputs[0].token_ids:
        logger.warning("No tokens generated for CoT extraction")
        return {"A": 0.33, "B": 0.33, "T": 0.34}, -1
    
    completion = output.outputs[0]
    token_ids = completion.token_ids
    logprobs_list = completion.logprobs if hasattr(completion, 'logprobs') else []
    generated_text = completion.text
    
    if not logprobs_list:
        logger.warning("No logprobs available for CoT extraction")
        return {"A": 0.33, "B": 0.33, "T": 0.34}, -1
    
    # Find the verdict pattern in the generated text
    # offset = how many characters from match.end() back to the verdict letter
    # For $$X$$: match.end() points after final $, so offset=3 ($$, then X)
    # For X\b: match.end() points after the letter (word boundary), so offset=1
    patterns = [
        (r"My final verdict is \$\$([ABT])\$\$", 3),  # $$X$$ format
        (r"final verdict is \$\$([ABT])\$\$", 3),
        (r"verdict is \$\$([ABT])\$\$", 3),
        (r"\$\$([ABT])\$\$", 3),  # Just the delimited verdict
        (r"My final verdict is ([ABT])\b", 1),  # Without $$ delimiters
        (r"final verdict is ([ABT])\b", 1),
        (r"verdict is ([ABT])\b", 1),
    ]
    
    verdict_char = None
    verdict_char_pos = None
    
    for pattern, offset_from_end in patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            verdict_char = match.group(1).upper()
            # The verdict character position is at match.end() minus the offset
            # For $$X$$, the X is 3 characters before the end (the "$$")
            # For just X, it's 0 characters before the end
            # Note: match.end() is exclusive, so match.end() - 3 gives us the B in "$$B$$"
            verdict_char_pos = match.end() - offset_from_end
            logger.debug(f"Found CoT verdict '{verdict_char}' at char position {verdict_char_pos} via pattern: {pattern}")
            break
    
    if verdict_char is None:
        logger.warning(f"Could not find verdict pattern in CoT response: {generated_text[-200:]}")
        return {"A": 0.33, "B": 0.33, "T": 0.34}, -1
    
    # Now map character position to token position
    # Decode tokens incrementally and track cumulative text length
    cumulative_text = ""
    verdict_token_idx = None
    
    for idx, token_id in enumerate(token_ids):
        try:
            # Decode this single token
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            prev_len = len(cumulative_text)
            cumulative_text += token_str
            
            # Check if the verdict character position falls within this token
            if prev_len <= verdict_char_pos < len(cumulative_text):
                verdict_token_idx = idx
                logger.debug(f"Verdict char at pos {verdict_char_pos} maps to token idx {idx} (token='{token_str}')")
                break
        except Exception as e:
            logger.debug(f"Error decoding token {token_id}: {e}")
            continue
    
    if verdict_token_idx is None:
        logger.warning(f"Could not map verdict char position {verdict_char_pos} to token index")
        # Fallback: try to find the verdict token by scanning from the end
        for idx in range(len(token_ids) - 1, -1, -1):
            try:
                decoded = tokenizer.decode([token_ids[idx]], skip_special_tokens=False).strip()
                if decoded == verdict_char:
                    verdict_token_idx = idx
                    logger.debug(f"Fallback: found verdict token '{verdict_char}' at position {idx}")
                    break
            except Exception:
                continue
    
    if verdict_token_idx is None:
        logger.warning(f"Could not find verdict token position for '{verdict_char}'")
        return {"A": 0.33, "B": 0.33, "T": 0.34}, -1
    
    # Get logprobs for that token position
    if verdict_token_idx >= len(logprobs_list):
        logger.warning(f"Token index {verdict_token_idx} out of range for logprobs list (len={len(logprobs_list)})")
        return {"A": 0.33, "B": 0.33, "T": 0.34}, verdict_token_idx
    
    token_logprobs_dict = logprobs_list[verdict_token_idx]
    
    # Token variants for aggregation (same as non-CoT)
    aliases = {
        "A": {"A", "▁A", " A", "\nA", "\tA"},
        "B": {"B", "▁B", " B", "\nB", "\tB"},
        "T": {"T", "▁T", " T", "\nT", "\tT"},
    }
    
    # Aggregate probabilities
    abt_probs = {"A": 0.0, "B": 0.0, "T": 0.0}
    
    for token_id, logprob_val in token_logprobs_dict.items():
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            # Extract logprob value (handle both float and Logprob object)
            if hasattr(logprob_val, 'logprob'):
                logprob = logprob_val.logprob
            else:
                logprob = float(logprob_val)
            prob = math.exp(logprob)
            
            # Check which label this token belongs to
            for label, variants in aliases.items():
                if decoded in variants:
                    abt_probs[label] += prob
                    logger.debug(f"CoT token '{decoded}' -> {label}, prob={prob:.6f}")
                    break
        except Exception as e:
            logger.debug(f"Error processing token {token_id}: {e}")
            continue
    
    # Normalize
    total = sum(abt_probs.values())
    if total > 0:
        abt_probs = {k: v / total for k, v in abt_probs.items()}
    else:
        logger.warning(f"No A/B/T probabilities found in CoT, using uniform")
        abt_probs = {"A": 0.33, "B": 0.33, "T": 0.34}
    
    # For CoT mode: Harden to 0/1 probabilities (set max to 1, rest to 0)
    # This matches how the original paper appears to record CoT verdicts
    winner = max(abt_probs.items(), key=lambda x: x[1])[0]
    abt_probs = {"A": 0.0, "B": 0.0, "T": 0.0}
    abt_probs[winner] = 1.0
    
    logger.debug(f"CoT final A/B/T probs (hardened): {abt_probs}")
    return abt_probs, verdict_token_idx


def setup_vllm_engine(model_id: str, config: Dict, logger: logging.Logger):
    """
    Set up vLLM engine for inference.

    Args:
        model_id (str): Hugging Face model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        config (Dict): Configuration dict with vLLM settings.
        logger (logging.Logger): Logger instance for recording setup progress.

    Returns:
        vllm.LLM: Configured vLLM engine instance.
    """
    logger.info(f"Setting up vLLM engine for {model_id}")
    if config is None:
        config = CONFIG
    vllm_kwargs = {k.replace("vllm_", ""): v for k, v in config.items() if k.startswith("vllm_")}
    return LLM(model_id,
            tokenizer=model_id,
            trust_remote_code=True,
            **vllm_kwargs)    
    


def generate_verdict_vllm(
    llm,
    tokenizer,
    prompts: List[list],
    temperature: float,
    max_tokens: Optional[int],
    config: Dict,
    logger: logging.Logger,
) -> List:
    """
    Generate verdicts using vLLM with logprob extraction.

    Args:
        llm: vLLM LLM instance.
        tokenizer: Tokenizer instance.
        prompts (List[list]): List of chat message lists.
        temperature (float): Sampling temperature.
        max_tokens (Optional[int]): Maximum tokens to generate. None = unlimited (until EOS).
        config (Dict): Configuration dict.
        logger (logging.Logger): Logger instance.

    Returns:
        List: vLLM outputs with logprobs.
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        logprobs=config["verdict_top_k"],
    )
    
    # Apply chat template - use STRING output instead of token IDs
    formatted_prompts = []
    for prompt_msgs in prompts:
        # Check if last message is from assistant (needs continuation)
        has_assistant_prefix = prompt_msgs[-1]["role"] == "assistant"
        
        if has_assistant_prefix:
            formatted = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,  # Return string, not token IDs
                continue_final_message=True,
            )
        else:
            formatted = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,  # Return string, not token IDs
                add_generation_prompt=True,
            )
        formatted_prompts.append(formatted)  # Pass string directly
        logger.debug(f"Formatted prompt ends with: ...{formatted[-100:]}")
    
    outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)
    return outputs


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_sp_data(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize llm-sp data format to expected format.
    
    Handles key name differences:
    - problem -> question
    - assistent_1_answer -> response_a
    - assistent_2_answer -> response_b
    - game_1_spb_score/game_2_spb_score wrapped in meta dict
    
    Args:
        row (Dict[str, Any]): Raw row from llm-sp/sp JSONL file.
    
    Returns:
        Dict[str, Any]: Normalized row with standard key names.
    """
    # Check if already normalized (has 'question' key)
    if "question" in row:
        return row
    
    # Normalize llm-sp format
    normalized = {
        "question": row.get("problem", ""),
        "response_a": row.get("assistent_1_answer", ""),
        "response_b": row.get("assistent_2_answer", ""),
        "meta": {
            "game_1_spb_score": row.get("game_1_spb_score", [0.33, 0.33, 0.34]),
            "game_2_spb_score": row.get("game_2_spb_score", [0.33, 0.33, 0.34]),
            "unique_id": row.get("unique_id", ""),
            "assistent_1_is_correct": row.get("assistent_1_is_correct", None),
            "assistent_2_is_correct": row.get("assistent_2_is_correct", None),
        }
    }
    
    return normalized


def get_verdict_label(probs: Dict[str, float]) -> str:
    """
    Get the verdict label (A/B/T) from probability dict.
    
    Args:
        probs (Dict[str, float]): Dictionary with A/B/T probabilities.
    
    Returns:
        str: Label with highest probability ("A", "B", or "T").
    """
    return max(probs.items(), key=lambda x: x[1])[0]


def analyze_verdict_flips(results: List[Dict[str, Any]], logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyze verdict flips between generated and reference verdicts.
    
    A "flip" occurs when the generated verdict differs from the reference verdict
    (e.g., reference says "A" but generated says "B").
    
    Args:
        results (List[Dict[str, Any]]): List of result records with game_1 and game_2.
        logger (logging.Logger): Logger instance.
    
    Returns:
        Dict[str, Any]: Analysis results including:
        - flip_counts: Count of flips per game
        - transition_matrix: Dict showing reference->generated transitions
        - high_confidence_flips: Flips where generated prob > 0.7
        - examples_with_flips: List of example IDs with flips
    """
    # Track flips
    flips_game1 = []
    flips_game2 = []
    
    # Transition matrices: {ref_label: {gen_label: count}}
    transition_g1 = {"A": {"A": 0, "B": 0, "T": 0}, "B": {"A": 0, "B": 0, "T": 0}, "T": {"A": 0, "B": 0, "T": 0}}
    transition_g2 = {"A": {"A": 0, "B": 0, "T": 0}, "B": {"A": 0, "B": 0, "T": 0}, "T": {"A": 0, "B": 0, "T": 0}}
    
    high_confidence_flips_g1 = []
    high_confidence_flips_g2 = []
    
    for r in results:
        # Game 1 analysis
        ref_label_g1 = get_verdict_label(r["game_1"]["reference_probs"])
        gen_label_g1 = get_verdict_label(r["game_1"]["generated_probs"])
        transition_g1[ref_label_g1][gen_label_g1] += 1
        
        if ref_label_g1 != gen_label_g1:
            flip_info = {
                "example_id": r["example_id"],
                "reference": ref_label_g1,
                "generated": gen_label_g1,
                "ref_prob": r["game_1"]["reference_probs"][ref_label_g1],
                "gen_prob": r["game_1"]["generated_probs"][gen_label_g1],
                "max_diff": r["game_1"]["max_diff"],
            }
            flips_game1.append(flip_info)
            
            # High confidence flip: generated probability > 0.7
            if r["game_1"]["generated_probs"][gen_label_g1] > 0.7:
                high_confidence_flips_g1.append(flip_info)
        
        # Game 2 analysis
        ref_label_g2 = get_verdict_label(r["game_2"]["reference_probs"])
        gen_label_g2 = get_verdict_label(r["game_2"]["generated_probs"])
        transition_g2[ref_label_g2][gen_label_g2] += 1
        
        if ref_label_g2 != gen_label_g2:
            flip_info = {
                "example_id": r["example_id"],
                "reference": ref_label_g2,
                "generated": gen_label_g2,
                "ref_prob": r["game_2"]["reference_probs"][ref_label_g2],
                "gen_prob": r["game_2"]["generated_probs"][gen_label_g2],
                "max_diff": r["game_2"]["max_diff"],
            }
            flips_game2.append(flip_info)
            
            if r["game_2"]["generated_probs"][gen_label_g2] > 0.7:
                high_confidence_flips_g2.append(flip_info)
    
    # Log flip statistics
    logger.info(f"Game 1 Flips: {len(flips_game1)}/{len(results)} ({100*len(flips_game1)/len(results):.1f}%)")
    logger.info(f"Game 2 Flips: {len(flips_game2)}/{len(results)} ({100*len(flips_game2)/len(results):.1f}%)")
    logger.info(f"High-confidence flips (prob>0.7) Game 1: {len(high_confidence_flips_g1)}")
    logger.info(f"High-confidence flips (prob>0.7) Game 2: {len(high_confidence_flips_g2)}")
    
    # Log transition matrices
    logger.info("\nGame 1 Transition Matrix (Ref -> Gen):")
    logger.info("         A      B      T")
    for ref_label in ["A", "B", "T"]:
        counts = [transition_g1[ref_label][gen_label] for gen_label in ["A", "B", "T"]]
        logger.info(f"  {ref_label}:  {counts[0]:4d}   {counts[1]:4d}   {counts[2]:4d}")
    
    logger.info("\nGame 2 Transition Matrix (Ref -> Gen):")
    logger.info("         A      B      T")
    for ref_label in ["A", "B", "T"]:
        counts = [transition_g2[ref_label][gen_label] for gen_label in ["A", "B", "T"]]
        logger.info(f"  {ref_label}:  {counts[0]:4d}   {counts[1]:4d}   {counts[2]:4d}")
    
    # Log most problematic flips
    if high_confidence_flips_g1:
        logger.info(f"\nMost confident WRONG verdicts in Game 1 (top 5):")
        sorted_flips = sorted(high_confidence_flips_g1, key=lambda x: x["gen_prob"], reverse=True)[:5]
        for flip in sorted_flips:
            logger.info(f"  Example {flip['example_id']}: {flip['reference']}->{flip['generated']} " +
                       f"(gen_prob={flip['gen_prob']:.3f}, max_diff={flip['max_diff']:.3f})")
    
    return {
        "game_1": {
            "flip_count": len(flips_game1),
            "flip_rate": len(flips_game1) / len(results),
            "flips": flips_game1,
            "high_confidence_flips": high_confidence_flips_g1,
            "transition_matrix": transition_g1,
        },
        "game_2": {
            "flip_count": len(flips_game2),
            "flip_rate": len(flips_game2) / len(results),
            "flips": flips_game2,
            "high_confidence_flips": high_confidence_flips_g2,
            "transition_matrix": transition_g2,
        },
    }


def run_reproduction_experiment(
    judge_model: str,
    data_path: Path,
    output_dir: Path,
    benchmark: str = None,
    judge_family: str = None,
    judge_short: str = None,
    evaluatee_short: str = None,
    reasoning_mode: str = "none",
    n_samples: int = None,
    seed: int = 42,
    config: Dict = None,
    logger: logging.Logger = None,
    tensor_parallel_size: int = None,
) -> None:
    """
    Run reproduction experiment comparing generated verdicts to reference scores.
    
    Args:
        judge_model (str): vLLM model ID for judge (generates assistant_1 and does judging).
        data_path (Path): Path to input JSONL with subsample.
        output_dir (Path): Output directory for results (base path for llm-sp-reprod structure).
        benchmark (str): Benchmark name (e.g., 'math500', 'mmlu', 'mbpp-plus'). 
                        If provided, creates llm-sp-reprod structure and uses benchmark-specific prompts.
        judge_family (str): Judge family name (e.g., 'llama', 'qwen'). Required if benchmark is provided.
        judge_short (str): Judge model short name (e.g., 'llama-3.1-8b'). Required if benchmark is provided.
        evaluatee_short (str): Evaluatee model short name (e.g., 'gpt-4o'). Required if benchmark is provided.
        reasoning_mode (str): Reasoning mode for verdict generation - "none", "cot", or "long_cot".
                             Defaults to "none" (direct A/B/T token output).
        n_samples (int): Number of samples to test (default: all).
        seed (int): Random seed for reproducibility.
        config (Dict): Configuration dict.
        logger (logging.Logger): Logger instance.
    
    Returns:
        None (writes results to llm-sp-reprod structure and generates visualizations).
    """
    if logger is None:
        logger = setup_logging(output_dir)
    if config is None:
        config = CONFIG
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from: {data_path}")
    rows = load_jsonl(data_path)
    logger.info(f"Loaded {len(rows)} total rows")
    
    # Normalize data format (handles llm-sp format with different key names)
    rows = [normalize_sp_data(row) for row in rows]
    logger.info("Data normalized to standard format")
    
    # Take deterministic sample if requested
    if n_samples and len(rows) > n_samples:
        random.seed(seed)
        rows = random.sample(rows, n_samples)
        logger.info(f"Sampled {n_samples} rows for testing")
    
    # Determine tensor parallel size: CLI arg > config > default (1)
    tp_size = tensor_parallel_size if tensor_parallel_size is not None else config.get("vllm_tensor_parallel_size", 1)
    logger.info(f"Setting up vLLM engine for: {judge_model} (tensor_parallel_size={tp_size})")
    
    # Use block_size=32 to avoid FlashInfer bug with head_size=256 models (e.g., gemma-2-9b)
    # "There is a bug in FlashInfer block_size 16 head size 256 support"
    block_size = config.get("vllm_block_size", 32)
    
    llm = LLM(
        model=judge_model,
        tokenizer=judge_model,
        trust_remote_code=True,
        gpu_memory_utilization=config.get("vllm_gpu_memory_utilization", 0.9),
        tensor_parallel_size=tp_size,
        block_size=block_size,
    )
    tokenizer = llm.get_tokenizer()
    
    # Determine temperature based on model type
    temperature = 0.6 if is_reasoning_model(judge_model) else 0.0
    logger.info(f"Using temperature: {temperature}")
    
    # Determine max_tokens based on reasoning mode
    # - "none": Paper uses max_tokens=1 for verdict generation
    # - "cot" / "long_cot": None = generate until EOS or model max context length
    if reasoning_mode == "none":
        max_tokens = config.get("verdict_max_tokens", 1)
    else:
        # None means vLLM will generate until EOS token or model's max context length
        max_tokens = None
    
    # Determine the benchmark for prompt selection
    # Default to math500 if not specified
    effective_benchmark = benchmark if benchmark else "math500"
    
    logger.info(f"Using benchmark: {effective_benchmark}")
    logger.info(f"Using reasoning mode: {reasoning_mode}")
    logger.info(f"Max tokens: {max_tokens if max_tokens else 'unlimited (until EOS)'}")
    
    logger.info("=" * 80)
    logger.info("RUNNING GENERATION ON GAME 1 (AB ORDER)")
    logger.info("=" * 80)
    
    # Game 1: A=response_a, B=response_b
    prompts_game1 = [
        build_verdict_prompt(
            row["question"], 
            row["response_a"], 
            row["response_b"],
            benchmark=effective_benchmark,
            reasoning_mode=reasoning_mode,
        )
        for row in rows
    ]
    
    outputs_game1 = generate_verdict_vllm(
        llm, tokenizer, prompts_game1, temperature, max_tokens, config, logger
    )
    
    logger.info("=" * 80)
    logger.info("RUNNING GENERATION ON GAME 2 (BA ORDER)")
    logger.info("=" * 80)
    
    # Game 2: A=response_b, B=response_a (swap)
    prompts_game2 = [
        build_verdict_prompt(
            row["question"], 
            row["response_b"], 
            row["response_a"],
            benchmark=effective_benchmark,
            reasoning_mode=reasoning_mode,
        )
        for row in rows
    ]
    
    outputs_game2 = generate_verdict_vllm(
        llm, tokenizer, prompts_game2, temperature, max_tokens, config, logger
    )
    
    logger.info("=" * 80)
    logger.info("EXTRACTING A/B/T PROBABILITIES AND COMPARING TO REFERENCE")
    logger.info("=" * 80)
    
    # Process results
    results = []
    all_diffs_game1 = []
    all_diffs_game2 = []
    
    for i, row in enumerate(rows):
        out1 = outputs_game1[i]
        out2 = outputs_game2[i]
        
        # Extract A/B/T probs using appropriate method based on reasoning mode
        if reasoning_mode == "none":
            # Non-CoT: verdict is at the start (forced by "My verdict is:" prefix)
            abt_probs_game1, token_pos1 = extract_abt_probs_from_logprobs(out1, tokenizer, logger)
            abt_probs_game2, token_pos2 = extract_abt_probs_from_logprobs(out2, tokenizer, logger)
        else:
            # CoT / Long CoT: verdict is at the end after "My final verdict is $$X$$"
            abt_probs_game1, token_pos1 = extract_cot_verdict_with_logprobs(out1, tokenizer, logger)
            abt_probs_game2, token_pos2 = extract_cot_verdict_with_logprobs(out2, tokenizer, logger)
        
        # Both Game 1 and Game 2 reference scores appear to be stored in 
        # [A_pref, B_pref, tie] format for their respective game orderings.
        # So we compare generated probs directly without normalization.

        # NO NORMALIZATION - compare raw generated probs to raw reference probs
        abt_probs_game2_normalized = abt_probs_game2
        
        # DEBUG: Log values
        logger.info(f"  Game 2 probs: A={abt_probs_game2['A']:.4f}, B={abt_probs_game2['B']:.4f}, T={abt_probs_game2['T']:.4f}")
        
        # Get reference scores (already in (response_a_pref, response_b_pref, tie) format)
        ref_game1 = row["meta"].get("game_1_spb_score", [0.33, 0.33, 0.34])
        ref_game2 = row["meta"].get("game_2_spb_score", [0.33, 0.33, 0.34])
        
        # Calculate differences
        diff_game1 = {
            "A": abs(abt_probs_game1["A"] - ref_game1[0]),
            "B": abs(abt_probs_game1["B"] - ref_game1[1]),
            "T": abs(abt_probs_game1["T"] - ref_game1[2]),
        }
        diff_game2 = {
            "A": abs(abt_probs_game2_normalized["A"] - ref_game2[0]),
            "B": abs(abt_probs_game2_normalized["B"] - ref_game2[1]),
            "T": abs(abt_probs_game2_normalized["T"] - ref_game2[2]),
        }
        
        all_diffs_game1.extend([diff_game1["A"], diff_game1["B"], diff_game1["T"]])
        all_diffs_game2.extend([diff_game2["A"], diff_game2["B"], diff_game2["T"]])
        
        max_diff_game1 = max(diff_game1.values())
        max_diff_game2 = max(diff_game2.values())
        
        logger.info(f"Example {i+1}/{len(rows)}:")
        logger.info(f"  Game 1 - Generated: A={abt_probs_game1['A']:.4f}, B={abt_probs_game1['B']:.4f}, T={abt_probs_game1['T']:.4f} [token@{token_pos1}]")
        logger.info(f"  Game 1 - Reference: A={ref_game1[0]:.4f}, B={ref_game1[1]:.4f}, T={ref_game1[2]:.4f}")
        logger.info(f"  Game 1 - Max diff: {max_diff_game1:.4f}")
        logger.info(f"  Game 2 - Generated: A={abt_probs_game2['A']:.4f}, B={abt_probs_game2['B']:.4f}, T={abt_probs_game2['T']:.4f} [token@{token_pos2}]")
        logger.info(f"  Game 2 - Reference: A={ref_game2[0]:.4f}, B={ref_game2[1]:.4f}, T={ref_game2[2]:.4f}")
        logger.info(f"  Game 2 - Max diff: {max_diff_game2:.4f}")
        
        # DEBUG: What are we about to store?
        logger.info(f"  STORING Game 2 normalized: {abt_probs_game2_normalized}")
        
        results.append({
            "example_id": i,
            "question": row["question"],
            "response_a": row["response_a"],
            "response_b": row["response_b"],
            "game_1": {
                "generated_probs": abt_probs_game1,
                "reference_probs": {"A": ref_game1[0], "B": ref_game1[1], "T": ref_game1[2]},
                "differences": diff_game1,
                "max_diff": max_diff_game1,
                "token_position": token_pos1,
                "generated_text": out1.outputs[0].text if out1.outputs else "",
            },
            "game_2": {
                "generated_probs": abt_probs_game2_normalized,
                "reference_probs": {"A": ref_game2[0], "B": ref_game2[1], "T": ref_game2[2]},
                "differences": diff_game2,
                "max_diff": max_diff_game2,
                "token_position": token_pos2,
                "generated_text": out2.outputs[0].text if out2.outputs else "",
            },
        })
    
    # Determine output paths based on structure matching
    if benchmark and judge_family and judge_short and evaluatee_short:
        # Create llm-sp-reprod structure matching llm-sp/sp
        # Pattern: {judge_short}_{evaluatee_short}_reprod.jsonl
        # Use different base directories for different reasoning modes:
        # - "none" -> llm-sp-reprod/ (backwards compatible)
        # - "cot" -> llm-sp-reprod-cot/
        # - "long_cot" -> llm-sp-reprod-long_cot/
        if reasoning_mode == "none":
            base_reprod_dir = "llm-sp-reprod"
        else:
            base_reprod_dir = f"llm-sp-reprod-{reasoning_mode}"
        
        eval_dir = Path(base_reprod_dir) / benchmark / judge_family
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename pattern: {judge_short}_{evaluatee_short}_reprod.jsonl
        output_file = eval_dir / f"{judge_short}_{evaluatee_short}_reprod.jsonl"
        
        # Visualizations go to separate plots directory (per judge/evaluatee pair)
        plot_base = "reproduction_results" if reasoning_mode == "none" else f"reproduction_results-{reasoning_mode}"
        plot_dir = Path(plot_base) / "plots" / benchmark / judge_family / judge_short / evaluatee_short
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using llm-sp-reprod structure (reasoning_mode={reasoning_mode}):")
        logger.info(f"  Eval file: {output_file}")
        logger.info(f"  Plots: {plot_dir}")
    else:
        # Fallback to legacy structure
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"reproduction_{judge_model.replace('/', '-')}_{len(rows)}examples.jsonl"
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using legacy structure:")
        logger.info(f"  Eval file: {output_file}")
        logger.info(f"  Plots: {plot_dir}")
    
    # Write results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info("=" * 80)
    logger.info(f"Results written to: {output_file}")
    logger.info("=" * 80)
    
    # Summary statistics
    avg_max_diff_game1 = sum(r["game_1"]["max_diff"] for r in results) / len(results)
    avg_max_diff_game2 = sum(r["game_2"]["max_diff"] for r in results) / len(results)
    
    logger.info("SUMMARY STATISTICS:")
    logger.info(f"  Average max difference Game 1: {avg_max_diff_game1:.4f}")
    logger.info(f"  Average max difference Game 2: {avg_max_diff_game2:.4f}")
    logger.info(f"  Median difference Game 1: {np.median(all_diffs_game1):.4f}")
    logger.info(f"  Median difference Game 2: {np.median(all_diffs_game2):.4f}")
    logger.info(f"  95th percentile diff Game 1: {np.percentile(all_diffs_game1, 95):.4f}")
    logger.info(f"  95th percentile diff Game 2: {np.percentile(all_diffs_game2, 95):.4f}")
    
    # Enhanced analytics: verdict flips and transition analysis
    logger.info("=" * 80)
    logger.info("VERDICT FLIP ANALYSIS")
    logger.info("=" * 80)
    
    flip_analysis = analyze_verdict_flips(results, logger)
    
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    try:
        # Plot 1: Distribution of differences
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(all_diffs_game1, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(avg_max_diff_game1, color='red', linestyle='--', label=f'Mean: {avg_max_diff_game1:.4f}')
        axes[0].axvline(np.median(all_diffs_game1), color='green', linestyle='--', label=f'Median: {np.median(all_diffs_game1):.4f}')
        axes[0].set_xlabel('Absolute Difference from Reference')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Game 1 (AB Order) - Distribution of Differences')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(all_diffs_game2, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1].axvline(avg_max_diff_game2, color='red', linestyle='--', label=f'Mean: {avg_max_diff_game2:.4f}')
        axes[1].axvline(np.median(all_diffs_game2), color='green', linestyle='--', label=f'Median: {np.median(all_diffs_game2):.4f}')
        axes[1].set_xlabel('Absolute Difference from Reference')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Game 2 (BA Order) - Distribution of Differences')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file1 = plot_dir / f"difference_distributions_{judge_model.replace('/', '-')}.png"
        plt.savefig(plot_file1, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {plot_file1}")
        plt.close()
        
        # Plot 2: Per-example max differences
        fig, ax = plt.subplots(figsize=(12, 6))
        example_ids = [r["example_id"] for r in results]
        max_diffs_g1 = [r["game_1"]["max_diff"] for r in results]
        max_diffs_g2 = [r["game_2"]["max_diff"] for r in results]
        
        x = np.arange(len(example_ids))
        width = 0.35
        
        ax.bar(x - width/2, max_diffs_g1, width, label='Game 1 (AB)', alpha=0.8, color='blue')
        ax.bar(x + width/2, max_diffs_g2, width, label='Game 2 (BA)', alpha=0.8, color='orange')
        ax.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='0.1 threshold')
        ax.set_xlabel('Example ID')
        ax.set_ylabel('Max Difference from Reference')
        ax.set_title('Per-Example Maximum Differences')
        ax.set_xticks(x)
        ax.set_xticklabels(example_ids)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file2 = plot_dir / f"per_example_diffs_{judge_model.replace('/', '-')}.png"
        plt.savefig(plot_file2, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {plot_file2}")
        plt.close()
        
        # Plot 3: Transition matrix heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Game 1 transition matrix
        trans_g1 = flip_analysis["game_1"]["transition_matrix"]
        matrix_g1 = np.array([[trans_g1[ref][gen] for gen in ["A", "B", "T"]] for ref in ["A", "B", "T"]])
        
        im1 = axes[0].imshow(matrix_g1, cmap='YlOrRd', aspect='auto')
        axes[0].set_xticks(np.arange(3))
        axes[0].set_yticks(np.arange(3))
        axes[0].set_xticklabels(["A", "B", "T"])
        axes[0].set_yticklabels(["A", "B", "T"])
        axes[0].set_xlabel("Generated Verdict")
        axes[0].set_ylabel("Reference Verdict")
        axes[0].set_title("Game 1: Reference → Generated Transitions")
        
        # Add counts to cells
        for i in range(3):
            for j in range(3):
                text_color = 'white' if matrix_g1[i, j] > matrix_g1.max() / 2 else 'black'
                axes[0].text(j, i, str(matrix_g1[i, j]), ha="center", va="center", color=text_color, fontsize=12, weight='bold')
        
        fig.colorbar(im1, ax=axes[0], label='Count')
        
        # Game 2 transition matrix
        trans_g2 = flip_analysis["game_2"]["transition_matrix"]
        matrix_g2 = np.array([[trans_g2[ref][gen] for gen in ["A", "B", "T"]] for ref in ["A", "B", "T"]])
        
        im2 = axes[1].imshow(matrix_g2, cmap='YlOrRd', aspect='auto')
        axes[1].set_xticks(np.arange(3))
        axes[1].set_yticks(np.arange(3))
        axes[1].set_xticklabels(["A", "B", "T"])
        axes[1].set_yticklabels(["A", "B", "T"])
        axes[1].set_xlabel("Generated Verdict")
        axes[1].set_ylabel("Reference Verdict")
        axes[1].set_title("Game 2: Reference → Generated Transitions")
        
        # Add counts to cells
        for i in range(3):
            for j in range(3):
                text_color = 'white' if matrix_g2[i, j] > matrix_g2.max() / 2 else 'black'
                axes[1].text(j, i, str(matrix_g2[i, j]), ha="center", va="center", color=text_color, fontsize=12, weight='bold')
        
        fig.colorbar(im2, ax=axes[1], label='Count')
        
        plt.tight_layout()
        plot_file3 = plot_dir / f"transition_matrices_{judge_model.replace('/', '-')}.png"
        plt.savefig(plot_file3, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {plot_file3}")
        plt.close()
        
        # Plot 4: Probability difference heatmap (where differences flow)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Probability Differences by Verdict Category (Generated - Reference)', fontsize=14, fontweight='bold')
        
        # Organize data by reference verdict for both games
        for game_idx, game_key in enumerate(["game_1", "game_2"]):
            ref_categories = {"A": [], "B": [], "T": []}
            
            for r in results:
                ref_label = get_verdict_label(r[game_key]["reference_probs"])
                
                # Calculate signed differences (positive = generated higher, negative = reference higher)
                diff_a = r[game_key]["generated_probs"]["A"] - r[game_key]["reference_probs"]["A"]
                diff_b = r[game_key]["generated_probs"]["B"] - r[game_key]["reference_probs"]["B"]
                diff_t = r[game_key]["generated_probs"]["T"] - r[game_key]["reference_probs"]["T"]
                
                ref_categories[ref_label].append([diff_a, diff_b, diff_t])
            
            # Plot heatmaps for each reference category
            for cat_idx, (ref_label, diffs) in enumerate(ref_categories.items()):
                ax = axes[game_idx, cat_idx]
                
                if len(diffs) > 0:
                    diffs_array = np.array(diffs).T  # Shape: (3, n_examples)
                    
                    # Create heatmap
                    im = ax.imshow(diffs_array, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                    ax.set_yticks([0, 1, 2])
                    ax.set_yticklabels(["A", "B", "T"])
                    ax.set_ylabel("Verdict Label")
                    ax.set_xlabel("Example Index")
                    
                    game_name = "Game 1 (AB)" if game_idx == 0 else "Game 2 (BA)"
                    ax.set_title(f"{game_name} | Ref={ref_label} (n={len(diffs)})")
                    
                    # Add colorbar
                    if cat_idx == 2:
                        cbar = fig.colorbar(im, ax=ax, label='Prob Diff')
                else:
                    ax.text(0.5, 0.5, 'No examples', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"Game {game_idx+1} | Ref={ref_label} (n=0)")
        
        plt.tight_layout()
        plot_file4 = plot_dir / f"probability_flow_heatmap_{judge_model.replace('/', '-')}.png"
        plt.savefig(plot_file4, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {plot_file4}")
        plt.close()
        
        # Plot 5: Scatter plot of high-difference examples
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for game_idx, game_key in enumerate(["game_1", "game_2"]):
            ax = axes[game_idx]
            
            # Get max differences and corresponding probability shifts
            max_diffs = []
            prob_shifts = []  # How much probability shifted from reference winner to generated winner
            colors = []
            
            for r in results:
                max_diff = r[game_key]["max_diff"]
                max_diffs.append(max_diff)
                
                ref_label = get_verdict_label(r[game_key]["reference_probs"])
                gen_label = get_verdict_label(r[game_key]["generated_probs"])
                
                # Calculate probability shift
                ref_winner_prob_gen = r[game_key]["generated_probs"][ref_label]
                ref_winner_prob_ref = r[game_key]["reference_probs"][ref_label]
                shift = ref_winner_prob_ref - ref_winner_prob_gen  # Positive = probability moved away from reference winner
                
                prob_shifts.append(shift)
                
                # Color by whether verdict flipped
                colors.append('red' if ref_label != gen_label else 'blue')
            
            ax.scatter(max_diffs, prob_shifts, c=colors, alpha=0.6, s=50)
            ax.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax.set_xlabel('Max Absolute Difference')
            ax.set_ylabel('Probability Shift from Reference Winner')
            game_name = "Game 1 (AB)" if game_idx == 0 else "Game 2 (BA)"
            ax.set_title(f"{game_name}: Max Diff vs Prob Shift\n(Red=Flip, Blue=Match)")
            ax.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.6, label='Verdict Flipped'),
                Patch(facecolor='blue', alpha=0.6, label='Verdict Matches')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plot_file5 = plot_dir / f"flip_analysis_scatter_{judge_model.replace('/', '-')}.png"
        plt.savefig(plot_file5, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {plot_file5}")
        plt.close()
        
        logger.info("Visualizations generated successfully")
    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")
    
    logger.info("=" * 80)
    logger.info("REPRODUCTION EXPERIMENT COMPLETE")
    logger.info("=" * 80)


def extract_abt_logprobs(
    logprobs_dict: Dict[int, float],
    tokenizer,
    logger: logging.Logger = None,
) -> Dict[str, float]:
    """
    DEPRECATED: Use extract_abt_probs_from_logprobs instead.
    
    This function is kept for backwards compatibility but is no longer used.
    The new approach scans the token sequence to find the first A/B/T token,
    which is more robust than assuming it's at a fixed position.
    """
    logger.warning("extract_abt_logprobs is deprecated, use extract_abt_probs_from_logprobs")
    return {"A": 0.33, "B": 0.33, "T": 0.34}



# ######################
# ### TEST UTILITIES ###
# ######################

def load_llm_sp_example(benchmark: str = "math500", judge_model_family: str = "llama") -> Optional[Dict]:
    """
    Load a real example from llm-sp/sp evaluation data.

    Args:
        benchmark (str): Benchmark name ("mmlu", "mbpp-plus", or "math500").
        judge_model_family (str): Judge model family ("llama", "gemma", "qwen").

    Returns:
        Optional[Dict]: Single example record if found, None otherwise.

    Raises:
        None: Returns None gracefully if data files not found.
    """
    data_dir = Path("llm-sp/sp") / benchmark / judge_model_family
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        return None
    
    # Load first example from first file
    try:
        with open(jsonl_files[0], 'r', encoding='utf-8') as f:
            line = f.readline()
            if line:
                return json.loads(line)
    except Exception as e:
        return None
    
    return None


def run_minimal_test(config: Dict = None, logger: logging.Logger = None) -> bool:
    """
    Run comprehensive validation tests using real llm-sp data.

    Args:
        config (Dict): Configuration dict (default: CONFIG).
        logger (logging.Logger): Logger instance (default: module logger).

    Returns:
        bool: True if all tests pass, False otherwise.

    Tests Performed:
    1. Helper functions: Thinking token removal, A/B/T probability extraction
    2. Real data validation: Load actual llm-sp examples and validate structure
    3. Probability validation: Ensure normalization produces valid results
    4. Robust verdict parsing: Test token variants (space-prefixed, etc.)
    5. Configuration completeness: Identify UNKNOWN values

    Uses real data from llm-sp/sp directory for integration validation.
    """
    if logger is None:
        logger = logging.getLogger("paper_reproduction")
    if config is None:
        config = CONFIG
    
    logger.info("=" * 80)
    logger.info("RUNNING COMPREHENSIVE VALIDATION TEST")
    logger.info("=" * 80)
    
    # ===== PHASE 1: Unit tests on helper functions =====
    logger.info("\nPHASE 1: Helper Function Validation")
    logger.info("-" * 80)
    
    # Test 1: Thinking token removal
    test_text_with_think = "<think>This is reasoning</think>The answer is correct."
    cleaned = remove_thinking_tokens(test_text_with_think, config)
    logger.info(f"Remove thinking tokens:")
    logger.info(f"  Input:  '{test_text_with_think}'")
    logger.info(f"  Output: '{cleaned}'")
    assert cleaned == "The answer is correct.", "Thinking token removal failed"
    logger.info("  ✓ PASS: Thinking token removal works correctly")
    
    # Test 2: Robust A/B/T probability extraction
    logger.info("\nRobust A/B/T Extraction:")
    # Simulate logprobs with token text variants
    mock_text_prob_map = {
        "A": 0.8,
        " B": 0.15,  # Space-prefixed variant
        "▁T": 0.05,  # Alternative space representation
    }
    abt_result = only_abt_probs(mock_text_prob_map, logger)
    logger.info(f"  Input text->prob map: {mock_text_prob_map}")
    logger.info(f"  Extracted A/B/T: {abt_result}")
    assert abt_result["A"] > 0.7, "A probability extraction failed"
    assert abt_result["B"] > 0.1, "B probability extraction failed"
    logger.info("  ✓ PASS: Robust A/B/T extraction handles variants")
    
    # Test 3: Model classification
    logger.info("\nModel Classification:")
    test_models = [
        ("meta-llama/Llama-3.1-8B-Instruct", False),
        ("google/gemma-2-9b-it", False),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", True),
    ]
    for model_id, expected_reasoning in test_models:
        is_reasoning = is_reasoning_model(model_id)
        status = "✓" if is_reasoning == expected_reasoning else "✗"
        logger.info(f"  {status} {model_id}: reasoning={is_reasoning}")
        assert is_reasoning == expected_reasoning, f"Model {model_id} classification failed"
    logger.info("  ✓ PASS: Model classification works correctly")
    
    # ===== PHASE 2: Real data integration validation =====
    logger.info("\n\nPHASE 2: Real Data Integration Validation")
    logger.info("-" * 80)
    
    logger.info("Loading real example from llm-sp/sp...")
    example = load_llm_sp_example("mmlu", "llama")
    
    if example is None:
        logger.error("✗ FAIL: Could not load real data from llm-sp/sp")
        logger.info("=" * 80)
        return False
    
    logger.info("✓ Loaded real MMLU example (Llama judge model)")
    logger.info(f"  Unique ID: {example.get('unique_id')[:8]}...")
    logger.info(f"  Problem length: {len(example.get('problem', ''))} chars")
    logger.info(f"  Assistant 1 answer length: {len(example.get('assistent_1_answer', ''))} chars")
    logger.info(f"  Assistant 2 answer length: {len(example.get('assistent_2_answer', ''))} chars")
    
    # Extract ground truth probabilities from data
    game_1_spb = example.get("game_1_spb_score", [0.0, 0.0, 0.0])
    game_2_spb = example.get("game_2_spb_score", [0.0, 0.0, 0.0])
    
    game_1_probs = {
        "assistant_1": game_1_spb[0],
        "assistant_2": game_1_spb[1],
        "tie": game_1_spb[2],
    }
    
    game_2_probs = {
        "assistant_1": game_2_spb[0],
        "assistant_2": game_2_spb[1],
        "tie": game_2_spb[2],
    }
    
    logger.info(f"\nGround truth probabilities from data:")
    logger.info(f"  Format: [assistant_1, assistant_2, tie] (response identity, not label position)")
    logger.info(f"  Game 1 (order AB, asst1=A, asst2=B):")
    logger.info(f"    asst_1={game_1_probs['assistant_1']:.6f}, asst_2={game_1_probs['assistant_2']:.6f}, tie={game_1_probs['tie']:.6f}")
    logger.info(f"  Game 2 (order BA, asst1=B, asst2=A):")
    logger.info(f"    asst_1={game_2_probs['assistant_1']:.6f}, asst_2={game_2_probs['assistant_2']:.6f}, tie={game_2_probs['tie']:.6f}")
    
    # Verify probabilities sum to 1
    game_1_sum = sum(game_1_spb)
    game_2_sum = sum(game_2_spb)
    logger.info(f"  Probability sums:")
    logger.info(f"    Game 1: {game_1_sum:.6f} (should be ~1.0)")
    logger.info(f"    Game 2: {game_2_sum:.6f} (should be ~1.0)")
    
    assert abs(game_1_sum - 1.0) < 1e-3, f"Game 1 probabilities don't sum to 1: {game_1_sum}"
    assert abs(game_2_sum - 1.0) < 1e-3, f"Game 2 probabilities don't sum to 1: {game_2_sum}"
    logger.info("  ✓ PASS: Ground truth probabilities are properly normalized")
    
    # ===== PHASE 3: Configuration completeness check =====
    logger.info("\n\nPHASE 3: Configuration Completeness Check")
    logger.info("-" * 80)
    
    unknown_keys = [k for k, v in config.items() if "UNKNOWN" in str(v)]
    if unknown_keys:
        logger.warning(f"Configuration has {len(unknown_keys)} UNKNOWN values that must be filled:")
        for key in unknown_keys:
            logger.warning(f"  ⚠ {key}: {_truncate(config[key], 80)}")
    else:
        logger.info("✓ PASS: All configuration values are defined (no UNKNOWNs)")
    
    # ===== SUMMARY =====
    logger.info("\n" + "=" * 80)
    logger.info("ALL VALIDATION TESTS PASSED ✓")
    logger.info("=" * 80)
    logger.info("\nSummary:")
    logger.info("  ✓ Helper functions work correctly")
    logger.info("  ✓ Real data loads and validates")
    logger.info("  ✓ Robust A/B/T extraction handles token variants")
    logger.info("  ✓ Probability normalization is correct")
    if unknown_keys:
        logger.warning(f"\nNext step: Fill in {len(unknown_keys)} missing configuration values")
    logger.info("=" * 80)
    
    return True



# ######################
# ### MAIN ENTRY     ###
# ######################

def main():
    """
    Main entry point for paper reproduction pipeline.

    Args:
        None (uses command-line arguments)

    Returns:
        None

    Command-line Arguments:
        --test: Run minimal test without full vLLM setup
        --model: Judge model to use (default: Qwen/Qwen2.5-3B-Instruct)
        --data: Input JSONL file with response pairs
        --output: Output directory for results (default: reproduction_results)
        --n_samples: Number of samples to test (default: all)
        --seed: Random seed for reproducibility (default: 42)

    Raises:
        SystemExit: If required arguments missing or errors encountered.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce paper experiments with vLLM and detailed logging"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run minimal test without full vLLM setup"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Judge model to use"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Input JSONL file with response pairs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reproduction_results",
        help="Output directory"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark name (e.g., math500, mmlu) - enables llm-sp-reprod structure"
    )
    parser.add_argument(
        "--judge_family",
        type=str,
        default=None,
        help="Judge family name (e.g., llama, qwen) - required with --benchmark"
    )
    parser.add_argument(
        "--judge_short",
        type=str,
        default=None,
        help="Judge model short name (e.g., llama-3.1-8b) - required with --benchmark"
    )
    parser.add_argument(
        "--evaluatee_short",
        type=str,
        default=None,
        help="Evaluatee model short name (e.g., gpt-4o) - required with --benchmark"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples to test (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip generating plots (for automated debugging)"
    )
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        default="none",
        choices=["none", "cot", "long_cot"],
        help="Reasoning mode for verdict generation: 'none' (direct A/B/T), 'cot' (chain-of-thought), 'long_cot' (extended reasoning with <think>)"
    )
    parser.add_argument(
        "--tensor_parallel",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism (e.g., 2 or 4 for 70B models). Overrides config if set."
    )
    
    args = parser.parse_args()
    load_dotenv()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    if args.test:
        logger.info("Running test mode")
        run_minimal_test(CONFIG, logger)
        return
    
    if not args.data:
        logger.error("--data argument required (unless --test is used)")
        sys.exit(1)
    
    # Validate llm-sp-reprod structure requirements
    if args.benchmark or args.judge_family or args.judge_short or args.evaluatee_short:
        if not all([args.benchmark, args.judge_family, args.judge_short, args.evaluatee_short]):
            logger.error("When using llm-sp-reprod structure, all of --benchmark, --judge_family, --judge_short, and --evaluatee_short are required")
            sys.exit(1)
    
    # Run reproduction experiment
    run_reproduction_experiment(
        judge_model=args.model,
        data_path=Path(args.data),
        output_dir=output_dir,
        benchmark=args.benchmark,
        judge_family=args.judge_family,
        judge_short=args.judge_short,
        evaluatee_short=args.evaluatee_short,
        reasoning_mode=args.reasoning_mode,
        n_samples=args.n_samples,
        seed=args.seed,
        config=CONFIG,
        logger=logger,
        tensor_parallel_size=args.tensor_parallel,
    )


if __name__ == "__main__":
    main()
