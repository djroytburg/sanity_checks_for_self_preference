"""
Cache residual stream activations from a model and dataset.
"""
# last modified by dani
# key changes: refactor for dbg, improved caching logic with overwrite and loading,
# added prepare_texts function for dbg dataset formatting
# archived old caching logic at bottom
# modified 01/12/2025 6:31PM EST

import argparse
import copy
import datetime
import json
import pickle
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from run_judge_swap_null_dbg import format_preference_prompt, get_hf_model_path

def load_jsonl(filepath: str):
    """Load a JSONL file and return a list of dictionaries."""
    import json

    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_tokenizer_and_model(
    model_name: str,
    dataset: str,
    device: str = 'auto',
):
    """Load tokenizer and model from HuggingFace."""
    if dataset == 'dbg':
        is_instruct = "Instruct" in model_name or "-it" in model_name or "-chat" in model_name

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if "gemma-2" in model_name.lower():
            tokenizer.padding_side = "right"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map='auto',
        )
        model.eval()
        return tokenizer, model
    
def extract_residual_activations(
    judge_model_path: str,
    paper: str,
    dataset: str,
    text_dict: list[dict],
    max_length: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    layer_indices: Optional[list[int]] = None,
    save_cache: bool = False,
    save_dir: Optional[str] = None,
    overwrite: bool = False,
    batch_size: int = 8,
) -> dict:
    """Extract residual stream activations from a model.
    Args:
        judge_model_path (str): HuggingFace model name or path.
        paper (str): Name of paper (e.g., 'dbg').
        dataset (str): Name of dataset (e.g., 'alpaca_eval').
        text_dict (list[dict]): List of dictionaries containing prompts and metadata.
        max_length (int): Maximum sequence length for processing.
        device (str): Device to run on (default: cuda if available).
        layer_indices (Optional[list[int]]): List of layer indices to extract (default: all).
        save_cache (bool): Whether to save cached activations.
        save_dir (Optional[str]): Directory to save cached activations.
        overwrite (bool): Whether to overwrite existing cached activations.
        batch_size (int): Batch size for processing.
    Returns:
        dict: Dictionary containing metadata and cached activations.
    """

    # --------
    # save_dir logic
    # recommended for caching
    # --------

    if save_cache and save_dir is not None: # Make sure we have calid args
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / f"{judge_model_path}_activations.pkl"
        
        if save_file.exists() and not overwrite: # Cache hit -- reuse depending on overwrite
            current_cache = pickle.load(open(save_file, "rb"))
            cache_metadata = current_cache.get("metadata", {})
            assert cache_metadata.get("judge_model", "") == judge_model_path, "Model name in cache does not match."
            assert cache_metadata.get("dataset", "") == dataset, "Dataset in cache does not match."
            assert cache_metadata.get("paper", "") == paper, "Paper in cache does not match."
            
            already_in_cache = current_cache.get("data", {})
            print(f"Loading existing cache from {save_file}, {len(already_in_cache)} samples found.")
        else:
            if save_file.exists(): # Wipe current cache
                print(f"Warning: Overwriting existing cache at {save_file}.")
            already_in_cache = dict()
            cache_metadata = {
                "judge_model": judge_model_path,
                "dataset": dataset,
                "paper": paper,
                "hyperparameters": {
                    "max_length": max_length,
                    "layer_indices": layer_indices,
                }
            }
    elif save_cache and save_dir is None: # Failure mode
        raise ValueError("If save_cache is True, save_dir must be provided.")
    else: # No caching
        already_in_cache = dict()
        cache_metadata = {
                "judge_model": judge_model_path,
                "dataset": dataset,
                "paper": paper,
                "hyperparameters": {
                    "max_length": max_length,
                    "layer_indices": layer_indices,
                }
            }
    
    # ----------------
    # main execution logic
    # per paper 
    # ----------------
    if paper == "dbg":
        
        # implemented wrt dbg
        judge_model_path = get_hf_model_path(judge_model_path)
        tokenizer, model = load_tokenizer_and_model(
            model_name=judge_model_path,
            dataset='dbg',
            device=device,
        )

        # adding metadata
        n_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        cache_metadata['hyperparameters'].update({
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
        })
        print(f"Model has {n_layers} layers, hidden_dim={hidden_dim}")
        print(f"Extracting activations from layers: {layer_indices}")

        if layer_indices is None:
            layer_indices = list(range(n_layers))
            print(f"Using all layers: {layer_indices}")
        else:
            print(f"Using specified layers: {layer_indices}")

        # --------
        # generate
        # --------

        with torch.no_grad():
            for i, text in enumerate(text_dict):
                
                # Check cache
                if (text['id'], text['judge'], text['reference']) in already_in_cache:
                    if already_in_cache[(text['id'], text['judge'], text['reference'])]['layer_indices'] != layer_indices:
                        print(f"Warning: Cached sample {text['id']} has different number of layers than requested. Recomputing.")
                    else:
                        print(f"Skipping sample {text['id']} with ref {text['reference']}--already in cache.")
                        continue  # Skip already cached samples
                
                cache_data = copy.deepcopy(text) #could be too expensive? TODO
                
                # ---- forward ----
                inputs = tokenizer(
                        [text['forward_prompt']],
                        return_tensors="pt",
                        add_special_tokens=False
                    ).to(model.device)
                outputs = model(**inputs, output_hidden_states=True)

                #   Extract residual stream activations (last token of each sequence)
                #   hidden_states is a tuple of (n_layers + 1) tensors of shape [batch, seq_len, hidden_dim]
                #   Index 0 is embeddings, 1..n_layers are layer outputs
                cache_data['forward_gen_text'] = tokenizer.decode(outputs.text[0].detach().cpu(), skip_special_tokens=False)
                hidden_states = outputs.hidden_states

                #   Get last token position for each sequence (before padding)
                attention_mask = inputs["attention_mask"]
                last_token_indices = attention_mask.sum(dim=1) - 1  # [batch_size]

                
                last_pos = last_token_indices[0].item()

                # Extract activations from requested layers
                sample_acts = []
                for layer_idx in layer_indices:
                    # hidden_states[0] is embeddings, hidden_states[layer_idx+1] is layer output
                    layer_output = hidden_states[layer_idx + 1][0, last_pos, :]
                    sample_acts.append(layer_output.cpu().float().numpy())

                cache_data['forward_activations'] = np.stack(sample_acts, axis=0)  # [n_layers, hidden_dim]
                del outputs  # Free memory
                torch.cuda.empty_cache()


                # ---- backward ----
                inputs = tokenizer(
                        [text['backward_prompt']],
                        return_tensors="pt",
                        add_special_tokens=False
                    ).to(model.device)
                outputs = model(**inputs, output_hidden_states=True)
                cache_data['backward_gen_text'] = tokenizer.decode(outputs.text[0].detach().cpu(), skip_special_tokens=False)

                hidden_states = outputs.hidden_states
                attention_mask = inputs["attention_mask"]
                last_token_indices = attention_mask.sum(dim=1) - 1  # [batch_size]
                last_pos = last_token_indices[0].item()
                sample_acts = []
                
                for layer_idx in layer_indices:
                    layer_output = hidden_states[layer_idx + 1][0, last_pos, :]
                    sample_acts.append(layer_output.cpu().float().numpy())
                
                cache_data['backward_activations'] = np.stack(sample_acts, axis=0)  # [n_layers, hidden_dim]
                cache_data['layer_indices'] = layer_indices
                del outputs  # Free memory
                torch.cuda.empty_cache()

                # ---- add to cache dict ----
                already_in_cache[(text['id'], text['judge'], text['reference'])] = cache_data
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} / {len(text_dict)} samples.")
                
                # ---- save intermittently ----
                if save_cache and save_dir is not None and (i + 1) % 100 == 0:
                    save_cache_file = save_dir / f"{judge_model_path}_activations.pkl"
                    with open(save_cache_file, "wb") as f:
                        pickle.dump({
                            "metadata": cache_metadata,
                            "data": already_in_cache,
                        }, f)
    else:
        raise NotImplementedError(f"Paper '{paper}' not supported for activation extraction.")
    
    return {
        "metadata": cache_metadata,
        "data": already_in_cache,
    }

def prepare_texts(paper: str, dataset: str, data_dir: str, judge_model: str):
    """
    Prepare activation extraction for a dataset and a paper's code.
    Args:
        paper (str): The paper name, e.g., 'dbg'.
        dataset (str): The dataset name, e.g., 'alpaca_eval'.
        data_dir (str): The base directory containing datasets and model responses. For DBG, this would be the root directory where 'model_preferences_fullset' and 'model_responses_fullset' are located.
        judge_model (str): The model used for judging, e.g., 'Qwen3-0.6B'. Do not include provider name.
    Returns:
        texts (List[dict]): List of dictionaries containing formatted prompts in the following form:
            {
                "judge": judge_model,
                "reference": reference_model,
                "id": idx,
                "gold_label": gold_label,
                "self_label": self_label,
                "forward_prompt": chat_templated_1,
                "backward_prompt": chat_templated_2,
                "dataset": dataset,
                "paper": paper,
                "raw_data": { ... }  # Additional raw data for reference
            }

    """
    if paper == 'dbg':
        model_path = get_hf_model_path(judge_model)
        is_instruct = "Instruct" in model_path or "-it" in model_path or "-chat" in model_path
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if "gemma-2" in model_path.lower():
            tokenizer.padding_side = "right"

        if dataset in ['alpaca_eval', 'translation', 'truthfulness']:
            preference_dir = Path(data_dir) / 'model_preferences_fullset' / f"{dataset}_500_id_wr"
            evaluator_subdir = f"evaluator_{judge_model}"
            if evaluator_subdir not in [d.name for d in preference_dir.iterdir() if d.is_dir()]:
                raise ValueError(f"Evaluator subdirectory '{evaluator_subdir}' not found in '{preference_dir}'.")
            evaluator_subdir = preference_dir / evaluator_subdir

            gold_responses_path = preference_dir / 'gold_aggregated.json'
            gold_data = json.load(open(gold_responses_path, 'r'))

            response_dir = Path(data_dir) / 'model_responses_fullset' / f"{dataset}"
            response_path = response_dir / f"{judge_model}.jsonl"
            response_data = load_jsonl(response_path)

            # Iteration through preference directory tree
            texts = []

            for preference_data_file in evaluator_subdir.iterdir():
                if preference_data_file.suffix != '.jsonl': # Shouldn't happen, but harmless to ignore.
                    continue
                if not preference_data_file.name.startswith('average_'):
                    raise ValueError(f"Unexpected file name: {preference_data_file.name}")
                _, model1, model2 = preference_data_file.stem.split('_')
                if model1 == judge_model:
                    reference_model = model2
                elif model2 == judge_model:
                    reference_model = model1
                else:
                    raise ValueError(f"Judge model '{judge_model}' not found in file name: {preference_data_file.stem}")
                
                #gold_data file for dbg is sorted by filenames relative to the gold judge directories.
                gold_data_keys = [
                    fname for fname in gold_data.keys()
                    if fname in [f'merge_{reference_model}_{judge_model}.jsonl', f'merge_{judge_model}_{reference_model}.jsonl']
                ]                
                if len(gold_data_keys) == 0:
                    raise ValueError(f"No matching gold data keys found for models '{reference_model}' and '{judge_model}'")
                elif len(gold_data_keys) > 1:
                    raise ValueError(f"Multiple matching gold data keys found for models '{reference_model}' and '{judge_model}': {gold_data_keys}")
                gold_data_key = gold_data_keys[0]
                j_r_gold_data = gold_data[gold_data_key]
                
                preference_data = load_jsonl(preference_data_file)

                for ex in preference_data:
                    idx = ex['id']
                    gold_lookup = next((item for item in j_r_gold_data if item['id'] == idx), None)
                    
                    if gold_lookup is None:
                        raise ValueError(f"ID '{idx}' not found in gold data for key '{gold_data_key}'")
                    assert gold_lookup[f"{judge_model}_response"] == ex[f"{judge_model}_response"], f"Gold data response does not match preference data for {dataset} {gold_data_key} {idx}."
                    assert gold_lookup[f"{reference_model}_response"] == ex[f"{reference_model}_response"], f"Gold data response does not match preference data for {dataset} {gold_data_key} {idx}."
                    
                    labels = [win for gold_judge in gold_lookup['preferences'].values() for win in gold_judge]
                    if labels.count(judge_model) == labels.count(reference_model):
                        continue  # Skip ties
                    
                    gold_label = 'lsp' if labels.count(judge_model) > labels.count(reference_model) else 'ilsp'
                    
                    assert isinstance(ex['preferences'], str), f"Preference data 'preferences' field should be a string indicating the winning model. {ex['preferences']}"
                    self_label = 'self' if ex['preferences'] == judge_model else 'other'
                    
                    response_lookup = next((item for item in response_data if item['id'] == idx), None)
                    if response_lookup is None:
                        raise ValueError(f"ID '{idx}' not found in response data for model '{response_path}'")
                    assert response_lookup['model_response'] == ex[f"{judge_model}_response"], f"Response data does not match preference data for {dataset} {gold_data_key} {idx}."
                    query_key = 'german' if dataset == 'translation' else 'query'
                    
                    input_text_forward = format_preference_prompt(
                        query=response_lookup[query_key],
                        response1=ex[f"{judge_model}_response"],
                        response2=ex[f"{reference_model}_response"],
                        dataset=dataset,
                        model_name=judge_model
                    )

                    input_text_backward = format_preference_prompt(
                        query=response_lookup[query_key],
                        response1=ex[f"{reference_model}_response"],
                        response2=ex[f"{judge_model}_response"],
                        dataset=dataset,
                        model_name=judge_model
                    )
                    if is_instruct:
                        chat_templated_1 = tokenizer.apply_chat_template(
                            input_text_forward,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        chat_templated_2 = tokenizer.apply_chat_template(
                            input_text_backward,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        chat_templated_1 = ""
                        for m in input_text_forward:
                            if m["role"] == "system":
                                chat_templated_1 += m["content"] + "\n\n"
                            elif m["role"] == "user":
                                chat_templated_1 += m["content"]
                        
                        chat_templated_2 = ""
                        for m in input_text_backward:
                            if m["role"] == "system":
                                chat_templated_2 += m["content"] + "\n\n"
                            elif m["role"] == "user":
                                chat_templated_2 += m["content"]
                    
                    text_data = {
                        "judge": judge_model,
                        "reference": reference_model,
                        "id": idx,
                        "gold_label": gold_label,
                        "self_label": self_label,
                        "forward_prompt": chat_templated_1,
                        "backward_prompt": chat_templated_2,
                        "dataset": dataset,
                        "paper": paper,
                        "raw_data": {
                            "from_files": {
                                "preference_file": str(preference_data_file),
                                "response_file": str(response_path),
                                "gold_file": str(gold_responses_path)
                            },
                            "query": response_lookup[query_key],
                            "judge_response": ex[f"{judge_model}_response"],
                            "reference_response": ex[f"{reference_model}_response"],
                            "gold_preferences": gold_lookup['preferences'],
                            "pre_template_prompts": {
                                "forward": input_text_forward,
                                "backward": input_text_backward
                            }
                        }
                    }

                    texts.append(text_data)
            return texts
        else:
            raise NotImplementedError(f"Dataset '{dataset}' not supported for paper '{paper}'.")
    else:
        raise NotImplementedError(f"Paper '{paper}' not supported.")


def main():
    parser = argparse.ArgumentParser(
        description="Cache residual stream activations from a model and dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--paper",
        type=str,
        required=True,
        help="name of paper (e.g., 'dbg')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset (e.g., 'alpaca_eval')",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Base data directory containing datasets and model responses",
    )
    parser.add_argument(
        "--dont_save_cache",
        action="store_true",
        default=False,
        help="Whether to not save cached activations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output dir for cache. dataset, paper, model deets added after",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing cached activations",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for pref generation (default: 512)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices to extract (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )

    args = parser.parse_args()
    load_dotenv()
    # Load dataset
    texts = prepare_texts(
        paper=args.paper,
        dataset=args.dataset,
        data_dir=args.data_dir,
        judge_model=args.model,
    )
    print(f"Total samples loaded: {len(texts)}")

    if args.num_samples is not None and args.num_samples < len(texts):
        texts = texts[:args.num_samples]
        print(f"Processing only first {args.num_samples} samples.")
    # Parse layer indices
    layer_indices = None
    if args.layers:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    # Extract activations
    save_dir = Path(args.output_dir) / args.paper / args.dataset / args.model.replace("/", "-")
    save_dir.mkdir(parents=True, exist_ok=True)

    data = extract_residual_activations(
        judge_model_path=args.model,
        paper=args.paper,
        dataset=args.dataset,
        text_dict=texts,
        max_length=args.max_length,
        device=args.device,
        layer_indices=layer_indices,
        save_cache=not args.dont_save_cache,
        save_dir=save_dir,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )

    print("\nDone!")
    print(f"Extracted activations for {len(data['data'])} samples.")
    if not args.dont_save_cache:
        print(f"Cached activations saved to {save_dir}")

if __name__ == "__main__":
    main()


# --------------------------
# Old caching logic, not used. 
# Archived 01/12/2025 6:31PM EST
# --------------------------


def save_activations(data: dict, output_path: str):
    """Save activations to .pkl file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved activations to {output_path}")


    # else:
    #     print(f"Loading model: {judge_model_path}")
    #     tokenizer = AutoTokenizer.from_pretrained(judge_model_path, trust_remote_code=True)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    #         device_map=device,
    #     )
    #     model.eval()

    #     # Set padding token if not set
    #     if tokenizer.pad_token is None:
    #         tokenizer.pad_token = tokenizer.eos_token

    #     # Get model config
    #     n_layers = model.config.num_hidden_layers
    #     hidden_dim = model.config.hidden_size

    #     if layer_indices is None:
    #         layer_indices = list(range(n_layers))

    #     print(f"Model has {n_layers} layers, hidden_dim={hidden_dim}")
    #     print(f"Extracting activations from layers: {layer_indices}")

    #     # Storage for activations
    #     all_activations = []

    #     # Process in batches
    #     n_batches = (len(texts) + batch_size - 1) // batch_size

    #     with torch.no_grad():
    #         for i in tqdm(range(n_batches), desc="Processing batches"):
    #             batch_texts = texts[i * batch_size : (i + 1) * batch_size]

    #             # Tokenize
    #             inputs = tokenizer(
    #                 batch_texts,
    #                 return_tensors="pt",
    #                 padding=True,
    #                 truncation=True,
    #                 max_length=max_length,
    #             ).to(device)

    #             # Forward pass with output_hidden_states
    #             outputs = model(**inputs, output_hidden_states=True)

    #             # Extract residual stream activations (last token of each sequence)
    #             # hidden_states is a tuple of (n_layers + 1) tensors of shape [batch, seq_len, hidden_dim]
    #             # Index 0 is embeddings, 1..n_layers are layer outputs
    #             hidden_states = outputs.hidden_states

    #             # Get last token position for each sequence (before padding)
    #             attention_mask = inputs["attention_mask"]
    #             last_token_indices = attention_mask.sum(dim=1) - 1  # [batch_size]

    #             batch_acts = []
    #             for batch_idx in range(len(batch_texts)):
    #                 last_pos = last_token_indices[batch_idx].item()

    #                 # Extract activations from requested layers
    #                 sample_acts = []
    #                 for layer_idx in layer_indices:
    #                     # hidden_states[0] is embeddings, hidden_states[layer_idx+1] is layer output
    #                     layer_output = hidden_states[layer_idx + 1][batch_idx, last_pos, :]
    #                     sample_acts.append(layer_output.cpu().float().numpy())

    #                 batch_acts.append(np.stack(sample_acts, axis=0))  # [n_layers, hidden_dim]

    #             all_activations.extend(batch_acts)

        
    #     activations = np.stack(all_activations, axis=0)  # [n_samples, n_layers, hidden_dim]

    #     print(f"Extracted activations shape: {activations.shape}")

    #     metadata = {
    #         "model_name": model_name,
    #         "n_samples": len(texts),
    #         "n_layers": len(layer_indices),
    #         "layer_indices": layer_indices,
    #         "hidden_dim": hidden_dim,
    #         "max_length": max_length,
    #     }

    #     return {
    #         "activations": activations,
    #         "metadata": metadata,
    #     }


def load_dataset(
    dataset_path: str,
    label_key: Optional[str] = None,
) -> tuple[list[str], Optional[list[int]]]:
    """Load dataset from file.

    Supports:
    - .txt: one text per line
    - .jsonl: JSON lines with 'text' field and optional label field

    Args:
        dataset_path: Path to dataset file
        label_key: Key for labels in JSONL (if None, returns None for labels)

    Returns:
        Tuple of (texts, labels)
    """
    import json

    path = Path(dataset_path)
    texts = []
    labels = [] if label_key else None

    if path.suffix == ".txt":
        with open(path, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif path.suffix == ".jsonl":
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                texts.append(data.get("text", ""))
                if label_key and label_key in data:
                    labels.append(data[label_key])
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    print(f"Loaded {len(texts)} samples from {dataset_path}")

    return texts, labels

