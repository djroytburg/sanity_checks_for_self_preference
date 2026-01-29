"""
Cache residual stream activations from a model and dataset.
"""
#DO NOT USE, NOT PART OF FINAL PAPER!!!!!!!
import argparse
import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_residual_activations(
    model_name: str,
    texts: list[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8,
    max_length: int = 512,
    layer_indices: Optional[list[int]] = None,
) -> dict:
    """Extract residual stream activations from a model.

    Returns:
        Dictionary with:
        - 'activations': np.ndarray of shape [n_samples, n_layers, hidden_dim]
        - 'metadata': dict with model info
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get model config
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    if layer_indices is None:
        layer_indices = list(range(n_layers))

    print(f"Model has {n_layers} layers, hidden_dim={hidden_dim}")
    print(f"Extracting activations from layers: {layer_indices}")

    # Storage for activations
    all_activations = []

    # Process in batches
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Processing batches"):
            batch_texts = texts[i * batch_size : (i + 1) * batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            # Forward pass with output_hidden_states
            outputs = model(**inputs, output_hidden_states=True)

            # Extract residual stream activations (last token of each sequence)
            # hidden_states is a tuple of (n_layers + 1) tensors of shape [batch, seq_len, hidden_dim]
            # Index 0 is embeddings, 1..n_layers are layer outputs
            hidden_states = outputs.hidden_states

            # Get last token position for each sequence (before padding)
            attention_mask = inputs["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1  # [batch_size]

            batch_acts = []
            for batch_idx in range(len(batch_texts)):
                last_pos = last_token_indices[batch_idx].item()

                # Extract activations from requested layers
                sample_acts = []
                for layer_idx in layer_indices:
                    # hidden_states[0] is embeddings, hidden_states[layer_idx+1] is layer output
                    layer_output = hidden_states[layer_idx + 1][batch_idx, last_pos, :]
                    sample_acts.append(layer_output.cpu().float().numpy())

                batch_acts.append(np.stack(sample_acts, axis=0))  # [n_layers, hidden_dim]

            all_activations.extend(batch_acts)

    
    activations = np.stack(all_activations, axis=0)  # [n_samples, n_layers, hidden_dim]

    print(f"Extracted activations shape: {activations.shape}")

    metadata = {
        "model_name": model_name,
        "n_samples": len(texts),
        "n_layers": len(layer_indices),
        "layer_indices": layer_indices,
        "hidden_dim": hidden_dim,
        "max_length": max_length,
    }

    return {
        "activations": activations,
        "metadata": metadata,
    }


def load_dataset(
    dataset_path: str = None,
    dataset_name: str = None,
    label_key: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> tuple[list[str], Optional[list[int]]]:
    """Load dataset from file or HuggingFace datasets.

    Supports:
    - .txt: one text per line
    - .jsonl: JSON lines with 'text' field and optional label field
    - HuggingFace datasets: 'xsum' or 'cnn' (cnn_dailymail)

    Args:
        dataset_path: Path to dataset file (if loading from file)
        dataset_name: Name of HuggingFace dataset ('xsum' or 'cnn')
        label_key: Key for labels in JSONL (if None, returns None for labels)
        max_samples: Maximum number of samples to load (if None, load all)

    Returns:
        Tuple of (texts, labels)
    """
    import json

    texts = []
    labels = None

    if dataset_name:
        # Load from HuggingFace datasets
        from datasets import load_dataset as hf_load_dataset

        if dataset_name.lower() == 'xsum':
            print(f"Loading XSum dataset...")
            dataset = hf_load_dataset('xsum', split='train')
            texts = [item['document'] for item in dataset]
        elif dataset_name.lower() == 'cnn':
            print(f"Loading CNN/DailyMail dataset...")
            dataset = hf_load_dataset('cnn_dailymail', '3.0.0', split='train')
            texts = [item['article'] for item in dataset]
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'xsum' or 'cnn'")

        if max_samples and max_samples < len(texts):
            texts = texts[:max_samples]
            print(f"Limited to {max_samples} samples")

    elif dataset_path:
        # Load from file
        path = Path(dataset_path)
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

        if max_samples and max_samples < len(texts):
            texts = texts[:max_samples]
            if labels:
                labels = labels[:max_samples]
            print(f"Limited to {max_samples} samples")
    else:
        raise ValueError("Must provide either dataset_path or dataset_name")

    print(f"Loaded {len(texts)} samples")

    return texts, labels


def save_activations(data: dict, output_path: str, save_per_layer: bool = False):
    """Save activations to .pkl file(s).

    Args:
        data: Dictionary with 'activations' and 'metadata'
        output_path: Base output path for .pkl file(s)
        save_per_layer: If True, save a separate .pkl file for each layer
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_per_layer:
        # Save each layer separately
        activations = data['activations']  # [n_samples, n_layers, hidden_dim]
        metadata = data['metadata']
        layer_indices = metadata['layer_indices']

        # Create base filename without extension
        base_path = output_path.with_suffix('')

        for i, layer_idx in enumerate(layer_indices):
            layer_activations = activations[:, i, :]  # [n_samples, hidden_dim]

            layer_data = {
                'activations': layer_activations,
                'metadata': {
                    **metadata,
                    'layer_index': layer_idx,
                    'n_layers': 1,
                    'layer_indices': [layer_idx],
                }
            }

            layer_output_path = Path(f"{base_path}_layer_{layer_idx}.pkl")
            with open(layer_output_path, "wb") as f:
                pickle.dump(layer_data, f)

            print(f"Saved layer {layer_idx} activations to {layer_output_path}")
    else:
        # Save all layers in one file
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved activations to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cache residual stream activations from a model and dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path (default: meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset file (.txt or .jsonl)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        choices=['xsum', 'cnn'],
        help="HuggingFace dataset name: 'xsum' or 'cnn' (cnn_dailymail)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for .pkl file",
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
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
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
        "--save_per_layer",
        action="store_true",
        help="Save a separate .pkl file for each layer (default: False)",
    )

    args = parser.parse_args()

    # Validate dataset arguments
    if not args.dataset and not args.dataset_name:
        parser.error("Must provide either --dataset or --dataset_name")
    if args.dataset and args.dataset_name:
        parser.error("Cannot provide both --dataset and --dataset_name")

    # Load dataset
    texts, _ = load_dataset(
        dataset_path=args.dataset,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
    )

    # Parse layer indices
    layer_indices = None
    if args.layers:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    # Extract activations
    data = extract_residual_activations(
        model_name=args.model,
        texts=texts,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        layer_indices=layer_indices,
    )

    # Save to .pkl
    save_activations(data, args.output, save_per_layer=args.save_per_layer)

    print("\nDone!")
    print(f"Activations shape: {data['activations'].shape}")
    print(f"Metadata: {data['metadata']}")


if __name__ == "__main__":
    main()
