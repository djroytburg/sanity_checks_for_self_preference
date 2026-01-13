"""
Train probes on per-layer activations from cache_activations.py and plot accuracy.

Hyperparameters from: "Detecting Strategic Deception Using Linear Probes" (arXiv:2502.03407)
https://github.com/ApolloResearch/deception-detection
"""

import argparse
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ProbeConfig:
    """Configuration for probe training."""
    probe_type: Literal["lr", "mlp"] = "lr"

    # Logistic Regression hyperparameters
    reg_coeff: float = 1e3  # C = 1/reg_coeff
    normalize: bool = True  # Normalize activations before training

    # MLP hyperparameters
    hidden_dim: int = 64
    lr: float = 0.0001
    epochs: int = 10000
    val_split: float = 0.2  # 80% train, 20% validation
    early_stopping: bool = True

    # PCA settings
    use_pca: bool = False  # Whether to apply PCA
    pca_n_components: Optional[int] = None  # Number of PCA components (None = use variance)
    pca_variance: Optional[float] = None  # Explained variance ratio (e.g., 0.95 for 95%)
    pca_whiten: bool = False  # Whether to whiten PCA components

    # General settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 42
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    test_split: float = 0.2  # Train/test split


class MLPProbe(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LinearProbe:
    """
    From arXiv:2502.03407:
    - reg_coeff: float = 1e3 (default)
    - normalize: bool = True (default)
    - sklearn config: C=1/reg_coeff, random_state=42, fit_intercept=False
    """

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.mean = None
        self.std = None
        self.model = LogisticRegression(
            C=1 / config.reg_coeff,
            random_state=config.random_state,
            fit_intercept=False,
            max_iter=1000
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        """Fit the logistic regression probe."""
        if self.config.normalize:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0) + 1e-8
            X = (X - self.mean) / self.std

        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.config.normalize and self.mean is not None:
            X = (X - self.mean) / self.std
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.config.normalize and self.mean is not None:
            X = (X - self.mean) / self.std
        return self.model.predict_proba(X)[:, 1]

    def get_direction(self) -> np.ndarray:
        """Get the probe direction (coefficients)."""
        return self.model.coef_[0]


class MLPProbeTrainer:
    """Trainer for MLP probes.

    From arXiv:2502.03407:
    - Optimizer: AdamW with lr=0.0001
    - Loss: BCELoss
    - Epochs: 10000 with early stopping
    - Val split: 80% train, 20% val
    """

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.model = None
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPProbeTrainer":
        """Train the MLP probe."""

        if self.config.normalize:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0) + 1e-8
            X = (X - self.mean) / self.std

        X_tensor = torch.tensor(X, dtype=self.config.dtype)
        y_tensor = torch.tensor(y, dtype=self.config.dtype).unsqueeze(1)

        n_val = int(len(X) * self.config.val_split)
        indices = torch.randperm(len(X), generator=torch.Generator().manual_seed(self.config.random_state))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
        X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]

        # Move to device
        device = torch.device(self.config.device)
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)

        input_dim = X.shape[1]
        self.model = MLPProbe(input_dim, self.config.hidden_dim).to(device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        criterion = nn.BCELoss()

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.config.epochs):
            # Training step
            self.model.train()
            optimizer.zero_grad()
            y_pred = self.model(X_train)
            train_loss = criterion(y_pred, y_train)
            train_loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            # Early stopping
            if self.config.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                elif best_state is not None:
                    # Val loss increased, stop training
                    self.model.load_state_dict(best_state)
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.config.normalize and self.mean is not None:
            X = (X - self.mean) / self.std

        X_tensor = torch.tensor(X, dtype=self.config.dtype).to(self.config.device)

        self.model.eval()
        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy().squeeze()

        return proba


def load_per_layer_activations(base_path: str, layer_indices: Optional[list[int]] = None) -> dict[int, np.ndarray]:
    """Load per-layer activations from .pkl files.

    Supports multiple patterns:
    1. Directory path: Loads all *_layer_*.pkl files from directory
    2. Specific file path: Loads all files with same prefix (e.g., 'lsp_activations_layer_*.pkl')

    Args:
        base_path: Path to directory containing layer files OR path to a specific layer file
        layer_indices: List of layer indices to load (default: auto-detect from files)

    Returns:
        Dictionary mapping layer index to activation array of shape [n_samples, hidden_dim]
    """
    base_path = Path(base_path).resolve()  # Resolve to absolute path

    # Determine if it's a directory or file
    if base_path.is_dir():
        # It's a directory - load all *_layer_*.pkl files
        parent_dir = base_path
        prefix = None
    elif base_path.is_file():
        # It's a file - extract prefix and directory
        parent_dir = base_path.parent
        stem = base_path.stem
        if '_layer_' in stem:
            prefix = stem.split('_layer_')[0]
        else:
            prefix = stem
    else:
        # Path doesn't exist - try as directory first
        if base_path.exists():
            raise ValueError(f"Path exists but is neither file nor directory: {base_path}")
        # Assume it's meant to be a directory
        parent_dir = base_path
        prefix = None

    activations = {}

    if layer_indices is None:
        # Auto-detect layer files
        if prefix:
            # Specific prefix given
            pattern = f"{prefix}_layer_*.pkl"
            layer_files = sorted(parent_dir.glob(pattern))
        else:
            # No prefix - load all *_layer_*.pkl files
            all_pkl_files = list(parent_dir.glob("*.pkl"))
            layer_files = [f for f in all_pkl_files if "_layer_" in f.name]
            layer_files = sorted(layer_files)

        if not layer_files:
            all_files = list(parent_dir.glob("*.pkl"))
            raise ValueError(f"No layer files found in {parent_dir}. Found {len(all_files)} pkl files total. Directory contents: {[f.name for f in all_files[:5]]}")

        for layer_file in layer_files:
            # Extract layer index from filename
            stem = layer_file.stem
            parts = stem.split('_layer_')
            if len(parts) >= 2:
                try:
                    layer_idx = int(parts[-1])
                    with open(layer_file, "rb") as f:
                        data = pickle.load(f)
                    activations[layer_idx] = data['activations']
                    print(f"Loaded layer {layer_idx} from {layer_file.name}")
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping {layer_file.name}: {e}")
                    continue
    else:
        # Specific layer indices requested
        for layer_idx in layer_indices:
            # Try to find file matching this layer index
            if prefix:
                layer_file = parent_dir / f"{prefix}_layer_{layer_idx}.pkl"
            else:
                # Search for any file with this layer index
                matches = list(parent_dir.glob(f"*_layer_{layer_idx}.pkl"))
                if not matches:
                    raise ValueError(f"Layer file not found for layer {layer_idx} in {parent_dir}")
                layer_file = matches[0]

            if not layer_file.exists():
                raise ValueError(f"Layer file not found: {layer_file}")
            with open(layer_file, "rb") as f:
                data = pickle.load(f)
            activations[layer_idx] = data['activations']
            print(f"Loaded layer {layer_idx} from {layer_file.name}")

    return activations


def load_activations(path: str) -> dict:
    """Load activations from a file.

    Expects either:
    - .npz file with 'activations' key (shape: [n_samples, n_layers, hidden_dim])
    - .pt file with tensor of same shape
    - .pkl file with dict containing 'activations' key (from cache_activations.py)

    Returns dict with layer indices as keys and activation arrays as values.
    """
    path = Path(path)

    if path.suffix == ".npz":
        data = np.load(path)
        activations = data["activations"]
    elif path.suffix == ".pt":
        activations = torch.load(path).numpy()
    elif path.suffix == ".npy":
        activations = np.load(path)
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
        activations = data["activations"]

        # Check if this is a single-layer file (2D) or multi-layer file (3D)
        if activations.ndim == 2:
            # Single layer file, return as dict with layer index
            layer_idx = data['metadata'].get('layer_index', 0)
            return {layer_idx: activations}
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Assume shape is [n_samples, n_layers, hidden_dim]
    n_samples, n_layers, hidden_dim = activations.shape

    return {
        layer_idx: activations[:, layer_idx, :]
        for layer_idx in range(n_layers)
    }


def train_and_evaluate_all_layers(
    positive_acts: dict[int, np.ndarray],
    negative_acts: dict[int, np.ndarray],
    config: ProbeConfig,
    layers: Optional[list[int]] = None,
) -> tuple[dict, dict, dict, dict, dict]:
    """Train probes on all layers and evaluate on train/test split.

    Returns:
        Tuple of (probes, train_results, test_results, layer_indices, pca_models)
    """
    if layers is None:
        layers = sorted(positive_acts.keys())

    probes = {}
    train_results = {}
    test_results = {}
    pca_models = {}

    for layer_idx in layers:
        print(f"Training probe for layer {layer_idx}...")

        pos_acts = positive_acts[layer_idx]
        neg_acts = negative_acts[layer_idx]

        X = np.concatenate([pos_acts, neg_acts], axis=0)
        y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_split, random_state=config.random_state, stratify=y
        )

        # Apply PCA if enabled (fit on train only!)
        pca = None
        if config.use_pca:
            if config.pca_n_components is not None:
                n_components = config.pca_n_components
            elif config.pca_variance is not None:
                n_components = config.pca_variance
            else:
                raise ValueError("PCA enabled but neither n_components nor variance specified")

            pca = PCA(n_components=n_components, whiten=config.pca_whiten, random_state=config.random_state)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            pca_models[layer_idx] = pca

            if config.pca_n_components is not None:
                print(f"  PCA: {pca.n_components_} components, explained variance: {pca.explained_variance_ratio_.sum():.4f}")
            else:
                print(f"  PCA: {pca.n_components_} components for {config.pca_variance:.2%} variance")

        # Train probe
        if config.probe_type == "lr":
            probe = LinearProbe(config)
        else:
            probe = MLPProbeTrainer(config)

        probe.fit(X_train, y_train)
        probes[layer_idx] = probe

        # Evaluate on train set
        y_train_pred = probe.predict(X_train)
        y_train_proba = probe.predict_proba(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_auroc = roc_auc_score(y_train, y_train_proba)
        train_results[layer_idx] = {"accuracy": train_acc, "auroc": train_auroc}

        # Evaluate on test set
        y_test_pred = probe.predict(X_test)
        y_test_proba = probe.predict_proba(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_auroc = roc_auc_score(y_test, y_test_proba)
        test_results[layer_idx] = {"accuracy": test_acc, "auroc": test_auroc}

        print(f"  Train: Accuracy={train_acc:.4f}, AUROC={train_auroc:.4f}")
        print(f"  Test:  Accuracy={test_acc:.4f}, AUROC={test_auroc:.4f}")

    return probes, train_results, test_results, layers, pca_models


def plot_accuracy_over_layers(
    train_results: dict[int, dict],
    test_results: dict[int, dict],
    layers: list[int],
    output_path: str,
    title: str = "Probe Accuracy Over Layers",
):
    """Plot training and test accuracy over all layers."""
    train_accs = [train_results[layer]["accuracy"] for layer in layers]
    test_accs = [test_results[layer]["accuracy"] for layer in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, train_accs, 'b-o', label='Train Accuracy', markersize=4)
    ax.plot(layers, test_accs, 'r-o', label='Test Accuracy', markersize=4)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.05])

    # Add horizontal line at 0.5 for random baseline
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy plot to {output_path}")


def plot_train_accuracy(
    train_results: dict[int, dict],
    layers: list[int],
    output_path: str,
    title: str = "Training Accuracy Over Layers",
):
    """Plot training accuracy over all layers."""
    train_accs = [train_results[layer]["accuracy"] for layer in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, train_accs, 'b-o', label='Train Accuracy', markersize=4)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.05])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training accuracy plot to {output_path}")


def plot_test_accuracy(
    test_results: dict[int, dict],
    layers: list[int],
    output_path: str,
    title: str = "Test Accuracy Over Layers",
):
    """Plot test accuracy over all layers."""
    test_accs = [test_results[layer]["accuracy"] for layer in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, test_accs, 'r-o', label='Test Accuracy', markersize=4)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.05])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved test accuracy plot to {output_path}")


def save_probes(
    probes: dict[int, LinearProbe | MLPProbeTrainer],
    output_dir: str,
    config: ProbeConfig,
):
    """Save trained probes to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, probe in probes.items():
        if config.probe_type == "lr":
            # Save logistic regression probe
            np.savez(
                output_dir / f"probe_layer_{layer_idx}.npz",
                coef=probe.model.coef_,
                mean=probe.mean if probe.mean is not None else np.array([]),
                std=probe.std if probe.std is not None else np.array([]),
            )
        else:
            # Save MLP probe
            torch.save(
                {
                    "state_dict": probe.model.state_dict(),
                    "mean": probe.mean,
                    "std": probe.std,
                    "hidden_dim": config.hidden_dim,
                },
                output_dir / f"probe_layer_{layer_idx}.pt",
            )

    # Save config
    config_dict = {
        "probe_type": config.probe_type,
        "reg_coeff": config.reg_coeff,
        "normalize": config.normalize,
        "hidden_dim": config.hidden_dim,
        "lr": config.lr,
        "epochs": config.epochs,
        "val_split": config.val_split,
        "early_stopping": config.early_stopping,
        "random_state": config.random_state,
        "test_split": config.test_split,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved probes to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train probes on per-layer activations and plot accuracy over layers"
    )
    parser.add_argument(
        "--positive_acts",
        type=str,
        required=True,
        help="Path to positive class activations (base path for per-layer .pkl files, or single file)",
    )
    parser.add_argument(
        "--negative_acts",
        type=str,
        required=True,
        help="Path to negative class activations (base path for per-layer .pkl files, or single file)",
    )
    parser.add_argument(
        "--per_layer",
        action="store_true",
        help="Load per-layer .pkl files (from cache_activations.py --save_per_layer)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./probes",
        help="Directory to save trained probes and plots",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        choices=["lr", "mlp"],
        default="mlp",
        help="Type of probe to train (lr=logistic regression, mlp=MLP)",
    )
    parser.add_argument(
        "--reg_coeff",
        type=float,
        default=1e2,
        help="Regularization coefficient for logistic regression (default: 1e3)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for MLP probe (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00001,
        help="Learning rate for MLP probe (default: 0.0001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20000,
        help="Max epochs for MLP probe (default: 10000)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable activation normalization",
    )
    parser.add_argument(
        "--no_early_stopping",
        action="store_true",
        help="Disable early stopping for MLP",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices to train on (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        default="Probe Accuracy Over Layers",
        help="Title for the plots",
    )
    parser.add_argument(
        "--use_pca",
        action="store_true",
        help="Apply PCA dimensionality reduction before training",
    )
    parser.add_argument(
        "--pca_n_components",
        type=int,
        default=None,
        help="Number of PCA components (default: None, use variance instead)",
    )
    parser.add_argument(
        "--pca_variance",
        type=float,
        default=None,
        help="Explained variance ratio for PCA (e.g., 0.95 for 95%%, default: None)",
    )
    parser.add_argument(
        "--pca_whiten",
        action="store_true",
        help="Whiten PCA components (default: False)",
    )

    args = parser.parse_args()

    # Create config
    config = ProbeConfig(
        probe_type=args.probe_type,
        reg_coeff=args.reg_coeff,
        normalize=not args.no_normalize,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        early_stopping=not args.no_early_stopping,
        random_state=args.seed,
        test_split=args.test_split,
        use_pca=args.use_pca,
        pca_n_components=args.pca_n_components,
        pca_variance=args.pca_variance,
        pca_whiten=args.pca_whiten,
    )

    print(f"Configuration:")
    print(f"  Probe type: {config.probe_type}")
    print(f"  Normalize: {config.normalize}")
    print(f"  Test split: {config.test_split}")
    if config.probe_type == "lr":
        print(f"  Regularization coefficient: {config.reg_coeff}")
    else:
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Learning rate: {config.lr}")
        print(f"  Max epochs: {config.epochs}")
        print(f"  Early stopping: {config.early_stopping}")
    print(f"  Device: {config.device}")
    if config.use_pca:
        print(f"  PCA: Enabled")
        if config.pca_n_components is not None:
            print(f"    Components: {config.pca_n_components}")
        if config.pca_variance is not None:
            print(f"    Variance: {config.pca_variance:.2%}")
        print(f"    Whiten: {config.pca_whiten}")
    else:
        print(f"  PCA: Disabled")
    print()

    # Load activations
    print("Loading activations...")
    if args.per_layer:
        # For per-layer mode, the path can be either:
        # 1. A directory containing *_layer_*.pkl files
        # 2. A file path - will extract directory and use glob pattern
        positive_acts = load_per_layer_activations(args.positive_acts)
        negative_acts = load_per_layer_activations(args.negative_acts)
    else:
        positive_acts = load_activations(args.positive_acts)
        negative_acts = load_activations(args.negative_acts)

    n_layers = len(positive_acts)
    print(f"Loaded activations for {n_layers} layers")
    print()

    # Parse layers argument
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    # Train and evaluate probes
    print("Training probes...")
    probes, train_results, test_results, layer_list, pca_models = train_and_evaluate_all_layers(
        positive_acts, negative_acts, config, layers
    )
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save probes
    save_probes(probes, args.output_dir, config)

    # Save results
    results = {
        "train": {str(k): v for k, v in train_results.items()},
        "test": {str(k): v for k, v in test_results.items()},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate plots
    print("\nGenerating plots...")

    # Combined plot
    plot_accuracy_over_layers(
        train_results, test_results, layer_list,
        output_dir / "accuracy_over_layers.png",
        title=args.plot_title
    )

    # Separate train plot
    plot_train_accuracy(
        train_results, layer_list,
        output_dir / "train_accuracy_over_layers.png",
        title=f"{args.plot_title} (Training)"
    )

    # Separate test plot
    plot_test_accuracy(
        test_results, layer_list,
        output_dir / "test_accuracy_over_layers.png",
        title=f"{args.plot_title} (Test)"
    )

    print("\nDone!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
