"""
Train linear probes on model activations from all layers.

Hyperparameters from: "Detecting Strategic Deception Using Linear Probes" (arXiv:2502.03407)
https://github.com/ApolloResearch/deception-detection

Probe types supported:
- Logistic Regression (linear probe)
- MLP (2-layer neural network probe)
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ProbeConfig:
    """Configuration for probe training.

    Hyperparameters from arXiv:2502.03407:
    - LogisticRegression: reg_coeff=1e3, normalize=True, fit_intercept=False
    - MLP: hidden_dim=64, lr=0.0001, epochs=10000, early_stopping=True
    """
    probe_type: Literal["lr", "mlp"] = "lr"

    # Logistic Regression hyperparameters
    reg_coeff: float = 1e3  # Regularization coefficient (C = 1/reg_coeff)
    normalize: bool = True  # Normalize activations before training

    # MLP hyperparameters
    hidden_dim: int = 64
    lr: float = 0.0001
    epochs: int = 10000
    val_split: float = 0.2  # 80% train, 20% validation
    early_stopping: bool = True

    # General settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 42
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)


class MLPProbe(nn.Module):
    """MLP probe with architecture: Linear -> ReLU -> Linear -> Sigmoid.

    From arXiv:2502.03407 detectors.py
    """

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
    """Logistic Regression probe wrapper.

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
        # Normalize activations
        if self.config.normalize:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0) + 1e-8
            X = (X - self.mean) / self.std

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=self.config.dtype)
        y_tensor = torch.tensor(y, dtype=self.config.dtype).unsqueeze(1)

        # Train/val split
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

        # Initialize model
        input_dim = X.shape[1]
        self.model = MLPProbe(input_dim, self.config.hidden_dim).to(device)

        # Optimizer and loss
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

            # Validation step
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


def load_activations(path: str) -> dict:
    """Load activations from a file.

    Expects either:
    - .npz file with 'activations' key (shape: [n_samples, n_layers, hidden_dim])
    - .pt file with tensor of same shape

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
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Assume shape is [n_samples, n_layers, hidden_dim]
    n_samples, n_layers, hidden_dim = activations.shape

    return {
        layer_idx: activations[:, layer_idx, :]
        for layer_idx in range(n_layers)
    }


def train_probes_all_layers(
    positive_acts: dict[int, np.ndarray],
    negative_acts: dict[int, np.ndarray],
    config: ProbeConfig,
    layers: Optional[list[int]] = None,
) -> dict[int, LinearProbe | MLPProbeTrainer]:
    """Train probes on all layers.

    Args:
        positive_acts: Dict mapping layer index to positive class activations
        negative_acts: Dict mapping layer index to negative class activations
        config: Probe configuration
        layers: List of layer indices to train on (default: all layers)

    Returns:
        Dict mapping layer index to trained probe
    """
    if layers is None:
        layers = list(positive_acts.keys())

    probes = {}

    for layer_idx in layers:
        print(f"Training probe for layer {layer_idx}...")

        # Combine positive and negative activations
        pos_acts = positive_acts[layer_idx]
        neg_acts = negative_acts[layer_idx]

        X = np.concatenate([pos_acts, neg_acts], axis=0)
        y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

        # Shuffle
        rng = np.random.RandomState(config.random_state)
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

        # Train probe
        if config.probe_type == "lr":
            probe = LinearProbe(config)
        else:
            probe = MLPProbeTrainer(config)

        probe.fit(X, y)
        probes[layer_idx] = probe

    return probes


def evaluate_probes(
    probes: dict[int, LinearProbe | MLPProbeTrainer],
    positive_acts: dict[int, np.ndarray],
    negative_acts: dict[int, np.ndarray],
) -> dict[int, dict]:
    """Evaluate probes on test data.

    Returns dict mapping layer index to metrics (accuracy, auroc).
    """
    results = {}

    for layer_idx, probe in probes.items():
        pos_acts = positive_acts[layer_idx]
        neg_acts = negative_acts[layer_idx]

        X = np.concatenate([pos_acts, neg_acts], axis=0)
        y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

        y_pred = probe.predict(X)
        y_proba = probe.predict_proba(X)

        accuracy = accuracy_score(y, y_pred)
        auroc = roc_auc_score(y, y_proba)

        results[layer_idx] = {
            "accuracy": accuracy,
            "auroc": auroc,
        }

        print(f"Layer {layer_idx}: Accuracy={accuracy:.4f}, AUROC={auroc:.4f}")

    return results


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
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved probes to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train probes on model activations from all layers"
    )
    parser.add_argument(
        "--positive_acts",
        type=str,
        required=True,
        help="Path to positive class activations (.npz, .pt, or .npy)",
    )
    parser.add_argument(
        "--negative_acts",
        type=str,
        required=True,
        help="Path to negative class activations (.npz, .pt, or .npy)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./probes",
        help="Directory to save trained probes",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        choices=["lr", "mlp"],
        default="lr",
        help="Type of probe to train (lr=logistic regression, mlp=MLP)",
    )
    parser.add_argument(
        "--reg_coeff",
        type=float,
        default=1e3,
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
        default=0.0001,
        help="Learning rate for MLP probe (default: 0.0001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Max epochs for MLP probe (default: 10000)",
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
    )

    print(f"Configuration:")
    print(f"  Probe type: {config.probe_type}")
    print(f"  Normalize: {config.normalize}")
    if config.probe_type == "lr":
        print(f"  Regularization coefficient: {config.reg_coeff}")
    else:
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Learning rate: {config.lr}")
        print(f"  Max epochs: {config.epochs}")
        print(f"  Early stopping: {config.early_stopping}")
    print(f"  Device: {config.device}")
    print()

    # Load activations
    print("Loading activations...")
    positive_acts = load_activations(args.positive_acts)
    negative_acts = load_activations(args.negative_acts)

    n_layers = len(positive_acts)
    print(f"Loaded activations for {n_layers} layers")
    print()

    # Parse layers argument
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    # Train probes
    print("Training probes...")
    probes = train_probes_all_layers(positive_acts, negative_acts, config, layers)
    print()

    # Evaluate
    print("Evaluating probes...")
    results = evaluate_probes(probes, positive_acts, negative_acts)
    print()

    # Save
    save_probes(probes, args.output_dir, config)

    # Save results
    results_serializable = {str(k): v for k, v in results.items()}
    with open(Path(args.output_dir) / "results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
