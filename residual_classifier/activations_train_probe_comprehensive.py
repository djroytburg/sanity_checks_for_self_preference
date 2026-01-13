# activations_train_probe_comprehensive.py: Comprehensive hyperparameter sweep for logistic regression probes
# Includes CV, penalties, solvers, class weights, and more to avoid overfitting and improve performance.
# Written by: Dani
# Created: Jan 12, 2026, 00:00 EST
# Last Modified: Jan 12, 2026, 00:00 EST

"""Comprehensive probe training with extensive hyperparameter sweeps to combat overfitting."""

import argparse
import getpass
import json
import logging
import os
import pickle
import random
import socket
import sys
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns



# ----------------------
# --- LOGGING SETUP ---
# ----------------------


def setup_logging(output_dir: Path, run_name: str) -> logging.Logger:
    """Set up comprehensive logging with metadata.

    Args:
        output_dir (Path): Root output directory for this run.
        run_name (str): Short name used for the log file prefix.

    Returns:
        logging.Logger: Configured logger writing to file and console.
    """
    log_dir = output_dir / "file_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name}_{timestamp}.log"

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE PROBE SWEEP RUN STARTED")
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
# --- PROBE DEFINITIONS ---
# -------------------------


@dataclass
class ProbeConfig:
    """Configuration for probe training.
    """
    probe_type: Literal["lr", "mlp"] = "lr"

    # Logistic Regression hyperparameters
    reg_coeff: float = 1e3  # (C = 1/reg_coeff)
    normalize: bool = True  # Normalize activations before training
    track_loss: bool = False  # Track loss curve during training (uses torch LR)
    penalty: Literal['l1', 'l2', 'elasticnet'] = 'l2'
    solver: str = 'auto'
    max_iter: int = 1000
    class_weight: Optional[str] = None
    l1_ratio: float = 0.0

    # MLP hyperparameters
    hidden_dim: int = 64
    lr: float = 0.0001
    epochs: int = 20000
    val_split: float = 0.2  # 80% train, 20% validation
    early_stopping: bool = True

    # General settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 42
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)


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
        self.track_loss = config.track_loss
        self.mean = None
        self.std = None
        self.loss_curve = [] if self.track_loss else None
        if not self.track_loss:
            solver = config.solver
            if solver == 'auto':
                solver = 'liblinear' if config.penalty == 'l1' else 'lbfgs'
            
            penalty_kwargs = {}
            if config.penalty == 'l1':
                penalty_kwargs = {'penalty': 'elasticnet', 'l1_ratio': 1.0}
                if solver not in ['saga']:
                    solver = 'saga'
            elif config.penalty == 'l2':
                # Use default penalty='l2' by not setting penalty
                pass
            else:
                penalty_kwargs = {'penalty': config.penalty, 'l1_ratio': config.l1_ratio}
            
            self.model = LogisticRegression(
                C=1 / config.reg_coeff,
                **penalty_kwargs,
                solver=solver,
                max_iter=config.max_iter,
                class_weight=config.class_weight,
                random_state=config.random_state,
                fit_intercept=False
            )
            self.torch_model = None
        else:
            self.model = None
            self.torch_model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        """Fit the logistic regression probe."""
        if not self.track_loss:
            if self.config.normalize:
                self.mean = X.mean(axis=0)
                self.std = X.std(axis=0) + 1e-8
                X = (X - self.mean) / self.std

            self.model.fit(X, y)
        else:
            if self.config.normalize:
                self.mean = np.array(X.mean(axis=0))
                self.std = np.array(X.std(axis=0) + 1e-8)
                X_norm = (X - self.mean) / self.std
            else:
                X_norm = X
                self.mean = None
                self.std = None

            X_tensor = torch.tensor(X_norm, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            input_dim = X.shape[1]
            self.torch_model = nn.Linear(input_dim, 1)

            optimizer = torch.optim.LBFGS(self.torch_model.parameters(), lr=1.0, max_iter=50)
            criterion = nn.BCEWithLogitsLoss()

            self.loss_curve = []

            def closure():
                optimizer.zero_grad()
                y_pred = self.torch_model(X_tensor).squeeze()
                loss = criterion(y_pred, y_tensor)
                loss.backward()
                self.loss_curve.append(loss.item())
                return loss

            optimizer.step(closure)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.torch_model is not None:
            if self.config.normalize and self.mean is not None:
                X = (X - self.mean) / self.std
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                logits = self.torch_model(X_tensor).squeeze()
                return (torch.sigmoid(logits) > 0.5).long().numpy()
        else:
            if self.config.normalize and self.mean is not None:
                X = (X - self.mean) / self.std
            return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.torch_model is not None:
            if self.config.normalize and self.mean is not None:
                X = (X - self.mean) / self.std
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                logits = self.torch_model(X_tensor).squeeze()
                return torch.sigmoid(logits).numpy()
        else:
            if self.config.normalize and self.mean is not None:
                X = (X - self.mean) / self.std
            return self.model.predict_proba(X)[:, 1]

    def get_direction(self) -> np.ndarray:
        """Get the probe direction (coefficients)."""
        if self.torch_model is not None:
            return self.torch_model.weight.data.numpy().squeeze()
        else:
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




# ---------------------------
# --- ACTIVATION LOADING  ---
# ---------------------------


def load_activations(path: str) -> dict[int, np.ndarray]:
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
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Assume shape is [n_samples, n_layers, hidden_dim]
    n_samples, n_layers, hidden_dim = activations.shape

    return {
        layer_idx: activations[:, layer_idx, :]
        for layer_idx in range(n_layers)
    }

@dataclass
class DataConfig:
    label_type: Literal['lsp', 'self'] = 'lsp'  # 'lsp' or 'self'
    train_split: float = 0.7  # Fraction of data to use for training
    balance_feature: str = 'combined'  # Feature to balance on: 'combined', 'order', 'reference', etc.
    train_balance: Optional[dict] = None  # Ratios for balancing train data, e.g., {'forward': 0.5, 'backward': 0.5}
    test_balance: Optional[dict] = None  # Ratios for balancing test data
    random_state: int = 42
    split_groupby: Literal["pair"] = "pair"  # Prevent leakage by splitting at cache-pair granularity


@dataclass
class PCAConfig:
    enabled: bool = False
    n_components: Optional[int] = None
    variance: Optional[float] = None
    whiten: bool = False

def _parse_cache_key(key: Any, sample: dict) -> tuple[Any, str, str]:
    """Normalize cache key variants into (id, judge, reference).

    Supports cache dictionaries keyed as either:
    - (id, reference)
    - (id, judge, reference)

    Args:
        key (Any): Raw key from cache['data'].
        sample (dict): Sample value from cache['data'][key].

    Returns:
        tuple[Any, str, str]: (id, judge, reference)

    Raises:
        ValueError: If key shape is unsupported and sample lacks needed fields.
    """
    if isinstance(key, tuple) and len(key) == 2:
        idx, reference = key
        judge = sample.get("judge")
        if judge is None:
            raise ValueError("Cache key is (id, reference) but sample missing 'judge'")
        return idx, str(judge), str(reference)

    if isinstance(key, tuple) and len(key) == 3:
        idx, judge, reference = key
        return idx, str(judge), str(reference)

    judge = sample.get("judge")
    reference = sample.get("reference")
    idx = sample.get("id")
    if idx is None or judge is None or reference is None:
        raise ValueError(f"Unsupported cache key format: {type(key)} {key}")
    return idx, str(judge), str(reference)


def load_cache_pickle(path: Path) -> tuple[dict, dict]:
    """Load a cache pickle produced by cache_activations.py.

    Args:
        path (Path): Path to cache pickle.

    Returns:
        tuple[dict, dict]: (metadata, data)

    Raises:
        ValueError: If pickle schema is not recognized.
    """
    with open(path, "rb") as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict) or "metadata" not in cache or "data" not in cache:
        raise ValueError("Cache pickle must be a dict with keys 'metadata' and 'data'.")
    if not isinstance(cache["data"], dict):
        raise ValueError("Cache['data'] must be a dict keyed by cache identifiers.")
    return cache["metadata"], cache["data"]


def load_from_cache(path: Path, data_config: Optional[DataConfig] = None) -> dict:
    """Load activations from a cache pickle into leakage-safe train/test splits.

    Split is performed at the cache-pair granularity (id/judge/reference). Forward/backward
    variants for a given pair are kept in the same split to avoid leakage.

    Args:
        path (Path): Path to cache pickle.
        data_config (Optional[DataConfig]): Data configuration.

    Returns:
        dict: {'metadata': ..., 'train': [...], 'test': [...]} sample entries.
    """
    if data_config is None:
        data_config = DataConfig()

    metadata, data = load_cache_pickle(path)

    keys = list(data.keys())
    rng = np.random.RandomState(data_config.random_state)
    rng.shuffle(keys)
    num_train_pairs = int(len(keys) * data_config.train_split)
    train_keys = set(keys[:num_train_pairs])
    test_keys = set(keys[num_train_pairs:])

    def expand_pair(pair_key: Any, sample: dict) -> list[dict]:
        idx, judge, reference = _parse_cache_key(pair_key, sample)
        if data_config.label_type == "lsp":
            label = 1 if sample.get("gold_label") == "lsp" else 0
        elif data_config.label_type == "ilsp":
            label = 1 if sample.get("gold_label") == "ilsp" else 0
        else:
            label = 1 if sample.get("self_label") == "self" else 0

        expanded = []
        for act_key, order in [("forward_activations", "forward"), ("backward_activations", "backward")]:
            if act_key not in sample:
                continue
            expanded.append(
                {
                    "group_key": (idx, reference),
                    "id": idx,
                    "judge": judge,
                    "reference": reference,
                    "lsp": sample.get("gold_label"),
                    "self": sample.get("self_label"),
                    "label": int(label),
                    "order": order,
                    "layer_indices": sample.get("layer_indices"),
                    "activations": sample[act_key],
                }
            )
        return expanded

    train_samples: list[dict] = []
    test_samples: list[dict] = []

    for k, sample in data.items():
        if k in train_keys:
            train_samples.extend(expand_pair(k, sample))
        elif k in test_keys:
            test_samples.extend(expand_pair(k, sample))

    if data_config.train_balance is not None:
        train_samples = balance_samples(train_samples, data_config.balance_feature, data_config.train_balance)
    if data_config.test_balance is not None:
        test_samples = balance_samples(test_samples, data_config.balance_feature, data_config.test_balance)

    return {"metadata": metadata, "train": train_samples, "test": test_samples}


def balance_samples(samples: list, balance_feature: str, balance_ratios: dict) -> list:
    """Balance samples based on a feature and ratios."""
    # Determine key for each sample
    def get_key(sample):
        if balance_feature == 'combined':
            return f"{'lsp' if sample['lsp'] == 'lsp' else 'ilsp'}/{'self' if sample['self'] == 'self' else 'other'}"
        elif balance_feature in sample:
            return sample[balance_feature]
        else:
            return 'default'
    
    # Count totals
    total_counts = {}
    for sample in samples:
        key = get_key(sample)
        total_counts[key] = total_counts.get(key, 0) + 1
    
    # Calculate targets
    total_samples = len(samples)
    targets = {k: int(total_samples * balance_ratios.get(k, 0)) for k in total_counts}
    
    # Collect balanced
    balanced = []
    counts = {k: 0 for k in total_counts}
    for sample in samples:
        key = get_key(sample)
        if counts[key] < targets.get(key, 0):
            balanced.append(sample)
            counts[key] += 1
    
    return balanced


def prepare_probe_data(samples: list) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Prepare positive and negative activations for probe training from samples."""
    positive_samples = [s for s in samples if s['label'] == 1]
    negative_samples = [s for s in samples if s['label'] == 0]
    
    # Assume all samples have the same layer structure
    n_layers = positive_samples[0]['activations'].shape[0]
    
    positive_acts = {layer: np.array([s['activations'][layer] for s in positive_samples]) for layer in range(n_layers)}
    negative_acts = {layer: np.array([s['activations'][layer] for s in negative_samples]) for layer in range(n_layers)}
    
    return positive_acts, negative_acts



# ---------------------------
# --- METRICS / TRAINING  ---
# ---------------------------


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy plus binary-positive and macro-averaged PRF metrics.

    Args:
        y_true (np.ndarray): Ground truth labels (0/1).
        y_pred (np.ndarray): Predicted labels (0/1).

    Returns:
        dict: Metric dict.
    """
    accuracy = accuracy_score(y_true, y_pred)

    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision_pos": float(precision_pos),
        "recall_pos": float(recall_pos),
        "f1_pos": float(f1_pos),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
    }


def _extract_xy_for_layer(samples: list[dict], layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([s["activations"][layer_idx] for s in samples], axis=0)
    y = np.array([s["label"] for s in samples], dtype=np.int64)
    return X, y


def _fit_pca_if_needed(X_train: np.ndarray, pca_config: PCAConfig) -> Optional[PCA]:
    if not pca_config.enabled:
        return None
    if pca_config.n_components is None and pca_config.variance is None:
        return None

    n_components: Any
    if pca_config.n_components is not None:
        n_components = int(pca_config.n_components)
    else:
        n_components = float(pca_config.variance)

    pca = PCA(n_components=n_components, whiten=pca_config.whiten, random_state=0)
    pca.fit(X_train)
    return pca


def train_eval_probe_all_layers_from_cache(
    train_samples: list[dict],
    test_samples: list[dict],
    probe_type: str,
    hyper_dict: dict,
    normalize: bool,
    pca_config: PCAConfig,
    layers: Optional[list[int]],
    track_loss: bool,
    cv_folds: int,
    logger: logging.Logger,
    lr: float = 0.0001,
    epochs: int = 20000,
    early_stopping: bool = True,
) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict], float]:
    """Train/evaluate probes for all layers for a single hyperparam dict.

    For LR: hyper_dict has 'reg_coeff', 'penalty', 'solver'
    For MLP: hyper_dict has 'hidden_dim'
    Returns:
        tuple: (train_metrics_by_layer, test_metrics_by_layer, artifacts_by_layer)
    """
    if not train_samples or not test_samples:
        raise ValueError("Train/test samples are empty; cannot train probes.")

    # Determine number of layers from first sample activations.
    n_layers = int(train_samples[0]["activations"].shape[0])
    if layers is None:
        layers = list(range(n_layers))

    train_metrics: dict[int, dict] = {}
    test_metrics: dict[int, dict] = {}
    artifacts: dict[int, dict] = {}

    if probe_type == "lr":
        cfg = ProbeConfig(probe_type="lr", reg_coeff=hyper_dict['reg_coeff'], penalty=hyper_dict['penalty'], solver=hyper_dict['solver'], class_weight=hyper_dict.get('class_weight'), l1_ratio=hyper_dict.get('l1_ratio', 0.0), normalize=normalize, track_loss=track_loss)
    else:
        cfg = ProbeConfig(probe_type="mlp", hidden_dim=hyper_dict['hidden_dim'], lr=lr, epochs=epochs, early_stopping=early_stopping, normalize=normalize)

    for layer_idx in layers:
        X_train, y_train = _extract_xy_for_layer(train_samples, layer_idx)
        X_test, y_test = _extract_xy_for_layer(test_samples, layer_idx)

        # If a split has only one class, sklearn LR will error. Fall back to constant prediction.
        if len(np.unique(y_train)) < 2:
            constant = int(y_train[0])
            y_pred_train = np.full_like(y_train, fill_value=constant)
            y_pred_test = np.full_like(y_test, fill_value=constant)

            train_metrics[layer_idx] = compute_binary_metrics(y_train, y_pred_train)
            test_metrics[layer_idx] = compute_binary_metrics(y_test, y_pred_test)
            artifacts[layer_idx] = {
                "hyperparam": hyper_dict,
                "normalize": bool(normalize),
                "coef": None,
                "mean": None,
                "std": None,
                "pca": None,
                "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
                "loss_curve": None,
                "note": "single_class_train_split_constant_predictor",
            }
            logger.warning(
                f"Layer {layer_idx}: train split has a single class; using constant predictor={constant}"
            )
            continue

        # PCA (fit on train only)
        pca = _fit_pca_if_needed(X_train, pca_config)
        if pca is not None:
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)
        else:
            X_train_pca = X_train
            X_test_pca = X_test

        if cv_folds > 0:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_state)
            cv_scores = []
            for train_idx, val_idx in skf.split(X_train_pca, y_train):
                X_tr, X_val = X_train_pca[train_idx], X_train_pca[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                probe_cv = LinearProbe(cfg)
                probe_cv.fit(X_tr, y_tr)
                y_pred_val = probe_cv.predict(X_val)
                val_metrics = compute_binary_metrics(y_val, y_pred_val)
                cv_scores.append(val_metrics['f1_macro'])
            cv_score = np.mean(cv_scores)
        else:
            cv_score = 0.0

        # Train on full train
        probe = LinearProbe(cfg)
        probe.fit(X_train_pca, y_train)

        # Eval
        y_pred_train = probe.predict(X_train_pca)
        y_pred_test = probe.predict(X_test_pca)

        train_metrics[layer_idx] = compute_binary_metrics(y_train, y_pred_train)
        test_metrics[layer_idx] = compute_binary_metrics(y_test, y_pred_test)

        artifacts[layer_idx] = {
            "hyperparam": hyper_dict,
            "normalize": bool(normalize),
            "coef": probe.get_direction().astype(np.float32) if probe.torch_model is None else probe.get_direction().astype(np.float32),
            "mean": None if probe.mean is None else probe.mean.astype(np.float32),
            "std": None if probe.std is None else probe.std.astype(np.float32),
            "pca": None
            if pca is None
            else {
                "components": pca.components_.astype(np.float32),
                "mean": pca.mean_.astype(np.float32),
                "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
                "n_components": int(pca.n_components_) if hasattr(pca, "n_components_") else None,
            },
            "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
            "loss_curve": probe.loss_curve,
        }

        if True:  # Log for all layers since sweep is small
            logger.info(
                f"{probe_type.upper()} hyper={hyper_dict} layer={layer_idx} "
                f"cv_f1={cv_score:.3f} test acc={test_metrics[layer_idx]['accuracy']:.3f} "
                f"test f1_macro={test_metrics[layer_idx]['f1_macro']:.3f}"
            )

    return train_metrics, test_metrics, artifacts, cv_score



# ----------------------
# --- PLOTTING UTILS ---
# ----------------------


def plot_confusion_matrix(cm: list[list[int]], output_path: Path, title: str) -> None:
    """Plot confusion matrix."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Ubuntu'],
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })
    im = ax.imshow(cm, interpolation='nearest', cmap=sns.color_palette("Blues", as_cmap=True))
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontweight='bold', fontfamily='Volkhov')
    ax.set_xlabel('Predicted Label', fontfamily='Ubuntu Mono')
    ax.set_ylabel('True Label', fontfamily='Ubuntu Mono')
    tick_marks = np.arange(len(['ILSP', 'LSP']))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['ILSP', 'LSP'], fontfamily='Ubuntu Mono')
    ax.set_yticklabels(['ILSP', 'LSP'], fontfamily='Ubuntu Mono')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold', fontfamily='Ubuntu Mono')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_all_layerwise_metrics(results_by_hyper: dict[str, dict], output_path: Path, title: str) -> None:
    """Plot accuracy/precision/recall/F1 across layers for all hyperparams."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Ubuntu'],
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.titleweight': 'bold',
    })

    layers = sorted(list(results_by_hyper.values())[0]['test'].keys())
    metrics = ["accuracy", "precision", "recall", "f1"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    colors = sns.color_palette("husl", len(results_by_hyper))

    for ax, metric in zip(axes, metrics):
        for i, (hyper, data) in enumerate(results_by_hyper.items()):
            test_vals = [data['test'][l][f"{metric}_macro"] for l in layers]
            ax.plot(layers, test_vals, label=f"reg={hyper}", color=colors[i], linewidth=2)
        ax.set_title(metric.capitalize(), fontweight='bold', fontfamily='Volkhov')
        ax.set_xlabel("Layer", fontfamily='Ubuntu Mono')
        ax.set_ylabel(metric.capitalize(), fontfamily='Ubuntu Mono')
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.grid(True, alpha=1.0, color='black', linewidth=0.8)

    # Find best performer
    best_hyper = max(results_by_hyper, key=lambda h: max(results_by_hyper[h]['test'][l]['f1_macro'] for l in layers))
    best_layer = max(layers, key=lambda l: results_by_hyper[best_hyper]['test'][l]['f1_macro'])
    best_val = results_by_hyper[best_hyper]['test'][best_layer]['f1_macro']

    ax = axes[3]  # f1 plot
    ax.annotate(f'Best: reg={best_hyper}, layer={best_layer}, f1={best_val:.3f}', 
                xy=(best_layer, best_val), xytext=(best_layer+1, best_val+0.05), 
                arrowprops=dict(arrowstyle='->', color='red'), 
                fontsize=10, fontfamily='Ubuntu Mono')

    fig.suptitle(title, fontsize=18, fontweight='bold', fontfamily='Volkhov')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)



# ----------------------
# --- MAIN EXECUTION ---
# ----------------------


def _parse_float_list(csv: str) -> list[float]:
    vals: list[float] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


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

        
        pos_acts = positive_acts[layer_idx]
        neg_acts = negative_acts[layer_idx]

        X = np.concatenate([pos_acts, neg_acts], axis=0)
        y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

        # Shuffle
        rng = np.random.RandomState(config.random_state)
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

        
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
        "--cache_pkl",
        type=str,
        default=None,
        help="Path to activation cache pickle from cache_activations.py (enables cache mode)",
    )
    parser.add_argument(
        "--positive_acts",
        type=str,
        required=False,
        default=None,
        help="Path to positive class activations (.npz, .pt, .npy, or .pkl) [legacy mode]",
    )
    parser.add_argument(
        "--negative_acts",
        type=str,
        required=False,
        default=None,
        help="Path to negative class activations (.npz, .pt, .npy, or .pkl) [legacy mode]",
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
        default=256,
        help="Hidden dimension for MLP probe (default: 256)",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="128,256,512",
        help="Comma-separated hidden_dims sweep for MLP (default: 128,256,512)",
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
    parser.add_argument(
        "--label_type",
        type=str,
        default="lsp",
        choices=["lsp", "self", "ilsp"],
        help="Label type for cache mode (default: lsp)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Train split fraction for cache mode (default: 0.8)",
    )
    parser.add_argument(
        "--reg_coeffs",
        type=str,
        default="10,100,1000",
        help="Comma-separated reg_coeff sweep for LR (default: 1..10000)",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=None,
        help="Optional PCA components (fit on train per-layer) before probing",
    )
    parser.add_argument(
        "--pca_variance",
        type=float,
        default=None,
        help="Optional PCA variance (0-1], alternative to --pca_components",
    )
    parser.add_argument(
        "--pca_whiten",
        action="store_true",
        help="Whiten PCA components (only if PCA enabled)",
    )
    parser.add_argument(
        "--track_loss",
        action="store_true",
        help="Track loss curve during training (uses torch LR)",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of CV folds for hyperparameter selection (0 to disable)",
    )
    parser.add_argument(
        "--penalties",
        type=str,
        default="l1,l2",
        help="Comma-separated penalties for LR",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default="liblinear,lbfgs",
        help="Comma-separated solvers for LR",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default=None,
        help="Class weight for LR (e.g., 'balanced')",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, run_name="activations_train_probe")

    # Create config (used for non-cache legacy mode)
    config = ProbeConfig(
        probe_type=args.probe_type,
        reg_coeff=args.reg_coeff,
        normalize=not args.no_normalize,
        track_loss=args.track_loss,
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

    # Cache mode: load cache pickle and run unbalanced full-set train/test + reg sweep
    if args.cache_pkl is not None:
        cache_path = Path(args.cache_pkl)
        data_cfg = DataConfig(
            label_type=args.label_type,
            train_split=float(args.train_split),
            train_balance=None,
            test_balance=None,
            random_state=int(args.seed),
        )
        pca_cfg = PCAConfig(
            enabled=(args.pca_components is not None or args.pca_variance is not None),
            n_components=args.pca_components,
            variance=args.pca_variance,
            whiten=bool(args.pca_whiten),
        )

        logger.info("Loading cache pickle (may be large)...")
        loaded = load_from_cache(cache_path, data_cfg)
        cache_meta = loaded.get("metadata", {})
        train_samples = loaded["train"]
        test_samples = loaded["test"]

        logger.info(f"Cache metadata: {cache_meta}")
        logger.info(f"Train samples: {len(train_samples)}")
        logger.info(f"Test samples: {len(test_samples)}")

        n_layers = train_samples[0]["activations"].shape[0]
        layers = None
        if args.layers:
            original_layers = [int(x.strip()) for x in args.layers.split(",")]
            layers = [l for l in original_layers if 0 <= l < n_layers]
            if len(layers) != len(original_layers):
                logger.warning(f"Some layers were out of bounds (0-{n_layers-1}), filtered to {layers}")
        else:
            layers = list(range(n_layers))  # All layers
            logger.info(f"Using all {len(layers)} layers")

        reg_coeffs = _parse_float_list(args.reg_coeffs)
        penalties = [p.strip() for p in args.penalties.split(',')]
        solvers = [s.strip() for s in args.solvers.split(',')]
        if args.probe_type == "lr":
            hyperparams = []
            for reg in reg_coeffs:
                for pen in penalties:
                    if pen == 'l1':
                        penalty = 'elasticnet'
                        l1_ratio = 1.0
                        allowed_solvers = ['saga']
                    elif pen == 'l2':
                        penalty = 'l2'
                        l1_ratio = 0.0
                        allowed_solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
                    else:
                        penalty = pen
                        l1_ratio = 0.0
                        allowed_solvers = solvers  # fallback
                    for sol in solvers:
                        if sol not in allowed_solvers:
                            continue
                        hyperparams.append({'reg_coeff': reg, 'penalty': penalty, 'l1_ratio': l1_ratio, 'solver': sol, 'class_weight': args.class_weight if hasattr(args, 'class_weight') else None})
            hyper_name = "lr_hypers"
        else:
            hyperparams = [{'hidden_dim': h} for h in _parse_float_list(args.hidden_dims)]
            hyper_name = "hidden_dims"
        logger.info(f"{args.probe_type.upper()} {hyper_name} sweep: {len(hyperparams)} combinations")
        logger.info(f"CV folds: {args.cv_folds}")
        logger.info(f"PCA enabled: {pca_cfg.enabled} (components={pca_cfg.n_components}, var={pca_cfg.variance})")

        results_by_hyper: dict[str, dict] = {}
        best_by_layer: dict[int, dict] = {}

        for hyper in hyperparams:
            logger.info(f"Training/evaluating hyper={hyper}")
            train_m, test_m, artifacts, cv_score = train_eval_probe_all_layers_from_cache(
                train_samples=train_samples,
                test_samples=test_samples,
                probe_type=args.probe_type,
                hyper_dict=hyper,
                normalize=not args.no_normalize,
                pca_config=pca_cfg,
                layers=layers,
                track_loss=args.track_loss,
                cv_folds=args.cv_folds,
                logger=logger,
            )
            results_by_hyper[str(hyper)] = {"train": train_m, "test": test_m}

            # Update best-by-layer selection (maximize cv_score if cv, else test f1_macro)
            score_key = 'cv_score' if args.cv_folds > 0 else 'test_f1'
            for layer_idx, test_metrics in test_m.items():
                candidate = {
                    "hyper": hyper,
                    "train": train_m[layer_idx],
                    "test": test_metrics,
                    "artifact": artifacts[layer_idx],
                    "cv_score": cv_score if args.cv_folds > 0 else test_metrics['f1_macro'],
                }
                if layer_idx not in best_by_layer:
                    best_by_layer[layer_idx] = candidate
                else:
                    if candidate["cv_score"] > best_by_layer[layer_idx]["cv_score"]:
                        best_by_layer[layer_idx] = candidate

        # Save best models per layer
        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx, best in best_by_layer.items():
            art = best["artifact"]
            npz_path = models_dir / f"best_lr_layer_{layer_idx}.npz"
            coef = np.array([]) if art.get("coef") is None else art["coef"]
            mean = np.array([]) if art.get("mean") is None else art["mean"]
            std = np.array([]) if art.get("std") is None else art["std"]
            np.savez(
                npz_path,
                hyperparam=art["hyperparam"],
                normalize=int(art["normalize"]),
                coef=coef,
                mean=mean,
                std=std,
                pca_components=np.array([]) if art["pca"] is None else art["pca"]["components"],
                pca_mean=np.array([]) if art["pca"] is None else art["pca"]["mean"],
                pca_explained_variance_ratio=np.array([])
                if art["pca"] is None
                else art["pca"]["explained_variance_ratio"],
            )

        # Save results JSON
        results_payload = {
            "run": {
                "cache_pkl": str(cache_path),
                "cache_metadata": cache_meta,
                "label_type": args.label_type,
                "train_split": float(args.train_split),
                "seed": int(args.seed),
                "layers": layers,
                "hyperparams": hyperparams,
                "normalize": bool(not args.no_normalize),
                "cv_folds": args.cv_folds,
                "penalties": penalties,
                "solvers": solvers,
                "class_weight": args.class_weight,
                "pca": {
                    "enabled": pca_cfg.enabled,
                    "n_components": pca_cfg.n_components,
                    "variance": pca_cfg.variance,
                    "whiten": pca_cfg.whiten,
                },
            },
            "sweep": results_by_hyper,
            "best_by_layer": {
                str(k): {"hyper": v["hyper"], "train": v["train"], "test": v["test"], "cv_score": v["cv_score"]}
                for k, v in best_by_layer.items()
            },
        }
        with open(output_dir / "results_cache_sweep.json", "w") as f:
            json.dump(results_payload, f, indent=2)

        # Plots
        plots_dir = output_dir / "plots"
        best_metrics_for_plot = {k: {"train": v["train"], "test": v["test"]} for k, v in best_by_layer.items()}
        plot_all_layerwise_metrics(
            best_metrics_for_plot,
            plots_dir / "metrics_by_layer_pos.png",
            title="Layerwise metrics (positive-class PRF) - best reg per layer",
            averaging="pos",
        )
        plot_all_layerwise_metrics(
            best_metrics_for_plot,
            plots_dir / "metrics_by_layer_macro.png",
            title="Layerwise metrics (macro PRF) - best reg per layer",
            averaging="macro",
        )

        # Confusion matrices
        cm_dir = output_dir / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)
        for layer, data in best_by_layer.items():
            cm = data["artifact"]["confusion_matrix"]
            plot_confusion_matrix(cm, cm_dir / f"layer_{layer}_confusion_matrix.png", f"Layer {layer} Confusion Matrix")

        # Loss curves
        if args.track_loss:
            loss_dir = output_dir / "loss_curves"
            loss_dir.mkdir(parents=True, exist_ok=True)
            for layer, data in best_by_layer.items():
                if "loss_curve" in data["artifact"] and data["artifact"]["loss_curve"]:
                    sns.set_style("whitegrid")
                    plt.rcParams.update({
                        'font.family': 'serif',
                        'font.size': 12,
                        'axes.labelsize': 14,
                        'axes.titlesize': 16,
                        'xtick.labelsize': 12,
                        'ytick.labelsize': 12,
                    })
                    fig, ax = plt.subplots(figsize=(8, 6))
                    loss_curve = data["artifact"]["loss_curve"]
                    ax.plot(loss_curve, color=sns.color_palette("husl", 1)[0], linewidth=2)
                    ax.set_title(f"Layer {layer} Loss Curve", fontweight='bold')
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Loss")
                    ax.grid(True, alpha=1.0, color='black')
                    fig.tight_layout()
                    fig.savefig(loss_dir / f"layer_{layer}_loss_curve.png", dpi=200, bbox_inches='tight')
                    plt.close(fig)

        logger.info(f"Artifacts saved to: {output_dir}")
        return

    # Legacy mode (requires explicit positive/negative act files)
    if args.positive_acts is None or args.negative_acts is None:
        raise ValueError("Legacy mode requires --positive_acts and --negative_acts, or use --cache_pkl")
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
