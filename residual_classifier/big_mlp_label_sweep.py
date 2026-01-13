# big_mlp_label_sweep.py: Run a fixed biggest/deepest MLP probe across labels/layers
# Generates consolidated metric plots per activation cache.
# Written by: Dani
# Created: Jan 13, 2026, 02:55 EST
# Last Modified: Jan 13, 2026, 02:55 EST

import argparse
import getpass
import json
import logging
import os
import pickle
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler



# ----------------------
# --- FILE UTILITIES ---
# ----------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure console + file logging for the sweep.

    Args:
        output_dir (Path): Directory where logs and artifacts will be written.

    Returns:
        logging.Logger: Configured logger.

    Raises:
        OSError: If log file cannot be created.
    """

    log_dir = output_dir / "file_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"big_mlp_label_sweep_{timestamp}.log"

    logger = logging.getLogger("big_mlp_label_sweep")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 80)
    logger.info("BIG MLP LABEL SWEEP - RUN STARTED")
    logger.info("=" * 80)
    logger.info(f"User: {getpass.getuser()}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    return logger


def safe_model_tag(path: Path) -> str:
    """Create a stable tag name from a pickle filename.

    Args:
        path (Path): Path to a pickle file.

    Returns:
        str: File stem suitable for naming artifacts.

    Raises:
        ValueError: If the path has no stem.
    """

    stem = path.stem
    if not stem:
        raise ValueError(f"Invalid path for tagging: {path}")
    return stem.replace(" ", "_")



# ----------------------
# --- CORE LOGIC ---
# ----------------------

@dataclass
class DataConfig:
    label_type: Literal["lsp", "ilsp", "self"]


class RobustMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, ...], dropout_rate: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _determine_label(sample: dict, label_type: str) -> int:
    """Convert a cached sample to a binary label.

    Args:
        sample (dict): Cache sample dict.
        label_type (str): One of {'lsp','ilsp','self'}.

    Returns:
        int: 1 if positive class else 0.

    Raises:
        ValueError: If label_type is unknown.
    """

    if label_type == "self":
        return 1 if sample.get("self_label") == "self" else 0
    if label_type == "ilsp":
        return 1 if sample.get("gold_label") == "ilsp" else 0
    if label_type == "lsp":
        return 1 if sample.get("gold_label") == "lsp" else 0
    raise ValueError(f"Unknown label_type: {label_type}")


def load_activation_cache(cache_pkl: Path, logger: logging.Logger) -> Tuple[dict, List[dict]]:
    """Load activation cache pickle and flatten into sample views.

    Args:
        cache_pkl (Path): Path to activation cache pickle.
        logger (logging.Logger): Logger.

    Returns:
        Tuple[dict, List[dict]]: (cache_metadata, samples) where samples contain fwd/bwd views.

    Raises:
        FileNotFoundError: If cache_pkl does not exist.
        KeyError: If required cache keys are missing.
    """

    if not cache_pkl.exists():
        raise FileNotFoundError(f"Missing cache pickle: {cache_pkl}")

    logger.info(f"Loading activation cache: {cache_pkl}")
    with open(cache_pkl, "rb") as f:
        cache = pickle.load(f)

    if "data" not in cache:
        raise KeyError(f"Cache missing 'data': {cache_pkl}")

    metadata = cache.get("metadata", {})
    data = cache["data"]

    samples: List[dict] = []
    for _, s in data.items():
        pid = f"{s.get('id')}_{s.get('reference')}"
        for view in ["forward_activations", "backward_activations"]:
            if view in s:
                samples.append(
                    {
                        "pid": pid,
                        "gold_label": s.get("gold_label"),
                        "self_label": s.get("self_label"),
                        "layer_indices": s.get("layer_indices"),
                        "act": s[view],
                    }
                )

    if not samples:
        raise ValueError(f"No activation views found in cache: {cache_pkl}")

    return metadata, samples


def build_layer_index_map(samples: List[dict]) -> Dict[int, int]:
    """Map true model layer indices to positions in the cached activation array.

    Args:
        samples (List[dict]): Loaded samples.

    Returns:
        Dict[int, int]: Map from layer index (e.g., 26) to array position.

    Raises:
        ValueError: If layer mapping cannot be inferred.
    """

    layer_indices = None
    for s in samples:
        if isinstance(s.get("layer_indices"), list) and len(s["layer_indices"]) > 0:
            layer_indices = s["layer_indices"]
            break

    if layer_indices is None:
        # Fall back: assume act has full layers from 0..N-1
        act0 = samples[0]["act"]
        if not hasattr(act0, "shape"):
            raise ValueError("Cannot infer layer mapping; act has no shape.")
        n_layers = int(act0.shape[0])
        return {i: i for i in range(n_layers)}

    return {int(layer): i for i, layer in enumerate(layer_indices)}


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Compute common binary classification metrics.

    Args:
        y_true (np.ndarray): Binary labels.
        probs (np.ndarray): Predicted probabilities for positive class.

    Returns:
        Dict[str, float]: Metric dict.

    Raises:
        ValueError: If shapes mismatch.
    """

    if y_true.shape[0] != probs.shape[0]:
        raise ValueError("y_true and probs length mismatch")

    preds = (probs > 0.5).astype(int)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5

    return {
        "acc": float(accuracy_score(y_true, preds)),
        "auc": float(auc),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
        "prec": float(precision_score(y_true, preds, zero_division=0)),
        "rec": float(recall_score(y_true, preds, zero_division=0)),
    }


def train_big_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_sizes: Tuple[int, ...],
    dropout: float,
    weight_decay: float,
    lr: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    logger: logging.Logger,
) -> nn.Module:
    """Train the fixed big MLP with early stopping on validation AUC.

    Args:
        X_train (np.ndarray): Train features.
        y_train (np.ndarray): Train labels.
        X_val (np.ndarray): Val features.
        y_val (np.ndarray): Val labels.
        hidden_sizes (Tuple[int, ...]): Hidden sizes for MLP.
        dropout (float): Dropout probability.
        weight_decay (float): AdamW weight decay.
        lr (float): Learning rate.
        device (torch.device): Torch device.
        logger (logging.Logger): Logger.

    Returns:
        nn.Module: Trained model.

    Raises:
        RuntimeError: If training fails.
    """

    input_size = int(X_train.shape[1])
    model = RobustMLP(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_auc = -1.0
    best_state = None
    patience_ctr = 0

    n_samples = X_train_t.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            bx = X_train_t[idx]
            by = y_train_t[idx]

            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).detach().cpu().numpy().flatten()
            try:
                val_auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.5
            except Exception:
                val_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is None:
        logger.warning("No best state captured; returning last epoch weights")
        return model

    model.load_state_dict(best_state)
    return model


def sweep_cache(
    cache_pkl: Path,
    layers: List[int] | None,
    label_types: List[str],
    cv_folds: int,
    hidden_sizes: Tuple[int, ...],
    dropout: float,
    weight_decay: float,
    lr: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Run the big-MLP probe sweep for one cache across labels/layers.

    Args:
        cache_pkl (Path): Cache pickle.
        layers (List[int]): True layer indices to evaluate.
        label_types (List[str]): Label types.
        cv_folds (int): Number of CV folds.
        hidden_sizes (Tuple[int, ...]): Hidden sizes.
        dropout (float): Dropout.
        weight_decay (float): Weight decay.
        lr (float): Learning rate.
        output_dir (Path): Output directory.
        logger (logging.Logger): Logger.

    Returns:
        dict: Nested results dict.

    Raises:
        ValueError: If requested layers not present.
    """

    metadata, samples = load_activation_cache(cache_pkl, logger)
    if layers is None:
        layers = list(range(0, metadata.get("n_layers", 26)))
    layer_map = build_layer_index_map(samples)

    missing_layers = [l for l in layers if l not in layer_map]
    if missing_layers:
        logger.warning(f"Requested layers missing from cache {cache_pkl}: {missing_layers}. These will be skipped.")
    # Only keep layers that exist in this cache
    available_layers = [l for l in layers if l in layer_map]
    if not available_layers:
        logger.warning(f"No requested layers available in cache {cache_pkl}; skipping this cache.")
        results["results"] = {}
        return results

    # Precompute X_all per layer position for speed (only for available layers)
    X_by_layer_pos: Dict[int, np.ndarray] = {}
    for l in available_layers:
        pos = layer_map[l]
        X_by_layer_pos[pos] = np.stack([s["act"][pos] for s in samples])

    pids = np.array([s["pid"] for s in samples])
    u_pids, u_idx = np.unique(pids, return_index=True)

    results: dict = {
        "cache_pkl": str(cache_pkl),
        "metadata": metadata,
        "hyperparameters": {
            "hidden_sizes": list(hidden_sizes),
            "dropout": dropout,
            "weight_decay": weight_decay,
            "lr": lr,
            "max_epochs": max_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "cv_folds": cv_folds,
        },
        "results": {},
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    for label_type in label_types:
        cfg = DataConfig(label_type=label_type)  # noqa: F841
        y_all = np.array([_determine_label(s, label_type) for s in samples])
        u_labels = y_all[u_idx]

        if len(np.unique(u_labels)) < 2:
            logger.warning(f"Skipping label_type={label_type}: only one class present")
            continue

        label_out: dict = {"layers": {}}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for layer in available_layers:
            pos = layer_map[layer]
            X_all = X_by_layer_pos[pos]

            fold_metrics_te: List[Dict[str, float]] = []
            fold_metrics_tr: List[Dict[str, float]] = []

            for fold, (tr_idx, te_idx) in enumerate(skf.split(u_pids, u_labels)):
                tr_set = set(u_pids[tr_idx])
                te_set = set(u_pids[te_idx])

                tr_mask = np.array([pid in tr_set for pid in pids])
                te_mask = np.array([pid in te_set for pid in pids])

                X_tr_raw, y_tr = X_all[tr_mask], y_all[tr_mask]
                X_te_raw, y_te = X_all[te_mask], y_all[te_mask]

                sc = StandardScaler().fit(X_tr_raw)
                X_tr = sc.transform(X_tr_raw)
                X_te = sc.transform(X_te_raw)

                # Inner split (no leakage by pid)
                tr_pids = pids[tr_mask]
                tr_unique, tr_u_idx = np.unique(tr_pids, return_index=True)
                tr_u_labels = y_tr[tr_u_idx]

                tr_u_train, tr_u_val = train_test_split(
                    tr_unique,
                    test_size=0.2,
                    random_state=42,
                    stratify=tr_u_labels,
                )
                in_train = np.isin(tr_pids, tr_u_train)
                in_val = np.isin(tr_pids, tr_u_val)

                model = train_big_mlp(
                    X_train=X_tr[in_train],
                    y_train=y_tr[in_train],
                    X_val=X_tr[in_val],
                    y_val=y_tr[in_val],
                    hidden_sizes=hidden_sizes,
                    dropout=dropout,
                    weight_decay=weight_decay,
                    lr=lr,
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=batch_size,
                    device=device,
                    logger=logger,
                )

                model.eval()
                with torch.no_grad():
                    tr_probs = (
                        torch.sigmoid(model(torch.tensor(X_tr, dtype=torch.float32).to(device)))
                        .detach()
                        .cpu()
                        .numpy()
                        .flatten()
                    )
                    te_probs = (
                        torch.sigmoid(model(torch.tensor(X_te, dtype=torch.float32).to(device)))
                        .detach()
                        .cpu()
                        .numpy()
                        .flatten()
                    )

                fold_metrics_tr.append(compute_metrics(y_tr, tr_probs))
                fold_metrics_te.append(compute_metrics(y_te, te_probs))

                logger.debug(
                    f"{safe_model_tag(cache_pkl)} | {label_type} | layer={layer} | fold={fold+1}/{cv_folds} "
                    f"te_acc={fold_metrics_te[-1]['acc']:.3f} te_auc={fold_metrics_te[-1]['auc']:.3f}"
                )

            # Aggregate folds
            def _agg(metric_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
                keys = metric_list[0].keys()
                out: Dict[str, Dict[str, float]] = {}
                for k in keys:
                    vals = np.array([m[k] for m in metric_list], dtype=float)
                    out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0)}
                return out

            label_out["layers"][str(layer)] = {
                "train": _agg(fold_metrics_tr),
                "test": _agg(fold_metrics_te),
                "n_folds": cv_folds,
            }

            logger.info(
                f"{safe_model_tag(cache_pkl)} | {label_type} | layer={layer}: "
                f"test acc={label_out['layers'][str(layer)]['test']['acc']['mean']:.3f}±{label_out['layers'][str(layer)]['test']['acc']['std']:.3f}, "
                f"auc={label_out['layers'][str(layer)]['test']['auc']['mean']:.3f}±{label_out['layers'][str(layer)]['test']['auc']['std']:.3f}"
            )

        results["results"][label_type] = label_out
    dataset = metadata.get("dataset", "unknown_dataset")
    out_json = output_dir / f"{safe_model_tag(cache_pkl)}_big_mlp_sweep_{dataset}_{label_type}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results: {out_json}")

    return results



def plot_results(results: dict, output_dir: Path, logger: logging.Logger) -> Path:
    """Create a consolidated multi-metric plot for one cache.

    Args:
        results (dict): Results dict from sweep_cache.
        output_dir (Path): Output directory.
        logger (logging.Logger): Logger.

    Returns:
        Path: Path to saved plot.

    Raises:
        KeyError: If results are missing expected keys.
    """

    cache_pkl = Path(results["cache_pkl"])
    tag = safe_model_tag(cache_pkl)

    metrics = ["acc", "auc", "f1_macro", "prec", "rec"]
    metric_titles = {
        "acc": "Accuracy",
        "auc": "ROC AUC",
        "f1_macro": "F1 (macro)",
        "prec": "Precision",
        "rec": "Recall",
    }

    label_types = [lt for lt in ["ilsp", "lsp", "self"] if lt in results["results"]]
    colors = {"ilsp": "tab:blue", "lsp": "tab:green", "self": "tab:orange"}

    # Infer layer ordering from first label type
    layers = sorted([int(k) for k in results["results"][label_types[0]]["layers"].keys()])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for lt in label_types:
            layer_dict = results["results"][lt]["layers"]

            te_means = np.array([layer_dict[str(l)]["test"][metric]["mean"] for l in layers], dtype=float)
            te_stds = np.array([layer_dict[str(l)]["test"][metric]["std"] for l in layers], dtype=float)

            tr_means = np.array([layer_dict[str(l)]["train"][metric]["mean"] for l in layers], dtype=float)

            ax.plot(layers, te_means, color=colors.get(lt, "black"), label=f"{lt} (test)")
            ax.fill_between(
                layers,
                np.clip(te_means - te_stds, 0.0, 1.0),
                np.clip(te_means + te_stds, 0.0, 1.0),
                color=colors.get(lt, "black"),
                alpha=0.15,
                linewidth=0,
            )
            ax.plot(layers, tr_means, color=colors.get(lt, "black"), linestyle="--", alpha=0.6, label=f"{lt} (train)")

        ax.set_title(metric_titles[metric])
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_titles[metric])
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)

    # last panel: legend + config text
    ax = axes[-1]
    ax.axis("off")
    ax.legend(*axes[0].get_legend_handles_labels(), loc="upper left", frameon=False)

    hp = results.get("hyperparameters", {})
    cfg_text = (
        f"Model: {tag}\n"
        f"Hidden: {hp.get('hidden_sizes')}\n"
        f"Dropout: {hp.get('dropout')}\n"
        f"Weight decay: {hp.get('weight_decay')}\n"
        f"LR: {hp.get('lr')}\n"
        f"Folds: {hp.get('cv_folds')}"
    )
    ax.text(0.02, 0.35, cfg_text, fontsize=10, va="top")

    fig.suptitle(f"Big MLP Probe Sweep ({tag})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = output_dir / f"{tag}_big_mlp_sweep_{cache_pkl['metadata']['dataset']}_{'-'.join(label_types)}.pdf"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    logger.info(f"Saved plot: {plot_path}")
    return plot_path



# ---------------------------
# --- MAIN EXECUTION FLOW ---
# ---------------------------

def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Run biggest/deepest MLP against ilsp/lsp/self and plot metrics.")
    parser.add_argument(
        "--cache_pkls",
        type=str,
        required=True,
        help="Comma-separated list of activation cache PKLs.",
    )
    parser.add_argument("--layers", type=str, required=False, help="Comma-separated true layer indices.")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="experiment_plots/big_mlp_label_sweep")

    # Fixed-big-MLP hyperparameters (strong regularization + dropout)
    parser.add_argument("--hidden_sizes", type=str, default="512,256", help="Comma-separated hidden sizes.")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=512)

    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = out_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    cache_pkls = [Path(p.strip()) for p in args.cache_pkls.split(",") if p.strip()]
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()] if args.layers else None
    hidden_sizes = tuple(int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip())

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Caches: {cache_pkls}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Hidden sizes: {hidden_sizes}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"LR: {args.lr}")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info(f"Patience: {args.patience}")
    logger.info(f"Batch size: {args.batch_size}")

    label_types = ["ilsp", "lsp", "self"]

    for cache_pkl in cache_pkls:
        res = sweep_cache(
            cache_pkl=cache_pkl,
            layers=layers,
            label_types=label_types,
            cv_folds=args.cv_folds,
            hidden_sizes=hidden_sizes,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            output_dir=output_dir,
            logger=logger,
        )
        plot_results(res, output_dir=output_dir, logger=logger)

    logger.info("=" * 80)
    logger.info("BIG MLP LABEL SWEEP - COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
