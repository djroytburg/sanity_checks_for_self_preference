import argparse
import logging
import pickle
import warnings
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

# Silence the scikit-learn 1.8+ and convergence clutter
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- DATA SCAFFOLDING ---

@dataclass
class DataConfig:
    label_type: Literal['lsp', 'self', 'ilsp'] = 'lsp'
    train_split: float = 0.8
    random_state: int = 42

def _parse_cache_key(key: Any, sample: dict) -> tuple[Any, str, str]:
    if isinstance(key, tuple) and len(key) == 2:
        return key[0], str(sample.get("judge")), str(key[1])
    if isinstance(key, tuple) and len(key) == 3:
        return key
    return sample.get("id"), str(sample.get("judge")), str(sample.get("reference"))

def load_from_cache_flat(path: Path, data_config: DataConfig):
    """Loads all samples into a list for group-safe CV processing."""
    with open(path, "rb") as f:
        cache = pickle.load(f)
    data = cache["data"]
    
    samples = []
    for k, s in data.items():
        idx, judge, ref = _parse_cache_key(k, s)
        label = 0
        if data_config.label_type == "lsp": label = 1 if s.get("gold_label") == "lsp" else 0
        elif data_config.label_type == "ilsp": label = 1 if s.get("gold_label") == "ilsp" else 0
        else: label = 1 if s.get("self_label") == "self" else 0
        
        # Unique ID to prevent leakage (forward/backward versions of same content)
        pair_id = f"{idx}_{ref}"
        
        for view in ["forward_activations", "backward_activations"]:
            if view in s:
                samples.append({
                    "pair_id": pair_id,
                    "label": label,
                    "activations": s[view]
                })
    return samples

# --- TOURNAMENT LOGIC ---

def run_cv_tournament(samples, layer, n_folds=5):
    # Prepare unique pairs for group-safe stratified splitting
    pair_ids = np.array([s["pair_id"] for s in samples])
    labels = np.array([s["label"] for s in samples])
    unique_pairs, pair_indices = np.unique(pair_ids, return_index=True)
    pair_labels = labels[pair_indices]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    tournament_stats = {}

    for fold, (train_pair_idx, test_pair_idx) in enumerate(skf.split(unique_pairs, pair_labels)):
        train_pairs = set(unique_pairs[train_pair_idx])
        
        # Index-based selection for speed
        X_tr_raw = np.stack([s["activations"][layer] for s in samples if s["pair_id"] in train_pairs])
        y_tr = np.array([s["label"] for s in samples if s["pair_id"] in train_pairs])
        X_te_raw = np.stack([s["activations"][layer] for s in samples if s["pair_id"] not in train_pairs])
        y_te = np.array([s["label"] for s in samples if s["pair_id"] not in train_pairs])

        # Feature Scaling (Crucial for high-dim probes)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)

        # FAST SOLVERS: liblinear for L1, lbfgs for L2. n_jobs uses all CPU cores.
        trials = {
            "MD (Centroid)": "MD",
            "LR-L2 (C=1.0)": LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=500, n_jobs=-1),
            "LR-L2 (C=0.01)": LogisticRegression(penalty='l2', C=0.01, solver='lbfgs', max_iter=500, n_jobs=-1),
            "LR-L1 (C=0.1)": LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=500),
            "PCA(128)+LR": "PCA_LR"
        }

        for name, model in trials.items():
            if name not in tournament_stats: tournament_stats[name] = {"tr": [], "te": []}
            
            if model == "MD":
                mu_pos, mu_neg = X_tr[y_tr==1].mean(0), X_tr[y_tr==0].mean(0)
                w = mu_pos - mu_neg
                tr_p, te_p = (X_tr @ w > 0), (X_te @ w > 0)
            elif model == "PCA_LR":
                pca = PCA(n_components=128, random_state=42).fit(X_tr)
                clf = LogisticRegression(max_iter=500, n_jobs=-1).fit(pca.transform(X_tr), y_tr)
                tr_p, te_p = clf.predict(pca.transform(X_tr)), clf.predict(pca.transform(X_te))
            else:
                model.fit(X_tr, y_tr)
                tr_p, te_p = model.predict(X_tr), model.predict(X_te)

            tournament_stats[name]["tr"].append(accuracy_score(y_tr, tr_p))
            tournament_stats[name]["te"].append(accuracy_score(y_te, te_p))

    return tournament_stats

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Probing Tournament with CV")
    parser.add_argument("--cache_pkl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--label_type", type=str, default="lsp")
    parser.add_argument("--layers", type=str, default="16")
    parser.add_argument("--cv_folds", type=int, default=5)
    args = parser.parse_args()

    layer_list = [int(x.strip()) for x in args.layers.split(",")]
    cfg = DataConfig(label_type=args.label_type)
    
    print(f"Loading cache: {args.cache_pkl}")
    samples = load_from_cache_flat(Path(args.cache_pkl), cfg)
    print(f"Total Samples: {len(samples)} | Folds: {args.cv_folds}")

    print(f"\n{'Method':<20} | {'Layer':<5} | {'Train':<7} | {'Test':<7} | {'Gap':<7}")
    print("-" * 65)

    for layer in layer_list:
        stats = run_cv_tournament(samples, layer, n_folds=args.cv_folds)
        
        for name, res in stats.items():
            tr_m, te_m = np.mean(res["tr"]), np.mean(res["te"])
            gap = tr_m - te_m
            # Color code the gap: Red for high overfitting, Green for robust
            color = "\033[91m" if gap > 0.15 else "\033[92m"
            reset = "\033[0m"
            print(f"{name:<20} | {layer:<5} | {tr_m:.3f}   | {te_m:.3f}   | {color}{gap:.3f}{reset}")

if __name__ == "__main__":
    main()