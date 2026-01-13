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
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

# Silence scikit-learn version and convergence noise
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

@dataclass
class DataConfig:
    label_type: Literal['lsp', 'self', 'ilsp'] = 'lsp'
    random_state: int = 42

def _parse_cache_key(key: Any, sample: dict) -> tuple[Any, str, str]:
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
        
        pair_id = f"{idx}_{ref}" # Grouping key to prevent prompt leakage
        
        for view in ["forward_activations", "backward_activations"]:
            if view in s:
                samples.append({
                    "pair_id": pair_id,
                    "label": label,
                    "activations": s[view]
                })
    return samples

def run_cv_tournament(samples, layer, n_folds=5):
    # Setup stratified K-Fold based on unique pairs
    pair_ids = np.array([s["pair_id"] for s in samples])
    labels = np.array([s["label"] for s in samples])
    unique_pairs, pair_indices = np.unique(pair_ids, return_index=True)
    pair_labels = labels[pair_indices]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    stats = {}

    for fold, (train_pair_idx, test_pair_idx) in enumerate(skf.split(unique_pairs, pair_labels)):
        train_set = set(unique_pairs[train_pair_idx])
        
        # Prepare data for this fold
        X_tr_raw = np.stack([s["activations"][layer] for s in samples if s["pair_id"] in train_set])
        y_tr = np.array([s["label"] for s in samples if s["pair_id"] in train_set])
        X_te_raw = np.stack([s["activations"][layer] for s in samples if s["pair_id"] not in train_set])
        y_te = np.array([s["label"] for s in samples if s["pair_id"] not in train_set])

        # 1. StandardScaler (Z-score: Mean 0, Var 1)
        sc = StandardScaler()
        X_tr_sc, X_te_sc = sc.fit_transform(X_tr_raw), sc.transform(X_te_raw)

        # 2. Normalizer (Unit Norm: Scales sample vector length to 1)
        norm = Normalizer(norm='l2')
        X_tr_un, X_te_un = norm.fit_transform(X_tr_raw), norm.transform(X_te_raw)

        # Define trials with specific scaling methods
        trials = {
            "MD (Standard)": ("MD", X_tr_sc, X_te_sc),
            "MD (UnitNorm)": ("MD", X_tr_un, X_te_un),
            "LR-L2 (Standard)": (LogisticRegression(C=0.1, solver='lbfgs', n_jobs=-1), X_tr_sc, X_te_sc),
            "LR-L2 (UnitNorm)": (LogisticRegression(C=0.1, solver='lbfgs', n_jobs=-1), X_tr_un, X_te_un),
            "LR-L1 (Standard)": (LogisticRegression(penalty='l1', C=0.1, solver='liblinear'), X_tr_sc, X_te_sc),
            "KBest(500)+LR": ("KBEST", X_tr_sc, X_te_sc),
            "PCA(128)+LR": ("PCA", X_tr_sc, X_te_sc)
        }

        for name, (model, xtr, xte) in trials.items():
            if name not in stats: stats[name] = {"tr": [], "te": []}
            
            if model == "MD":
                w = xtr[y_tr==1].mean(0) - xtr[y_tr==0].mean(0)
                tr_p, te_p = (xtr @ w > 0), (xte @ w > 0)
            elif model == "KBEST":
                sel = SelectKBest(f_classif, k=500).fit(xtr, y_tr)
                clf = LogisticRegression(C=0.1).fit(sel.transform(xtr), y_tr)
                tr_p, te_p = clf.predict(sel.transform(xtr)), clf.predict(sel.transform(xte))
            elif model == "PCA":
                pca = PCA(n_components=128, random_state=42).fit(xtr)
                clf = LogisticRegression(C=0.1).fit(pca.transform(xtr), y_tr)
                tr_p, te_p = clf.predict(pca.transform(xtr)), clf.predict(pca.transform(xte))
            else:
                model.fit(xtr, y_tr)
                tr_p, te_p = model.predict(xtr), model.predict(xte)

            stats[name]["tr"].append(accuracy_score(y_tr, tr_p))
            stats[name]["te"].append(accuracy_score(y_te, te_p))

    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_pkl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--label_type", type=str, default="lsp")
    parser.add_argument("--layers", type=str, default="16")
    parser.add_argument("--cv_folds", type=int, default=5)
    args = parser.parse_args()

    layer_list = [int(l.strip()) for l in args.layers.split(",")]
    cfg = DataConfig(label_type=args.label_type)
    samples = load_from_cache_flat(Path(args.cache_pkl), cfg)
    
    print(f"\n--- TOURNAMENT: {len(samples)} samples | Folds: {args.cv_folds} ---")
    print(f"{'Method':<20} | {'Layer':<5} | {'Train':<7} | {'Test':<7} | {'Gap':<7}")
    print("-" * 65)

    for layer in layer_list:
        stats = run_cv_tournament(samples, layer, n_folds=args.cv_folds)
        for name, res in stats.items():
            tr, te = np.mean(res["tr"]), np.mean(res["te"])
            gap = tr - te
            color = "\033[91m" if gap > 0.15 else "\033[92m"
            reset = "\033[0m"
            print(f"{name:<20} | {layer:<5} | {tr:.3f}   | {te:.3f}   | {color}{gap:.3f}{reset}")

if __name__ == "__main__":
    main()