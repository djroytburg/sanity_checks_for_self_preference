import argparse
import pickle
import warnings
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

# Silence scikit-learn noise
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
    """Loads samples and calculates the Oracle Accuracy ceiling."""
    with open(path, "rb") as f:
        cache = pickle.load(f)
    data = cache["data"]
    
    samples = []
    oracle_matches = 0
    total_samples = 0

    for k, s in data.items():
        idx, judge, ref = _parse_cache_key(k, s)
        
        # Determine target label
        if data_config.label_type == "lsp": label = 1 if s.get("gold_label") == "lsp" else 0
        elif data_config.label_type == "ilsp": label = 1 if s.get("gold_label") == "ilsp" else 0
        else: label = 1 if s.get("self_label") == "self" else 0
        
        # Oracle: Does the model's text generation actually match the label?
        # Note: We divide total_samples by 2 because each pair has fwd/bwd views
        gen_text = str(s.get("forward_gen_text", "")).lower()
        # This check should be tuned to your specific prompt/output format
        if (label == 1 and "choice a" in gen_text) or (label == 0 and "choice b" in gen_text):
            oracle_matches += 1
        total_samples += 1

        pair_id = f"{idx}_{ref}"
        for view in ["forward_activations", "backward_activations"]:
            if view in s:
                samples.append({"pair_id": pair_id, "label": label, "activations": s[view]})

    oracle_acc = oracle_matches / (total_samples / 2) if total_samples > 0 else 0
    return samples, oracle_acc

def run_tournament(samples, layer, n_folds=5):
    pair_ids = np.array([s["pair_id"] for s in samples])
    labels = np.array([s["label"] for s in samples])
    unique_pairs, pair_indices = np.unique(pair_ids, return_index=True)
    pair_labels = labels[pair_indices]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    stats = {}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(unique_pairs, pair_labels)):
        train_set = set(unique_pairs[tr_idx])
        X_tr_raw = np.stack([s["activations"][layer] for s in samples if s["pair_id"] in train_set])
        y_tr = np.array([s["label"] for s in samples if s["pair_id"] in train_set])
        X_te_raw = np.stack([s["activations"][layer] for s in samples if s["pair_id"] not in train_set])
        y_te = np.array([s["label"] for s in samples if s["pair_id"] not in train_set])

        # Feature Scaling Variants
        rs = RobustScaler().fit(X_tr_raw)
        X_tr_rs, X_te_rs = rs.transform(X_tr_raw), rs.transform(X_te_raw)
        
        ss = StandardScaler().fit(X_tr_raw)
        X_tr_ss, X_te_ss = ss.transform(X_tr_raw), ss.transform(X_te_raw)

        un = Normalizer().fit(X_tr_raw)
        X_tr_un, X_te_un = un.transform(X_tr_raw), un.transform(X_te_raw)

        trials = {
            "LR-L2 (Robust)": (LogisticRegression(C=0.1, n_jobs=-1), X_tr_rs, X_te_rs),
            "LR-L2 (Unit)": (LogisticRegression(C=0.1, n_jobs=-1), X_tr_un, X_te_un),
            "LR-L1 (Aggressive)": (LogisticRegression(penalty='elasticnet', l1_ratio=1.0, C=0.01, solver='saga'), X_tr_ss, X_te_ss),
            "PCA(256)+LR": ("PCA", X_tr_ss, X_te_ss),
            "SelectK(512)+LR": ("KBEST", X_tr_ss, X_te_ss)
        }

        for name, (model, xtr, xte) in trials.items():
            if name not in stats: stats[name] = {"tr": [], "te": [], "tr_auc": [], "te_auc": []}
            if name.startswith("PCA"):
                pca = PCA(n_components=256, random_state=42).fit(xtr)
                clf = LogisticRegression(C=0.1).fit(pca.transform(xtr), y_tr)
                tr_p, te_p = clf.predict(pca.transform(xtr)), clf.predict(pca.transform(xte))
                tr_proba, te_proba = clf.predict_proba(pca.transform(xtr))[:,1], clf.predict_proba(pca.transform(xte))[:,1]
            elif name.startswith("SelectK"):
                sel = SelectKBest(f_classif, k=512).fit(xtr, y_tr)
                clf = LogisticRegression(C=0.1).fit(sel.transform(xtr), y_tr)
                tr_p, te_p = clf.predict(sel.transform(xtr)), clf.predict(sel.transform(xte))
                tr_proba, te_proba = clf.predict_proba(sel.transform(xtr))[:,1], clf.predict_proba(sel.transform(xte))[:,1]
            else:
                model.fit(xtr, y_tr)
                tr_p, te_p = model.predict(xtr), model.predict(xte)
                tr_proba, te_proba = model.predict_proba(xtr)[:,1], model.predict_proba(xte)[:,1]
            
            stats[name]["tr"].append(accuracy_score(y_tr, tr_p))
            stats[name]["te"].append(accuracy_score(y_te, te_p))
            stats[name]["tr_auc"].append(roc_auc_score(y_tr, tr_proba))
            stats[name]["te_auc"].append(roc_auc_score(y_te, te_proba))

    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_pkl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--label_type", type=str, default="lsp")
    parser.add_argument("--layers", type=str, default="24,26,28")
    parser.add_argument("--cv_folds", type=int, default=5)
    args = parser.parse_args()

    layer_list = [int(l.strip()) for l in args.layers.split(",")]
    cfg = DataConfig(label_type=args.label_type)
    
    samples, oracle_acc = load_from_cache_flat(Path(args.cache_pkl), cfg)
    
    print(f"\n--- ORACLE CEILING ---")
    print(f"Llama-3.1 Behavior Matches Gold Labels: {oracle_acc*100:.2f}%")
    print(f"If your probe hits this number, it is 100% optimal for this model.\n")

    print(f"{'Method':<20} | {'Layer':<5} | {'Train':<7} | {'Test':<7} | {'Gap':<7} | {'Tr AUC':<7} | {'Te AUC':<7}")
    print("-" * 85)

    for layer in layer_list:
        stats = run_tournament(samples, layer, n_folds=args.cv_folds)
        for name, res in stats.items():
            tr, te = np.mean(res["tr"]), np.mean(res["te"])
            tr_auc, te_auc = np.mean(res["tr_auc"]), np.mean(res["te_auc"])
            gap = tr - te
            color = "\033[91m" if gap > 0.15 else "\033[92m"
            print(f"{name:<20} | {layer:<5} | {tr:.3f}   | {te:.3f}   | {color}{gap:.3f}\033[0m | {tr_auc:.3f} | {te_auc:.3f}")

if __name__ == "__main__":
    main()