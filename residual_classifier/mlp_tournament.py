import argparse
import pickle
import logging
import sys
import warnings
import os
import copy
import itertools
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)

# --- Config ---
os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[logging.FileHandler("grid_search.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Colors for Terminal ---
class Colors:
    RESET = "\033[0m"
    GREEN = "\033[92m"  # > 0.80
    BLUE = "\033[94m"   # > 0.85
    RED = "\033[91m"    # Fail / Low

def fmt(val):
    """Color codes values based on thresholds."""
    if val >= 0.85: return f"{Colors.BLUE}{val:.3f}{Colors.RESET}"
    if val >= 0.80: return f"{Colors.GREEN}{val:.3f}{Colors.RESET}"
    return f"{val:.3f}"

@dataclass
class DataConfig:
    label_type: Literal['lsp', 'self', 'ilsp'] = 'self'

class RobustMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.BatchNorm1d(h)) # Added BatchNorm for stability
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_data(path: Path, cfg: DataConfig):
    logger.info(f"Loading activations from {path}...")
    with open(path, "rb") as f:
        cache = pickle.load(f)
    samples = []
    for k, s in cache["data"].items():
        if cfg.label_type == "self": label = 1 if s.get("self_label") == "self" else 0
        elif cfg.label_type == "ilsp": label = 1 if s.get("gold_label") == "ilsp" else 0
        else: label = 1 if s.get("gold_label") == "lsp" else 0
        
        pid = f"{s.get('id')}_{s.get('reference')}"
        for view in ["forward_activations", "backward_activations"]:
            if view in s:
                samples.append({"pid": pid, "label": label, "act": s[view]})
    return samples

def get_metrics(y_true, probs):
    preds = (probs > 0.5).astype(int)
    return {
        "acc": accuracy_score(y_true, preds),
        "auc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5,
        "f1": f1_score(y_true, preds, average='macro'),
        "prec": precision_score(y_true, preds, zero_division=0),
        "rec": recall_score(y_true, preds, zero_division=0)
    }

def train_one_config(X_tr, y_tr, X_val, y_val, config, device):
    input_size = X_tr.shape[1]
    model = RobustMLP(input_size, config['hidden'], config['dropout']).to(device)
    
    # Use DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    criterion = nn.BCEWithLogitsLoss()
    
    # Batching (crucial for speed on GPUs)
    batch_size = 256
    n_samples = X_tr.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    # y_val is numpy for metrics

    history = {'train_loss': [], 'val_auc': []}
    best_val_auc = 0
    patience = 15
    counter = 0
    best_state = None

    for epoch in range(100): # Hard cap 100 epochs
        model.train()
        avg_loss = 0
        
        # Mini-batch loop
        perm = torch.randperm(n_samples)
        for i in range(n_batches):
            idx = perm[i*batch_size : (i+1)*batch_size]
            bx, by = X_tr_t[idx].to(device), y_tr_t[idx].to(device)
            
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        
        history['train_loss'].append(avg_loss / n_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            try:
                val_auc = roc_auc_score(y_val, probs)
            except:
                val_auc = 0.5
        
        history['val_auc'].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    print(f"done with training.", end="")
    print(f"  Best Val AUC: {best_val_auc:.4f} at epoch {epoch+1}")
    return model, best_val_auc, history

def plot_analytics(history, y_true, y_probs, layer, output_dir):
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Loss Trajectory
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title(f"L{layer} Loss Trajectory")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    
    # 2. Calibration / Confidence Hist
    plt.subplot(1, 2, 2)
    sns.histplot(y_probs[y_true==0], color='red', label='Class 0', kde=True, alpha=0.5)
    sns.histplot(y_probs[y_true==1], color='blue', label='Class 1', kde=True, alpha=0.5)
    plt.title(f"L{layer} Post-Hoc Confidence")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/L{layer}_analytics.png")
    plt.close()

def run_full_sweep(samples, layer, n_folds, output_dir):
    # Setup Data
    pids = np.array([s["pid"] for s in samples])
    labels = np.array([s["label"] for s in samples])
    u_pids, u_idx = np.unique(pids, return_index=True)
    u_labels = labels[u_idx]
    
    X_all = np.stack([s["act"][layer] for s in samples])
    y_all = np.array([s["label"] for s in samples])
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Define The Grid ---
    # Reduced grid for faster search, added more regularization options and model architectures
    param_grid = {
        'hidden': [(32,), (64,), (128, 64), (256, 128), (512, 256)],
        'lr': [0.001, 0.0001],
        'wd': [0.01, 0.05],
        'dropout': [0.2, 0.4]
    }
    
    keys, values = zip(*param_grid.items())
    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n{'='*80}")
    print(f"LAYER {layer}: Starting Grid Search over {len(grid)} configs across {n_folds} folds.")
    print(f"{'='*80}")
    
    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(u_pids, u_labels)):
        tr_set = set(u_pids[tr_idx])
        mask = np.array([s["pid"] in tr_set for s in samples])
        
        # Scaling
        sc = StandardScaler()
        X_fold_tr = sc.fit_transform(X_all[mask])
        X_test = sc.transform(X_all[~mask])
        y_fold_tr = y_all[mask]
        y_test = y_all[~mask]

        # Inner Split for Grid Search
        X_train, X_val, y_train, y_val = train_test_split(X_fold_tr, y_fold_tr, test_size=0.3, random_state=42)
        
        best_fold_model = None
        best_fold_auc = -1
        best_fold_cfg = {}
        best_fold_hist = {}

        # Run Grid
        for i, cfg in enumerate(grid):
            # Progress bar dot
            print(".", end="", flush=True) 
            model, val_auc, hist = train_one_config(X_train, y_train, X_val, y_val, cfg, device)
            
            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                best_fold_model = model
                best_fold_cfg = cfg
                best_fold_hist = hist
        
        print(f" Fold {fold+1} Done.")

        # Final Eval
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        X_tr_t = torch.tensor(X_fold_tr, dtype=torch.float32).to(device) # Full train for metrics
        
        best_fold_model.eval()
        with torch.no_grad():
            te_probs = torch.sigmoid(best_fold_model(X_test_t)).cpu().numpy().flatten()
            tr_probs = torch.sigmoid(best_fold_model(X_tr_t)).cpu().numpy().flatten()
        
        te_metrics = get_metrics(y_test, te_probs)
        tr_metrics = get_metrics(y_fold_tr, tr_probs)
        
        fold_results.append({
            'tr': tr_metrics, 'te': te_metrics, 'cfg': best_fold_cfg, 
            'hist': best_fold_hist, 'probs': te_probs, 'y_true': y_test
        })

    # --- Aggregate & Report ---
    avg_te_auc = np.mean([r['te']['auc'] for r in fold_results])
    best_fold_idx = np.argmax([r['te']['auc'] for r in fold_results])
    best_run = fold_results[best_fold_idx]
    
    print(f"\n{Colors.BLUE}>> LAYER {layer} WINNER CONFIG: {best_run['cfg']} {Colors.RESET}")
    
    # Print Table
    print("-" * 115)
    print(f"{'Metric':<10} | {'Train':<15} | {'Test':<15}")
    print("-" * 115)
    for m in ['acc', 'auc', 'f1', 'prec', 'rec']:
        tr_val = np.mean([r['tr'][m] for r in fold_results])
        te_val = np.mean([r['te'][m] for r in fold_results])
        print(f"{m.upper():<10} | {fmt(tr_val):<25} | {fmt(te_val):<25}")
    print("-" * 115)

    # Plot
    plot_analytics(best_run['hist'], best_run['y_true'], best_run['probs'], layer, output_dir)
    print(f"Saved analytics plots to {output_dir}/L{layer}_analytics.png\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_pkl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="grid_results")
    parser.add_argument("--label_type", type=str, default="self")
    parser.add_argument("--layers", type=str, default="26,27,30,31")
    parser.add_argument("--cv_folds", type=int, default=3)
    args = parser.parse_args()

    layers = [int(l) for l in args.layers.split(",")]
    samples = load_data(Path(args.cache_pkl), DataConfig(label_type=args.label_type))
    
    print(f"Loaded {len(samples)} samples. Running on {torch.cuda.device_count()} GPUs.")
    
    for l in layers:
        run_full_sweep(samples, l, args.cv_folds, args.output_dir)

if __name__ == "__main__":
    main()