import argparse
import pickle
import logging
import sys
import warnings
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
import numpy as np

# --- Configuration & Safety ---
# Limit CPU threads to avoid fighting with GPU drivers
os.environ["OMP_NUM_THREADS"] = "4" 
# Suppress annoying warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

import torch
import torch.nn as nn
import torch.optim as optim

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("search_stable.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    label_type: Literal['lsp', 'self', 'ilsp'] = 'self'

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_data(path: Path, cfg: DataConfig):
    logger.info(f"Loading activations: {path}")
    with open(path, "rb") as f:
        cache = pickle.load(f)
    samples = []
    for k, s in cache["data"].items():
        # Determine label based on config
        if cfg.label_type == "self": label = 1 if s.get("self_label") == "self" else 0
        elif cfg.label_type == "ilsp": label = 1 if s.get("gold_label") == "ilsp" else 0
        else: label = 1 if s.get("gold_label") == "lsp" else 0
        
        pid = f"{s.get('id')}_{s.get('reference')}"
        for view in ["forward_activations", "backward_activations"]:
            if view in s:
                samples.append({"pid": pid, "label": label, "act": s[view]})
    return samples

def run_layer_stable(samples, layer, n_folds, n_iter):
    # Prepare data
    pids = np.array([s["pid"] for s in samples])
    labels = np.array([s["label"] for s in samples])
    u_pids, u_idx = np.unique(pids, return_index=True)
    u_labels = labels[u_idx]
    
    # Pre-stack activations for this layer
    X_all = np.stack([s["act"][layer] for s in samples])
    y_all = np.array([s["label"] for s in samples])
    
    # Outer Loop: Test Set Assessment
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    print(f"\n{'Method':<12} | {'Layer':<5} | {'Acc':<7} | {'Gap':<7} | {'AUC':<7}")
    print("-" * 55)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(u_pids, u_labels)):
        # Identify Train/Test split based on Pair IDs (prevents leakage)
        tr_set = set(u_pids[tr_idx])
        mask = np.array([s["pid"] in tr_set for s in samples])
        
        # Standard Scaling is mandatory for these models
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_all[mask])
        X_te = sc.transform(X_all[~mask])
        y_tr, y_te = y_all[mask], y_all[~mask]

        models = {}

        # # --- 1. Logistic Regression (LBFGS for speed) ---
        # # Randomized search over C (Regularization)
        # try:
        #     lr_search = RandomizedSearchCV(
        #         LogisticRegression(max_iter=1000, solver='lbfgs'), 
        #         {'C': np.logspace(-5, 5, 20)}, 
        #         n_iter=n_iter, cv=3, n_jobs=4, random_state=42
        #     ).fit(X_tr, y_tr)
        #     models["LR"] = lr_search.best_estimator_
        # except Exception as e:
        #     logger.error(f"LR Error: {e}")

        # # --- 2. PCA Pipeline ---
        # # Search over n_components and C
        # try:
        #     pca_pipe = Pipeline([('pca', PCA()), ('lr', LogisticRegression(solver='lbfgs'))])
        #     pca_search = RandomizedSearchCV(
        #         pca_pipe,
        #         {'pca__n_components': [10, 50, 100, 256], 'lr__C': [0.01, 0.1, 1.0, 10]},
        #         n_iter=n_iter, cv=3, n_jobs=4, random_state=42
        #     ).fit(X_tr, y_tr)
        #     models["PCA-LR"] = pca_search.best_estimator_
        # except Exception as e:
        #     logger.error(f"PCA Error: {e}")

        # # --- 3. XGBoost (GPU Accelerated) ---
        # if XGBClassifier:
        #     try:
        #         xgb_search = RandomizedSearchCV(
        #             XGBClassifier(tree_method='hist', device='cuda', eval_metric='logloss'),
        #             {
        #                 'n_estimators': [100, 200],
        #                 'max_depth': [3, 5], 
        #                 'learning_rate': [0.01, 0.1, 0.2]
        #             },
        #             n_iter=n_iter, cv=3, n_jobs=1, # GPU handles parallelism
        #             random_state=42
        #         ).fit(X_tr, y_tr)
        #         models["XGB"] = xgb_search.best_estimator_
        #     except Exception as e:
        #         logger.error(f"XGB Error: {e}")

        # --- 4. MLP (GPU Accelerated) ---
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_size = X_tr.shape[1]
            best_acc = 0
            best_model = None
            for hidden_sizes in [(32,), (64,), (64, 32), (512, 128), (128,), (256, 128)]:
                for alpha in [0.0001, 0.01, 0.1]:
                    model = SimpleMLP(input_size, hidden_sizes).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=alpha)
                    criterion = nn.BCEWithLogitsLoss()
                    # Split for validation
                    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
                    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
                    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
                    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
                    # Train for 100 epochs
                    for epoch in range(100):
                        model.train()
                        optimizer.zero_grad()
                        outputs = model(X_train_t)
                        loss = criterion(outputs, y_train_t)
                        loss.backward()
                        optimizer.step()
                    # Eval on val
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_t)
                        val_preds = (torch.sigmoid(val_outputs) > 0.5).float().cpu().numpy().flatten()
                        val_acc = accuracy_score(y_val, val_preds)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_model = model
            models["MLP"] = best_model
        except Exception as e:
            logger.error(f"MLP Error: {e}")

        # --- Report Results for this Fold ---
        for name, best_model in models.items():
            te_preds = best_model.predict(X_te)
            tr_preds = best_model.predict(X_tr)
            
            te_acc = accuracy_score(y_te, te_preds)
            tr_acc = accuracy_score(y_tr, tr_preds)
            
            try:
                if hasattr(best_model, "predict_proba"):
                    te_auc = roc_auc_score(y_te, best_model.predict_proba(X_te)[:,1])
                else:
                    te_auc = 0.0
            except:
                te_auc = 0.0

            print(f"{name:<12} | {layer:<5} | {te_acc:.3f}   | {tr_acc-te_acc:.3f} | {te_auc:.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_pkl", type=str, required=True)
    parser.add_argument("--label_type", type=str, default="self")
    parser.add_argument("--layers", type=str, default="26,27,30,31")
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--n_iter", type=int, default=10, help="Number of random hyperparams to try per fold")
    args = parser.parse_args()

    layers = [int(l) for l in args.layers.split(",")]
    samples = load_data(Path(args.cache_pkl), DataConfig(label_type=args.label_type))
    
    for l in layers:
        run_layer_stable(samples, l, args.cv_folds, args.n_iter)

if __name__ == "__main__":
    main()