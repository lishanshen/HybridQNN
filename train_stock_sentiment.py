"""
Usage:
  python train_stock_sentiment.py                    # Single training + evaluation
  python train_stock_sentiment.py --n_runs 5         # Run 5 seeds, report mean ± std
  python train_stock_sentiment.py --skip_baselines   # Run main model only, skip ablation baselines
"""

import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pennylane as qml
from pennylane.templates import AngleEmbedding
import matplotlib.pyplot as plt
import json
import time
from scipy import stats as scipy_stats

# Import common utilities
from utils import (
    load_data,
    prepare_features, clean_data,
    DataScaler, preprocess_features_for_angle,
    evaluate_model, print_metrics,
    plot_predictions, plot_training_curve, plot_comparison,
    get_device, save_model_artifacts
)


# ============================================================
#  Quantum circuit components (improved version)
# ============================================================

N_QUBITS = 10          # Number of qubits (equal to the number of features)
N_LAYERS = 4           # Number of variational layers
ENTANGLING = "circular"  # Changed to ring entanglement (was "full")
N_Q_FEATURES = 19      # Quantum output dimension: 10 Z + 9 nearest-neighbor ZZ correlations


def get_quantum_device(n_qubits=N_QUBITS):
    """Get a PennyLane quantum device, preferring GPU."""
    if torch.cuda.is_available():
        try:
            return qml.device("lightning.gpu", wires=n_qubits)
        except Exception:
            print("Cannot use lightning.gpu, falling back to lightning.qubit")
    return qml.device("lightning.qubit", wires=n_qubits)


def encode_features_layer(features, wires, rotation="X"):
    """Angle encoding: map classical features to rotation-gate parameters."""
    AngleEmbedding(features, wires=wires, rotation=rotation)


def variational_layer(weights, wires, entangling_pattern="circular"):
    """
    Single variational layer: Rot(alpha, beta, gamma) on each qubit → CNOT entanglement

    Weight shape: (n_wires, 3) — 3 rotation parameters per qubit
    """
    n_w = len(wires)
    for i in range(n_w):
        qml.Rot(*weights[i], wires=wires[i])

    if entangling_pattern == "linear":
        for i in range(n_w - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
    elif entangling_pattern == "circular":
        for i in range(n_w - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        qml.CNOT(wires=[wires[-1], wires[0]])
    elif entangling_pattern == "full":
        for i in range(n_w):
            for j in range(i + 1, n_w):
                qml.CNOT(wires=[wires[i], wires[j]])


def data_reuploading_block(weights, features, wires, n_layers=4, entangling_pattern="circular"):
    """
    Data re-uploading variational block:
    Each layer = encode features → variational layer (rotation + entanglement)
    Repeated n_layers times

    Weight shape: (n_layers, n_wires, 3)
    """
    for layer in range(n_layers):
        encode_features_layer(features, wires=wires, rotation="X")
        variational_layer(weights[layer], wires=wires, entangling_pattern=entangling_pattern)


#  Quantum circuit QNode
dev = get_quantum_device()

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(params, x):
    """
    Quantum circuit:
      1. Data re-uploading + variational block
      2. Read Z expectation values + nearest-neighbor ZZ correlation expectation values
    Returns: 19-dimensional vector [Z_0, ..., Z_9, Z_0Z_1, ..., Z_8Z_9]
    """
    wires = range(N_QUBITS)
    data_reuploading_block(params, x, wires=wires, n_layers=N_LAYERS, entangling_pattern=ENTANGLING)

    # Single-qubit Z expectation (10 dimensions)
    z_exps = [qml.expval(qml.Z(i)) for i in range(N_QUBITS)]
    # Nearest-neighbor ZZ correlation (9 dimensions)
    zz_exps = [qml.expval(qml.Z(i) @ qml.Z(i + 1)) for i in range(N_QUBITS - 1)]

    return z_exps + zz_exps


#  Hybrid quantum-neural network model
class HybridQNN(nn.Module):
    def __init__(self, n_qubits, n_classical, n_layers, dropout=0.2):
        super().__init__()
        # Quantum parameters: always on CPU
        self.quantum_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )
        q_output_dim = N_Q_FEATURES  # 19
        input_dim = q_output_dim + n_classical

        self.post_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def to(self, *args, **kwargs):
        """Move only the classical network to the target device; quantum parameters always stay on CPU."""
        self.post_net = self.post_net.to(*args, **kwargs)
        return self

    def forward(self, x_quantum, x_classical):
        if x_quantum.device.type != 'cpu':
            x_quantum = x_quantum.cpu()

        q_raw = quantum_circuit(self.quantum_params, x_quantum)
        q_feat = torch.stack(q_raw, dim=1)

        if q_feat.device != x_classical.device:
            q_feat = q_feat.to(x_classical.device)

        # Concatenate quantum features + classical features
        combined = torch.cat([q_feat, x_classical], dim=1)

        # Classical network output
        return self.post_net(combined)

#  Pure classical MLP (matched parameter count, used for ablation baseline)
class ClassicalMLP(nn.Module):
    """
    Pure classical multilayer perceptron, with parameter count aligned to HybridQNN.
    """

    def __init__(self, n_features, dropout=0.2):
        super().__init__()
        # 10 → 52 → 40 → 24 → 1, ~4150 parameters
        self.net = nn.Sequential(
            nn.Linear(n_features, 52),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(52, 40),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(40, 24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, X_train_q, X_train_c, y_train_t,
                X_val_q, X_val_c, y_val_t,
                device, lr=0.01, weight_decay=1e-5,
                batch_size=64, max_epochs=200, early_stop_patience=25,
                verbose=True):
    """
    Train the hybrid model, returning:
      - best_state: best model weights
      - train_losses, val_losses: loss histories
      - best_val_loss: best validation loss
    """
    model = model.to(device)

    X_train_c = X_train_c.to(device)
    y_train_t = y_train_t.to(device)
    X_val_c = X_val_c.to(device)
    y_val_t = y_val_t.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=10, verbose=verbose)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X_train_q))
        epoch_loss = 0.0

        for start in range(0, len(X_train_q), batch_size):
            end = min(start + batch_size, len(X_train_q))
            idx = perm[start:end]

            optimizer.zero_grad()
            preds = model(X_train_q[idx], X_train_c[idx.to(device)])
            loss = nn.functional.mse_loss(preds, y_train_t[idx.to(device)])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
        epoch_loss /= len(X_train_q)
        train_losses.append(epoch_loss)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_q, X_val_c)
            val_loss = nn.functional.mse_loss(val_preds, y_val_t).item()
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}, best validation loss: {best_val_loss:.6f}")
                break

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{max_epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

    model.load_state_dict(best_state)
    return best_state, train_losses, val_losses, best_val_loss


def train_classical_only(model, X_train, y_train_t, X_val, y_val_t,
                         device, lr=0.01, weight_decay=1e-5,
                         batch_size=64, max_epochs=200, early_stop_patience=25,
                         verbose=True):
    """Train a pure classical MLP model."""
    model = model.to(device)
    X_train = X_train.to(device)
    y_train_t = y_train_t.to(device)
    X_val = X_val.to(device)
    y_val_t = y_val_t.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                   patience=10, verbose=verbose)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        epoch_loss = 0.0

        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            idx = perm[start:end]

            optimizer.zero_grad()
            preds = model(X_train[idx])
            loss = nn.functional.mse_loss(preds, y_train_t[idx])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = nn.functional.mse_loss(val_preds, y_val_t).item()
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}, best validation loss: {best_val_loss:.6f}")
                break

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{max_epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

    model.load_state_dict(best_state)
    return best_state, train_losses, val_losses, best_val_loss


def evaluate_hybrid(model, X_test_q, X_test_c, y_test, scaler, device):
    """Evaluate the hybrid model."""
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds_norm = model(
            X_test_q.to(device), X_test_c.to(device)
        ).cpu().numpy().flatten()
    y_pred = scaler.denormalize_y(preds_norm)
    return evaluate_model(y_test, y_pred), y_pred


def evaluate_classical(model, X_test, y_test, scaler, device):
    """Evaluate the pure classical model."""
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds_norm = model(X_test.to(device)).cpu().numpy().flatten()
    y_pred = scaler.denormalize_y(preds_norm)
    return evaluate_model(y_test, y_pred), y_pred


def run_single_experiment(feature_cols, target_col, df, seed, result_dir,
                          run_baselines=True, use_walk_forward=True, verbose=True):
    """
    A single complete experiment run:
      1. Load data + feature engineering + preprocessing
      2. Train HybridQNN (optimized quantum)
      3. Train random-quantum baseline (frozen quantum parameters)
      4. Train pure classical MLP
      5. Evaluate + compare
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(result_dir, exist_ok=True)

    # ---------- 1. Data preparation ----------
    X_raw, y = prepare_features(df, feature_cols, target_col)
    X_raw, y = clean_data(X_raw, y)

    if verbose:
        print(f"Samples after cleaning: {len(y)}, feature dim: {X_raw.shape[1]}")

    # ---------- 2. Data split ----------
    n = len(X_raw)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    X_train_raw, y_train = X_raw[:train_end], y[:train_end]
    X_val_raw,   y_val   = X_raw[train_end:val_end], y[train_end:val_end]
    X_test_raw,  y_test  = X_raw[val_end:], y[val_end:]

    if verbose:
        print(f"Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

    # ---------- 3. Standardization ----------
    scaler = DataScaler()
    scaler.fit(X_train_raw, y_train)
    X_train_scaled = scaler.transform_X(X_train_raw)
    X_val_scaled = scaler.transform_X(X_val_raw)
    X_test_scaled = scaler.transform_X(X_test_raw)

    y_train_norm = scaler.normalize_y(y_train)
    y_val_norm = scaler.normalize_y(y_val)
    y_test_norm = scaler.normalize_y(y_test)

    # ---------- 4. Quantum encoding ----------
    n_qubits = len(feature_cols)
    X_train_q = preprocess_features_for_angle(X_train_scaled, n_qubits)
    X_val_q = preprocess_features_for_angle(X_val_scaled, n_qubits)
    X_test_q = preprocess_features_for_angle(X_test_scaled, n_qubits)

    # ---------- 5. Device ----------
    device = get_device()

    # ---------- 6. Convert to tensors ----------
    X_train_q_t = torch.tensor(X_train_q, dtype=torch.float32)
    X_val_q_t = torch.tensor(X_val_q, dtype=torch.float32)
    X_test_q_t = torch.tensor(X_test_q, dtype=torch.float32)

    X_train_c_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_c_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_c_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_t = torch.tensor(y_train_norm, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val_norm, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test_norm, dtype=torch.float32).unsqueeze(1)

    results = {}

    # ========== Main model: HybridQNN (train quantum parameters) ==========
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Main model: HybridQNN (data re-uploading + circular entanglement + ZZ readout)")
        print(f"{'='*60}")

    model = HybridQNN(n_qubits, len(feature_cols), N_LAYERS, dropout=0.2)

    t_start = time.time()
    best_state, train_losses, val_losses, best_val_loss = train_model(
        model, X_train_q_t, X_train_c_t, y_train_t,
        X_val_q_t, X_val_c_t, y_val_t,
        device, lr=0.01, weight_decay=1e-5, verbose=verbose
    )
    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    metrics_trained, y_pred_trained = evaluate_hybrid(
        model, X_test_q_t, X_test_c_t, y_test, scaler, device
    )

    if verbose:
        print_metrics(metrics_trained, "HybridQNN (trained quantum) test-set evaluation")
        print(f"Training time: {train_time:.1f}s")

    results["hybrid_trained"] = {
        "metrics": metrics_trained,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "train_time": train_time,
        "y_pred": y_pred_trained.tolist(),
    }

    # Save main model
    save_model_artifacts(best_state, scaler.feature_scaler,
                         scaler.y_mean, scaler.y_std, result_dir)

    # Prediction plot
    plot_predictions(y_test, y_pred_trained,
                     "HybridQNN (optimized, trained quantum) prediction",
                     os.path.join(result_dir, "prediction_plot.png"),
                     metrics_trained)
    plot_training_curve(train_losses, val_losses,
                        "HybridQNN (optimized) training curve",
                        os.path.join(result_dir, "training_curve.png"))

    if not run_baselines:
        return results

    # ========== Baseline 1: HybridQNN (random frozen quantum parameters) ==========
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Baseline 1: HybridQNN (random quantum parameters, only classical layers trained)")
        print(f"{'='*60}")

    model_random = HybridQNN(n_qubits, len(feature_cols), N_LAYERS, dropout=0.2)
    # Freeze quantum parameters
    model_random.quantum_params.requires_grad = False

    # Re-initialize classical layers
    def _reset_post_net(m):
        for layer in m.post_net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    _reset_post_net(model_random)

    best_state_r, train_losses_r, val_losses_r, _ = train_model(
        model_random, X_train_q_t, X_train_c_t, y_train_t,
        X_val_q_t, X_val_c_t, y_val_t,
        device, lr=0.01, weight_decay=1e-5, verbose=verbose
    )

    model_random.load_state_dict(best_state_r)
    metrics_random, y_pred_random = evaluate_hybrid(
        model_random, X_test_q_t, X_test_c_t, y_test, scaler, device
    )

    if verbose:
        print_metrics(metrics_random, "HybridQNN (random quantum) test-set evaluation")

    results["hybrid_random"] = {
        "metrics": metrics_random,
        "train_losses": train_losses_r,
        "val_losses": val_losses_r,
        "y_pred": y_pred_random.tolist(),
    }

    # ========== Baseline 2: pure classical MLP ==========
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Baseline 2: pure classical MLP (matched parameter count)")
        print(f"{'='*60}")

    model_classical = ClassicalMLP(len(feature_cols), dropout=0.2)
    total_classical_params = sum(p.numel() for p in model_classical.parameters())

    best_state_c, train_losses_c, val_losses_c, _ = train_classical_only(
        model_classical, X_train_c_t, y_train_t, X_val_c_t, y_val_t,
        device, lr=0.01, weight_decay=1e-5, verbose=verbose
    )

    model_classical.load_state_dict(best_state_c)
    metrics_classical, y_pred_classical = evaluate_classical(
        model_classical, X_test_c_t, y_test, scaler, device
    )

    if verbose:
        print_metrics(metrics_classical, "Pure classical MLP test-set evaluation")

    results["classical_mlp"] = {
        "metrics": metrics_classical,
        "params": total_classical_params,
        "train_losses": train_losses_c,
        "val_losses": val_losses_c,
        "y_pred": y_pred_classical.tolist(),
    }

    # ========== Comparison summary ==========
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Model comparison summary (seed={seed})")
        print(f"{'='*70}")
        print(f"{'Model':<25} {'MSE':>12} {'R²':>10} {'RMSE':>10} {'MAPE':>8}")
        print("-" * 70)

        for name, data in [
            ("HybridQNN trained quantum", results["hybrid_trained"]),
            ("HybridQNN random quantum", results["hybrid_random"]),
            ("Pure classical MLP", results["classical_mlp"]),
        ]:
            m = data["metrics"]
            print(f"{name:<25} {m['MSE']:>12.2f} {m['R2']:>10.4f} {m['RMSE']:>10.2f} {m['MAPE']:>7.2f}%")

        # Trained quantum vs random quantum
        mse_improvement_q = (results["hybrid_random"]["metrics"]["MSE"] -
                             results["hybrid_trained"]["metrics"]["MSE"])
        r2_improvement_q = (results["hybrid_trained"]["metrics"]["R2"] -
                            results["hybrid_random"]["metrics"]["R2"])

        # Hybrid quantum vs pure classical
        mse_improvement_c = (results["classical_mlp"]["metrics"]["MSE"] -
                             results["hybrid_trained"]["metrics"]["MSE"])
        mse_pct_c = (mse_improvement_c / results["classical_mlp"]["metrics"]["MSE"]) * 100

        print("-" * 70)
        print(f"Trained quantum vs random quantum: MSE {'lower' if mse_improvement_q>0 else 'higher'} "
              f"{abs(mse_improvement_q):.2f}, R² {'higher' if r2_improvement_q>0 else 'lower'} "
              f"{abs(r2_improvement_q):.4f}")
        print(f"Hybrid quantum vs pure classical:   MSE {'lower' if mse_improvement_c>0 else 'higher'} "
              f"{abs(mse_improvement_c):.2f} ({abs(mse_pct_c):.2f}%)")

        # Prediction comparison plot
        plot_comparison(
            y_test,
            {
                "HybridQNN (trained quantum)": y_pred_trained,
                "HybridQNN (random quantum)": y_pred_random,
                "Classical MLP": y_pred_classical,
            },
            f"Model prediction comparison (seed={seed})",
            os.path.join(result_dir, "comparison_plot.png")
        )

    return results


# ============================================================
#  Multiple runs + statistics
# ============================================================

def run_multi_seed(feature_cols, target_col, df, n_runs, base_dir,
                   run_baselines=True):
    """Run multiple times (different seeds), report mean ± std."""
    all_results = []
    seeds = list(range(42, 42 + n_runs))

    for i, seed in enumerate(seeds):
        print(f"\n{'#'*70}")
        print(f"  Run {i+1}/{n_runs} — seed: {seed}")
        print(f"{'#'*70}")

        result_dir = os.path.join(base_dir, f"run_{i+1}_seed_{seed}")
        results = run_single_experiment(
            feature_cols, target_col, df, seed, result_dir,
            run_baselines=run_baselines, verbose=(n_runs <= 2)
        )
        all_results.append(results)

    # --- Aggregate statistics ---
    print(f"\n{'#'*70}")
    print(f"  Multi-run statistics summary ({n_runs} runs)")
    print(f"{'#'*70}")

    model_names = ["hybrid_trained", "hybrid_random", "classical_mlp"]

    for name in model_names:
        mses = [r[name]["metrics"]["MSE"] for r in all_results]
        r2s = [r[name]["metrics"]["R2"] for r in all_results]
        mapes = [r[name]["metrics"]["MAPE"] for r in all_results]

        mean_mse = np.mean(mses)
        std_mse = np.std(mses, ddof=1)
        mean_r2 = np.mean(r2s)
        std_r2 = np.std(r2s, ddof=1)
        mean_mape = np.mean(mapes)
        std_mape = np.std(mapes, ddof=1)

        print(f"\n{name}:")
        print(f"  MSE:  {mean_mse:.2f} ± {std_mse:.2f}")
        print(f"  R²:   {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"  MAPE: {mean_mape:.2f}% ± {std_mape:.2f}%")

    # t-test: hybrid quantum vs pure classical
    trained_mses = [r["hybrid_trained"]["metrics"]["MSE"] for r in all_results]
    classical_mses = [r["classical_mlp"]["metrics"]["MSE"] for r in all_results]

    t_stat, p_value = scipy_stats.ttest_rel(classical_mses, trained_mses)
    print(f"\nPaired t-test (hybrid quantum vs pure classical, MSE):")
    print(f"  t = {t_stat:.4f}, p = {p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✓ Difference is statistically significant (p < 0.05)")
    else:
        print(f"  ✗ Difference is not significant (p >= 0.05)")

    # Trained quantum vs random quantum
    random_mses = [r["hybrid_random"]["metrics"]["MSE"] for r in all_results]
    t_stat_q, p_value_q = scipy_stats.ttest_rel(random_mses, trained_mses)
    print(f"\nPaired t-test (trained quantum vs random quantum, MSE):")
    print(f"  t = {t_stat_q:.4f}, p = {p_value_q:.4f}")
    if p_value_q < 0.05:
        print(f"  ✓ Difference is statistically significant (p < 0.05)")
    else:
        print(f"  ✗ Difference is not significant (p >= 0.05)")

    # Save statistics summary
    summary = {
        "n_runs": n_runs,
        "seeds": seeds,
        "models": {},
    }
    for name in model_names:
        mses = [r[name]["metrics"]["MSE"] for r in all_results]
        r2s = [r[name]["metrics"]["R2"] for r in all_results]
        mapes = [r[name]["metrics"]["MAPE"] for r in all_results]
        summary["models"][name] = {
            "MSE_mean": float(np.mean(mses)),
            "MSE_std": float(np.std(mses, ddof=1)),
            "R2_mean": float(np.mean(r2s)),
            "R2_std": float(np.std(r2s, ddof=1)),
            "MAPE_mean": float(np.mean(mapes)),
            "MAPE_std": float(np.std(mapes, ddof=1)),
            "MSE_list": [float(x) for x in mses],
            "R2_list": [float(x) for x in r2s],
        }
    summary["t_test_hybrid_vs_classical"] = {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
    summary["t_test_trained_vs_random_quantum"] = {
        "t_statistic": float(t_stat_q),
        "p_value": float(p_value_q),
        "significant": bool(p_value_q < 0.05),
    }

    with open(os.path.join(base_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nStatistics summary saved to: {os.path.join(base_dir, 'summary.json')}")

    return all_results, summary

#  Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized hybrid quantum-neural network stock prediction")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of training runs (different seeds)")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip ablation baseline comparison")
    parser.add_argument("--no_walk_forward", action="store_true",
                        help="Use a simple 80/20 split instead of walk-forward")
    parser.add_argument("--seed", type=int, default=42,
                        help="Starting random seed")
    parser.add_argument("--result_dir", type=str, default="./results/optimized",
                        help="Directory to save results")
    args = parser.parse_args()

    # ---------- Load data ----------
    CSV_PATH = "./Dataset/DJIA/processed_data.csv"
    print(f"Loading data: {CSV_PATH}")
    df = load_data(CSV_PATH)

    # Feature columns
    feature_cols = [
        "Open", "High", "Low", "Volume", "pos_mean",
        "open_today", "prev_adj_Close", "ma_5", "ma_20", "Close"
    ]
    target_col = "Adj Close"

    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target column: {target_col}")

    # Check for NaN columns (e.g., ma_5, ma_20 have NaN at the start of the data)
    nan_counts = df[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"\nNaN present in features:")
        for col, cnt in nan_counts.items():
            if cnt > 0:
                print(f"  {col}: {cnt} rows")
        print("Will use forward fill to handle NaN...")
        # Time-series-sensitive: forward fill then backward fill
        df[feature_cols] = df[feature_cols].ffill().bfill()

    if df[feature_cols].isna().any().any():
        print("Warning: NaN still present; related rows will be dropped")

    # ---------- Run experiments ----------
    if args.n_runs > 1:
        run_multi_seed(feature_cols, target_col, df, args.n_runs,
                       args.result_dir, run_baselines=not args.skip_baselines)
    else:
        result_dir = args.result_dir
        if args.n_runs == 1:
            result_dir = os.path.join(args.result_dir, f"seed_{args.seed}")
        run_single_experiment(
            feature_cols, target_col, df, args.seed, result_dir,
            run_baselines=not args.skip_baselines,
            use_walk_forward=not args.no_walk_forward,
            verbose=True
        )

    print("\nAll done.")
