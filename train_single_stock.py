"""
Usage:
  python train_single_stock.py --stock AAWW
  python train_single_stock.py --stock ACRX --skip_baselines
  python train_single_stock.py --stock AAWW --seed 43
"""

import os
import sys
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

from torch.optim.lr_scheduler import ReduceLROnPlateau
from pennylane.templates import AngleEmbedding
from datetime import datetime
from utils import (
    prepare_features, clean_data, temporal_split,
    DataScaler, preprocess_features_for_angle,
    evaluate_model, print_metrics,
    plot_predictions, plot_training_curve,
    get_device,
)

class TeeLogger:
    """Write to both terminal stdout and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


#  Quantum circuit components
N_LAYERS = 4
ENTANGLING = "circular"


def get_quantum_device(n_qubits):
    if torch.cuda.is_available():
        try:
            return qml.device("lightning.gpu", wires=n_qubits)
        except Exception:
            print("Cannot use lightning.gpu, falling back to lightning.qubit")
    return qml.device("lightning.qubit", wires=n_qubits)


def encode_features_layer(features, wires, rotation="X"):
    AngleEmbedding(features, wires=wires, rotation=rotation)


def variational_layer(weights, wires, entangling_pattern="circular"):
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
    for layer in range(n_layers):
        encode_features_layer(features, wires=wires, rotation="X")
        variational_layer(weights[layer], wires=wires, entangling_pattern=entangling_pattern)


#  HybridQNN model
class HybridQNN(nn.Module):
    def __init__(self, n_qubits, n_classical, n_layers, dropout=0.2):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )
        dev = get_quantum_device(n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def _circuit(params, x):
            wires = range(n_qubits)
            data_reuploading_block(params, x, wires=wires, n_layers=n_layers, entangling_pattern=ENTANGLING)
            z_exps = [qml.expval(qml.Z(i)) for i in range(n_qubits)]
            zz_exps = [qml.expval(qml.Z(i) @ qml.Z(i + 1)) for i in range(n_qubits - 1)]
            return z_exps + zz_exps

        self._circuit = _circuit

        q_output_dim = n_qubits * 2 - 1  # n Z + (n-1) ZZ
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
        self.post_net = self.post_net.to(*args, **kwargs)
        return self

    def forward(self, x_quantum, x_classical):
        if x_quantum.device.type != 'cpu':
            x_quantum = x_quantum.cpu()
        q_raw = self._circuit(self.quantum_params, x_quantum)
        q_feat = torch.stack(q_raw, dim=1)

        if q_feat.device != x_classical.device:
            q_feat = q_feat.to(x_classical.device)

        combined = torch.cat([q_feat, x_classical], dim=1)
        return self.post_net(combined)

#  ClassicalMLP
class ClassicalMLP(nn.Module):
    def __init__(self, n_features, dropout=0.2):
        super().__init__()
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
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds_norm = model(X_test_q.to(device), X_test_c.to(device)).cpu().numpy().flatten()
    y_pred = scaler.denormalize_y(preds_norm)
    return evaluate_model(y_test, y_pred), y_pred


def evaluate_classical(model, X_test, y_test, scaler, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds_norm = model(X_test.to(device)).cpu().numpy().flatten()
    y_pred = scaler.denormalize_y(preds_norm)
    return evaluate_model(y_test, y_pred), y_pred


def load_stock_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)

    # Standardize column names
    rename = {"open": "Open", "high": "High", "low": "Low",
              "close": "Close", "volume": "Volume", "adj close": "Adj Close"}
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    # Compute price-derived features
    df["prev_adj_Close"] = df["Adj Close"].shift(1)
    df["open_today"] = df["Open"]
    df["ma_5"] = df["prev_adj_Close"].rolling(5).mean()
    df["ma_20"] = df["prev_adj_Close"].rolling(20).mean()

    # Remove leading rows containing NaN produced by rolling windows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Stock price data: {len(df)} rows, dates {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Single-stock training (price features only, no sentiment)")
    parser.add_argument("--stock", type=str, default="AADR", help="Stock ticker, corresponding to data/{STOCK}.csv")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip the ClassicalMLP ablation baseline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--result_dir", type=str, default="./results/single_stock", help="Directory to save results")
    args = parser.parse_args()

    # ---------- Fix random seeds (ensure reproducibility across seeds) ----------
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---------- Load data ----------
    csv_path = f"./Dataset/{args.stock}.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    print(f"Loading data: {csv_path}")
    df = load_stock_data(csv_path)

    # ---------- Feature columns (price only, no sentiment) ----------
    feature_cols = [
        "Open", "High", "Low", "Volume", "open_today",
        "prev_adj_Close", "ma_5", "ma_20", "Close"
    ]
    target_col = "Adj Close"

    n_qubits = len(feature_cols)
    print(f"Feature columns ({n_qubits}): {feature_cols}")
    print(f"Target column: {target_col}")

    # ---------- Prepare features ----------
    X_raw, y = prepare_features(df, feature_cols, target_col)
    X_raw, y = clean_data(X_raw, y)
    print(f"Samples after cleaning: {len(y)}, feature dim: {X_raw.shape[1]}")

    # ---------- Data split ----------
    X_train_raw, X_test_raw, y_train, y_test = temporal_split(X_raw, y, train_ratio=0.8)
    val_split = int(len(X_train_raw) * 0.85)
    X_val_raw, y_val = X_train_raw[val_split:], y_train[val_split:]
    X_train_raw, y_train = X_train_raw[:val_split], y_train[:val_split]
    print(f"Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

    # ---------- Standardization ----------
    scaler = DataScaler()
    scaler.fit(X_train_raw, y_train)
    X_train_scaled = scaler.transform_X(X_train_raw)
    X_val_scaled = scaler.transform_X(X_val_raw)
    X_test_scaled = scaler.transform_X(X_test_raw)

    y_train_norm = scaler.normalize_y(y_train)
    y_val_norm = scaler.normalize_y(y_val)
    y_test_norm = scaler.normalize_y(y_test)

    # ---------- Quantum encoding ----------
    X_train_q = preprocess_features_for_angle(X_train_scaled, n_qubits)
    X_val_q = preprocess_features_for_angle(X_val_scaled, n_qubits)
    X_test_q = preprocess_features_for_angle(X_test_scaled, n_qubits)

    # ---------- Tensors ----------
    X_train_q_t = torch.tensor(X_train_q, dtype=torch.float32)
    X_val_q_t = torch.tensor(X_val_q, dtype=torch.float32)
    X_test_q_t = torch.tensor(X_test_q, dtype=torch.float32)

    X_train_c_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_c_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_c_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_t = torch.tensor(y_train_norm, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val_norm, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test_norm, dtype=torch.float32).unsqueeze(1)

    device = get_device()

    # ---------- Result directory ----------
    result_dir = os.path.join(args.result_dir, args.stock, f"seed_{args.seed}")
    os.makedirs(result_dir, exist_ok=True)

    # Redirect logs to a file
    log_path = os.path.join(result_dir,
        f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = TeeLogger(log_path)

    print(f"Log saved to: {log_path}")

    # ========== HybridQNN (train quantum parameters) ==========
    print(f"\n{'='*60}")
    print(f"  Main model: HybridQNN ({n_qubits} qubits, {N_LAYERS} layers, circular entanglement)")
    print(f"{'='*60}")

    model = HybridQNN(n_qubits, len(feature_cols), N_LAYERS, dropout=0.2)

    t_start = time.time()
    best_state, train_losses, val_losses, _ = train_model(
        model, X_train_q_t, X_train_c_t, y_train_t,
        X_val_q_t, X_val_c_t, y_val_t,
        device, lr=0.01, weight_decay=1e-5
    )
    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    metrics_hybrid, y_pred_hybrid = evaluate_hybrid(
        model, X_test_q_t, X_test_c_t, y_test, scaler, device
    )
    print_metrics(metrics_hybrid, f"HybridQNN Test Set ({args.stock})")
    print(f"Training time: {train_time:.1f}s")

    # Save
    torch.save(best_state, os.path.join(result_dir, "model.pth"))
    import joblib
    joblib.dump(scaler.feature_scaler, os.path.join(result_dir, "feature_scaler.pkl"))
    np.save(os.path.join(result_dir, "y_mean_std.npy"),
            np.array([scaler.y_mean, scaler.y_std]))

    plot_predictions(y_test, y_pred_hybrid,
                     f"HybridQNN — {args.stock}",
                     os.path.join(result_dir, "prediction_plot.png"),
                     metrics_hybrid)
    plot_training_curve(train_losses, val_losses,
                        f"HybridQNN training curves — {args.stock}",
                        os.path.join(result_dir, "training_curve.png"))

    if args.skip_baselines:
        print("\nDone.")
        return

    # ========== ClassicalMLP (ablation baseline) ==========
    print(f"\n{'='*60}")
    print(f"  Baseline: pure classical MLP (matched parameter count)")
    print(f"{'='*60}")

    model_classical = ClassicalMLP(len(feature_cols), dropout=0.2)
    best_state_c, train_losses_c, val_losses_c, _ = train_classical_only(
        model_classical, X_train_c_t, y_train_t, X_val_c_t, y_val_t,
        device, lr=0.01, weight_decay=1e-5
    )
    model_classical.load_state_dict(best_state_c)
    metrics_classical, y_pred_classical = evaluate_classical(
        model_classical, X_test_c_t, y_test, scaler, device
    )
    print_metrics(metrics_classical, f"ClassicalMLP Test Set ({args.stock})")

    # ---------- Comparison summary ----------
    print(f"\n{'='*70}")
    print(f"  Model comparison — {args.stock}")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'MSE':>12} {'R²':>10} {'RMSE':>10} {'MAPE':>8}")
    print("-" * 70)
    for name, m in [("HybridQNN", metrics_hybrid), ("ClassicalMLP", metrics_classical)]:
        print(f"{name:<25} {m['MSE']:>12.2f} {m['R2']:>10.4f} {m['RMSE']:>10.2f} {m['MAPE']:>7.2f}%")

    mse_diff = metrics_classical["MSE"] - metrics_hybrid["MSE"]
    mse_pct = (mse_diff / metrics_classical["MSE"]) * 100
    print("-" * 70)
    direction = "lower" if mse_diff > 0 else "higher"
    print(f"HybridQNN vs ClassicalMLP: MSE {direction} by {abs(mse_diff):.2f} ({abs(mse_pct):.2f}%)")

    # Save metrics
    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump({"hybrid": metrics_hybrid, "classical": metrics_classical}, f, indent=2)

    # Comparison plot
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(y_test)), y_test, label="True", color="black", linewidth=1.5, alpha=0.8)
    plt.plot(range(len(y_pred_hybrid)), y_pred_hybrid, label="HybridQNN", color="#2196F3", linewidth=1)
    plt.plot(range(len(y_pred_classical)), y_pred_classical, label="ClassicalMLP", color="#FF5722", linewidth=1)
    plt.xlabel("Test Sample Index (chronological order)")
    plt.ylabel("Adj Close")
    plt.title(f"Model Prediction Comparison — {args.stock}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "comparison_plot.png"), dpi=150)
    plt.close()

    print(f"\nResults saved to: {result_dir}/")
    print("All done.")


if __name__ == "__main__":
    main()
