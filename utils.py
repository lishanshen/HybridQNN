"""
7_utils.py — Common utility module
  Data loading, feature engineering, preprocessing, evaluation metrics, visualization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import matplotlib.pyplot as plt
import os


def load_data(csv_path="./Dataset/DJIA/processed_data.csv"):
    """Load the processed stock data and sort it by date."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

#  Feature engineering: technical indicator computation
def compute_technical_indicators(df):
    """
    Compute new technical indicators from raw data.
    Note: all rolling calculations use .shift(1) to avoid future information leakage.

    Returns:
        df_out: DataFrame containing the new columns
        new_cols: list of new column names
    """
    df_out = df.copy()

    close = df_out["Close"].values.astype(float)
    volume = df_out["Volume"].values.astype(float)

    # --- RSI(14) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # Compute avg_gain / avg_loss using simple moving average (Wilder's RSI)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    period = 14
    for i in range(period, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0  # set initial values to neutral
    # Lag by one day: today we only know yesterday's RSI
    rsi = np.roll(rsi, 1)
    rsi[0] = 50.0
    df_out["rsi_14"] = rsi

    # --- MACD ---
    # EMA(12) - EMA(26)
    ema_12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd_line = ema_12 - ema_26
    macd_line = np.roll(macd_line, 1)
    macd_line[0] = 0.0
    df_out["macd"] = macd_line

    # --- Volatility (20-day rolling standard deviation) ---
    returns = np.diff(close, prepend=close[0]) / (np.abs(close) + 1e-8)
    returns = np.roll(returns, 1)
    returns[0] = 0.0
    vol_20 = pd.Series(returns).rolling(20, min_periods=5).std().values
    vol_20 = np.nan_to_num(vol_20, nan=0.0)
    df_out["volatility_20"] = vol_20

    # --- Volume ratio ---
    vol_ma_20 = pd.Series(volume).rolling(20, min_periods=5).mean().values
    # The most recent available value is yesterday's
    vol_ma_20 = np.roll(vol_ma_20, 1)
    vol_ma_20[0] = vol_ma_20[1] if len(vol_ma_20) > 1 else 1.0
    df_out["vol_ratio"] = np.divide(volume, vol_ma_20, out=np.ones_like(volume), where=vol_ma_20 > 0)

    new_cols = ["rsi_14", "macd", "volatility_20", "vol_ratio"]
    return df_out, new_cols


def compute_sentiment_detailed(csv_path="./Dataset/DJIA/sentiment_features.csv"):
    """
    Extract finer-grained sentiment features from raw sentiment data.
    Return a DataFrame aggregated by date, containing pos_mean, pos_std, pos_trend.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    pos_cols = [f"pos_{i}" for i in range(1, 51)]

    results = []
    for _, row in df.iterrows():
        pos_vals = row[pos_cols].values.astype(float)
        # Filter out NaN
        pos_vals = pos_vals[~np.isnan(pos_vals)]
        if len(pos_vals) == 0:
            pos_vals = np.array([0.5])

        results.append({
            "date": row["date"],
            "pos_mean": float(np.mean(pos_vals)),
            "pos_std": float(np.std(pos_vals)),
        })

    sent_df = pd.DataFrame(results)
    sent_df = sent_df.sort_values("date").reset_index(drop=True)

    # pos_trend: direction of change of the 3-day moving average
    sent_df["pos_mean_ma3"] = sent_df["pos_mean"].rolling(3, min_periods=1).mean()
    sent_df["pos_trend"] = sent_df["pos_mean_ma3"].diff().fillna(0.0)
    sent_df = sent_df.drop(columns=["pos_mean_ma3"])

    return sent_df

#  Feature preparation

def prepare_features(df, feature_cols, target_col="Adj Close"):
    """
    Extract feature matrix X and target vector y from the DataFrame.
    Target alignment: use today's features to predict tomorrow's Adj Close.
    """
    X_raw = df[feature_cols].values.astype(float)
    y_raw = df[target_col].values.astype(float)

    # Align target to the next day
    y = y_raw[1:]
    X_raw = X_raw[:-1]

    return X_raw, y


def clean_data(X, y):
    """Remove rows containing NaN or inf."""
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    removed = (~mask).sum()
    if removed > 0:
        print(f"Detected {removed} rows containing NaN or inf; removed them.")
    else:
        print("No NaN or inf found in the data.")
    return X[mask], y[mask]

#  Data splitting
def temporal_split(X, y, train_ratio=0.8):
    """Simple temporal split (train/test)."""
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def walk_forward_splits(X, y, n_splits=3, train_init_ratio=0.6, val_ratio=0.15):
    """
    Walk-forward validation split generator.

    Each iteration returns (X_train, X_val, X_test, y_train, y_val, y_test)
    The training window keeps expanding, while the test window slides forward.

    Args:
        n_splits: number of sliding windows
        train_init_ratio: initial training set ratio
        val_ratio: validation set ratio
    """
    n = len(X)
    for i in range(n_splits):
        train_end = int(n * (train_init_ratio + i * 0.1))
        val_end = train_end + int(n * val_ratio)
        if val_end > n:
            val_end = n

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]

        # The last window's test set = all remaining data
        if i == n_splits - 1:
            X_test = X[val_end:]
            y_test = y[val_end:]
        else:
            test_end = val_end + int(n * 0.1)
            if test_end > n:
                test_end = n
            X_test = X[val_end:test_end]
            y_test = y[val_end:test_end]

        yield X_train, X_val, X_test, y_train, y_val, y_test

#  Standardization
class DataScaler:
    """Standardize features and target; fit uses the training set only to avoid data leakage."""

    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.y_mean = None
        self.y_std = None

    def fit(self, X_train, y_train):
        self.feature_scaler.fit(X_train)
        self.y_mean = y_train.mean()
        self.y_std = y_train.std()
        return self

    def transform_X(self, X):
        return self.feature_scaler.transform(X)

    def fit_transform_X(self, X):
        return self.feature_scaler.fit_transform(X)

    def normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, y_norm):
        return y_norm * self.y_std + self.y_mean


#  Quantum encoding preprocessing
def preprocess_features_for_angle(X, n_qubits):
    """
    Scale classical feature vectors to the [-π, π] range for angle encoding.
    - num features < n_qubits: zero-pad
    - num features > n_qubits: truncate
    """
    n_features = X.shape[1]
    if n_features < n_qubits:
        pad = np.zeros((X.shape[0], n_qubits - n_features))
        X = np.hstack([X, pad])
    else:
        X = X[:, :n_qubits]

    X_scaled = np.zeros_like(X)
    for i in range(X.shape[0]):
        row = X[i]
        max_abs = np.max(np.abs(row))
        if max_abs == 0:
            max_abs = 1.0
        X_scaled[i] = np.pi * row / max_abs
    return X_scaled


#  Evaluation metrics

def evaluate_model(y_true, y_pred):
    """Compute regression evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    # MAPE guards against values close to zero
    mask = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        "MSE": float(mse),
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


def print_metrics(metrics, title="Evaluation Results"):
    """Format and print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  MSE : {metrics['MSE']:.4f}")
    print(f"  R²  : {metrics['R2']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")


#  Visualization
def plot_predictions(y_true, y_pred, title, save_path, metrics=None):
    """Plot true vs. predicted line chart."""
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_true)), y_true, label="True (Adj Close)",
             color="black", linewidth=1.5, alpha=0.8)
    plt.plot(range(len(y_pred)), y_pred, label="Predicted",
             color="#2196F3", linewidth=1, alpha=0.8)

    title_str = title
    if metrics:
        title_str += f"  (MSE={metrics['MSE']:.4f}, R²={metrics['R2']:.4f})"
    plt.xlabel("Test Sample Index (chronological order)")
    plt.ylabel("Adj Close")
    plt.title(title_str)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Prediction plot saved to: {save_path}")


def plot_training_curve(train_losses, val_losses, title, save_path):
    """Plot training curves."""
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (Normalized Target)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curve saved to: {save_path}")


def plot_comparison(y_true, predictions_dict, title, save_path):
    """
    Plot a multi-model prediction comparison chart.
    predictions_dict: {model name: y_pred}
    """
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(y_true)), y_true, label="True",
             color="black", linewidth=1.5, alpha=0.8)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]
    for idx, (name, y_pred) in enumerate(predictions_dict.items()):
        color = colors[idx % len(colors)]
        plt.plot(range(len(y_pred)), y_pred, label=name,
                 color=color, linewidth=1, alpha=0.7)

    plt.xlabel("Test Sample Index (chronological order)")
    plt.ylabel("Adj Close")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {save_path}")

#  Device selection
def get_device():
    """Get the available PyTorch device."""
    if torch.cuda.is_available():
        print("Using GPU for training")
        return torch.device("cuda")
    else:
        print(f"GPU unavailable, training on CPU ({os.cpu_count()} threads)")
        return torch.device("cpu")

#  Result saving
def save_model_artifacts(model_state, scaler, y_mean, y_std, result_dir):
    """Save the model, scaler, and target statistics."""
    os.makedirs(result_dir, exist_ok=True)
    torch.save(model_state, os.path.join(result_dir, "model.pth"))
    import joblib
    joblib.dump(scaler, os.path.join(result_dir, "feature_scaler.pkl"))
    np.save(os.path.join(result_dir, "y_mean_std.npy"),
            np.array([y_mean, y_std]))
    print(f"Model artifacts saved to: {result_dir}/")
