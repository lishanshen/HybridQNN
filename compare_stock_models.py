"""Compare four existing DJIA models on identical next-day target dates.

Run with the training environment:
    D:/anaconda24/envs/pennylane_gpu/python.exe compare_stock_models.py
    python compare_stock_models.py --points 100 --start-date 2015-01-01

All labels and output messages are English. No model is retrained.
The historical GRU used full-data scaling and test-set early stopping;
this reproduces its checkpoint, not a leakage-free benchmark.
QLSTM dimensions are inferred from its checkpoint. Sequence length, feature
order, and normalization split are read from the adjacent result.txt.
Legacy QLSTM checkpoints without quantum weights require seeded recovery
and agreement with archived predictions. Complete checkpoints load strictly.
"""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parent
FEATURES = ["Open", "High", "Low", "Volume", "pos_mean", "open_today",
            "prev_adj_Close", "ma_5", "ma_20", "Close"]
TARGET = "Adj Close"


def read_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    if df.date.isna().any() or df.date.duplicated().any():
        raise ValueError(f"Invalid or duplicate dates: {path}")
    if not df.date.is_monotonic_increasing:
        raise ValueError(f"Data must be in chronological order: {path}")
    df[FEATURES + [TARGET]] = df[FEATURES + [TARGET]].apply(pd.to_numeric)
    return df


def state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def angles(x):
    return np.pi * x / np.maximum(np.abs(x).max(axis=1, keepdims=True), 1e-30)


def quantum_features(x, weights, entangling, reupload=False, rotation="X"):
    n_layers, n_qubits, _ = weights.shape
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs):
        if not reupload:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation)
        for layer in weights:
            if reupload:
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation)
            for i in range(n_qubits):
                qml.Rot(*layer[i], wires=i)
            if entangling == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
            elif entangling in ("linear", "circular"):
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if entangling == "circular":
                    qml.CNOT(wires=[n_qubits - 1, 0])
            else:
                raise ValueError(f"Unsupported entangling pattern: {entangling}")
        out = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        if reupload:
            out += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1))
                    for i in range(n_qubits - 1)]
        return out

    return np.concatenate([np.asarray(circuit(x[i:i + 32])).T
                           for i in range(0, len(x), 32)])


def next_day_frame(df):
    frame = df.iloc[:-1].copy()
    frame["target_date"] = df.date.iloc[1:].to_numpy()
    frame["actual"] = df[TARGET].iloc[1:].to_numpy()
    return frame


class GRUForecaster(nn.Module):
    def __init__(self, weights):
        super().__init__()
        hidden = weights["gru.weight_hh_l0"].shape[1]
        layers = sum(k.startswith("gru.weight_hh_l") for k in weights)
        self.gru = nn.GRU(len(FEATURES), hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        self.load_state_dict(weights, strict=True)

    def forward(self, x):
        return self.fc(self.gru(x)[0][:, -1])


def prepare(args):
    gru = read_data(args.gru_data)
    hybrid = read_data(args.hybrid_data)
    vqc = read_data(args.vqc_data)
    qlstm = read_data(args.qlstm_data)
    # Preserve each original training preprocessing recipe.
    gru[FEATURES + [TARGET]] = gru[FEATURES + [TARGET]].ffill().bfill()
    hybrid[FEATURES] = hybrid[FEATURES].ffill().bfill()
    h = next_day_frame(hybrid)
    h = h[np.isfinite(h[FEATURES + ["actual"]]).all(axis=1)].reset_index(drop=True)
    v = next_day_frame(vqc)
    v = v[np.isfinite(v[FEATURES + ["actual"]]).all(axis=1)].reset_index(drop=True)
    report = read_qlstm_report(args.qlstm_model)
    qcols = ast.literal_eval(report["Features"])
    if args.qlstm_preprocessing == "v3":
        qlstm[qcols] = qlstm[qcols].ffill().bfill()
    else:
        qlstm = qlstm.dropna(subset=qcols).reset_index(drop=True)
    q = next_day_frame(qlstm).dropna(subset=["actual"]).reset_index(drop=True)
    gs = 5 + int((len(gru) - 5) * .8)
    vs, qs = int(len(v) * .8), int(len(q) * .8)
    # New runs use an 80% boundary; legacy optimized runs used 394 dates.
    hybrid_start = (int(len(h) * .8) if args.hybrid_model.parent.parent.name == "qxg"
                    else max(0, len(h) - 394))
    test_dates = [pd.Index(gru.date.iloc[gs:]), pd.Index(h.target_date.iloc[hybrid_start:]),
                  pd.Index(v.target_date.iloc[vs:]), pd.Index(q.target_date.iloc[qs:])]
    dates = test_dates[0]
    for other in test_dates[1:]:
        dates = dates.intersection(other)
    dates = dates.sort_values()
    if args.start_date:
        dates = dates[dates >= pd.Timestamp(args.start_date)]
    if args.end_date:
        dates = dates[dates <= pd.Timestamp(args.end_date)]
    if args.points:
        dates = dates[:args.points]
    if len(dates) < 2:
        raise ValueError("At least two common test dates are required.")
    actual = gru.set_index("date").loc[dates, TARGET].to_numpy()
    for name, frame in [("HybridQNN", h), ("VQC+XGBoost", v), ("QLSTM", q)]:
        other = frame.set_index("target_date").loc[dates, "actual"].to_numpy()
        if not np.allclose(actual, other, rtol=0, atol=1e-4):
            raise ValueError(f"Actual prices do not match for {name} on common dates.")
    return gru, h, v, q, qs, dates, actual


def predict_gru(args, df, dates):
    x = StandardScaler().fit_transform(df[FEATURES].to_numpy())
    sy = StandardScaler().fit(df[[TARGET]].to_numpy())
    indices = df.index[df.date.isin(dates)]
    windows = np.stack([x[i - 5:i] for i in indices])
    model = GRUForecaster(state(args.gru_model)).eval()
    with torch.no_grad():
        pred = model(torch.tensor(windows, dtype=torch.float32)).numpy()
    return sy.inverse_transform(pred).ravel()


def predict_hybrid(args, frame, dates):
    folder = args.hybrid_model.parent
    weights = state(args.hybrid_model)
    x = frame.set_index("target_date").loc[dates, FEATURES].to_numpy()
    x = joblib.load(folder / "feature_scaler.pkl").transform(x)
    params = weights["quantum_params"].numpy()
    q = quantum_features(angles(x), params, "circular", reupload=True)
    net = nn.Sequential(nn.Linear(q.shape[1] + x.shape[1], 64), nn.ReLU(),
                        nn.Dropout(.2), nn.Linear(64, 32), nn.ReLU(),
                        nn.Dropout(.2), nn.Linear(32, 1))
    net.load_state_dict({k.removeprefix("post_net."): val for k, val in weights.items()
                         if k.startswith("post_net.")}, strict=True)
    net.eval()
    mean, std = np.load(folder / "y_mean_std.npy")
    with torch.no_grad():
        pred = net(torch.tensor(np.hstack([q, x]), dtype=torch.float32)).numpy().ravel()
    return pred * std + mean


def predict_vqc(args, frame, dates):
    folder = args.vqc_model.parent
    config = joblib.load(folder / "vqc_config_optimized.pkl")
    if config["encoding"] != "angle":
        raise ValueError("Only the original angle encoding is supported.")
    x = frame.set_index("target_date").loc[dates, FEATURES].to_numpy()
    x = joblib.load(folder / "feature_scaler_optimized.pkl").transform(x)
    params = np.load(folder / "vqc_params_optimized.npy")
    if params.shape != (config["n_layers"], config["n_qubits"], 3):
        raise ValueError("VQC parameter dimensions do not match the saved configuration.")
    q = quantum_features(angles(x), params, config["entangling"],
                         rotation=config["rotation"])
    return joblib.load(args.vqc_model).predict(np.hstack([q, x]))


def read_qlstm_report(model_path):
    report_path = model_path.parent / "result.txt"
    if not report_path.is_file():
        raise ValueError(f"QLSTM training configuration is required: {report_path}")
    return dict(line.split(": ", 1) for line in report_path.read_text(
        encoding="utf-8").splitlines() if ": " in line)


def predict_qlstm(args, frame, split, dates):
    spec = importlib.util.spec_from_file_location("comparison_qlstm", args.qlstm_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Original code seeds before constructing the model. The unregistered VQC
    # parameters were never optimized, so the original seed can recover them.
    weights = state(args.qlstm_model)
    report = read_qlstm_report(args.qlstm_model)
    cols = ast.literal_eval(report["Features"])
    sequence_length = int(report["Sequence length"])
    val_ratio = float(report.get("Val ratio (of train split)", 0))
    if sequence_length < 1 or not 0 <= val_ratio < 1:
        raise ValueError("Invalid QLSTM sequence length or validation ratio.")
    hidden = weights["linear.weight"].shape[1]
    qubits, combined = weights["lstm.clayer_in.weight"].shape
    if combined - hidden != len(cols):
        raise ValueError("QLSTM feature count does not match the checkpoint.")
    quantum_keys = [f"lstm.vqc_{gate}.weights" for gate in
                    ("forget", "input", "update", "output")]
    has_quantum = all(k in weights for k in quantum_keys)
    if any(k in weights for k in quantum_keys) and not has_quantum:
        raise ValueError("QLSTM checkpoint contains only some quantum gate weights.")
    layers = weights[quantum_keys[0]].shape[0] if has_quantum else int(report["QLayers"])
    torch.manual_seed(int(report.get("Seed", 101)))
    model = module.QShallowRegressionLSTM(len(cols), hidden, n_qubits=qubits, n_qlayers=layers)
    if has_quantum:
        model.load_state_dict(weights, strict=True)
    else:
        # Older checkpoints omitted frozen gates; current source registers them.
        incompatible = model.load_state_dict(weights, strict=False)
        if set(incompatible.missing_keys) - set(quantum_keys) or incompatible.unexpected_keys:
            raise ValueError(f"Incompatible legacy QLSTM checkpoint: {incompatible}")
    model.eval()
    is_v3 = args.qlstm_preprocessing == "v3"
    train_end = int(len(frame) * .60) if is_v3 else int(split * (1 - val_ratio))
    ddof = 0 if is_v3 else 1
    train, test = frame.iloc[:train_end], frame.iloc[split:]
    print(f"QLSTM configuration: {qubits} qubits, {layers} layers, "
          f"{sequence_length} time steps; scaler fitted on {train_end} training rows.", flush=True)
    std = train[cols].std(ddof=ddof)
    if (std == 0).any():
        raise ValueError("QLSTM training features have zero standard deviation.")
    x = ((test[cols] - train[cols].mean()) / std).to_numpy()
    mean, target_std = train.actual.mean(), train.actual.std(ddof=ddof)

    def one(i):
        indices = np.maximum(np.arange(i - sequence_length + 1, i + 1), 0)
        with torch.no_grad():
            value = model(torch.tensor(x[indices][None], dtype=torch.float32)).item()
        return value * target_std + mean

    cache = {}
    archive_path = args.qlstm_model.parent / "predictions.csv"
    if not has_quantum or archive_path.exists():
        if not archive_path.exists():
            raise ValueError("QLSTM quantum weights are missing. The original predictions.csv "
                             "is required to verify seeded reconstruction.")
        archive = pd.read_csv(archive_path, parse_dates=["date"]).set_index("date")
        # The archive labels rows by INPUT date, not next-day TARGET date.
        for i in sorted(set([0, len(test) // 2, len(test) - 1])):
            row = test.iloc[i]
            recorded = archive.loc[row.date]
            if not np.isclose(recorded.Actual, row.actual, rtol=0, atol=1e-4):
                raise ValueError("QLSTM archive target alignment does not match training data.")
            cache[i] = one(i)
            if not np.isclose(cache[i], recorded.Pred, rtol=0, atol=.1):
                raise ValueError("QLSTM predictions do not match the training archive. "
                                 "Check the model source, training data, and result.txt configuration.")
        print("QLSTM inference verified against three archived predictions.", flush=True)
    positions = pd.Index(test.target_date).get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("Requested target dates are missing from the QLSTM test set.")
    result = []
    for count, i in enumerate(positions):
        result.append(cache[i] if i in cache else one(i))
        if (count + 1) % 25 == 0:
            print(f"QLSTM: {count + 1}/{len(positions)} predictions completed.", flush=True)
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = {
        "gru-model": ROOT / "CRU/results/cru/run_seed_42/best_gru_model.pth",
        "hybrid-model": ROOT / "QXGBoot 1/results/qxg/run_3_seed_44/model.pth",
        "vqc-model": Path("D:/Research2/QXGBoot/results/run_20260926_234314/seed_2024/xgboost_regressor_optimized.pkl"),
        "qlstm-model": Path("D:/Research2/SPP_QLSTM/I2/results/robustness_v3/run_seed303/model.pt"),
        "gru-data": ROOT / "CRU/data/processed_data.csv",
        "hybrid-data": ROOT / "QXGBoot 1/data/processed_data.csv",
        "vqc-data": Path("D:/Research2/QXGBoot/data/processed_data.csv"),
        "qlstm-data": Path("D:/Research2/SPP_QLSTM/I2/data/processed_data.csv"),
        "qlstm-source": Path("D:/Research2/SPP_QLSTM/I2/QLSTM.py"),
        "output-dir": ROOT / "output/model_comparison",
    }
    for name, value in defaults.items():
        parser.add_argument("--" + name, type=Path, default=value)
    parser.add_argument("--start-date", help="First target date, YYYY-MM-DD")
    parser.add_argument("--qlstm-preprocessing", choices=("auto", "legacy", "v2", "v3"),
                        default="auto", help="Training preprocessing recipe; auto recognizes original run folders")
    parser.add_argument("--end-date", help="Last target date, YYYY-MM-DD")
    parser.add_argument("--points", type=int, default=0,
                        help="First N common test dates; 0 uses the entire common test period")
    args = parser.parse_args()
    if args.qlstm_preprocessing == "auto":
        if "robustness_v3" in args.qlstm_model.parts:
            args.qlstm_preprocessing = "v3"
        elif "robustness_v2" in args.qlstm_model.parts:
            args.qlstm_preprocessing = "v2"
        elif args.qlstm_model.parent.name == "DJIA":
            args.qlstm_preprocessing = "legacy"
        else:
            parser.error("Unknown QLSTM training recipe. Set --qlstm-preprocessing legacy, v2, or v3.")
    if args.points < 0:
        parser.error("--points must be nonnegative")
    for name in defaults:
        if name != "output-dir" and not getattr(args, name.replace("-", "_")).is_file():
            parser.error(f"File not found: {getattr(args, name.replace('-', '_'))}")
    torch.set_num_threads(4)
    g, h, v, q, qs, dates, actual = prepare(args)
    print(f"Common test period: {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d} "
          f"({len(dates)} target dates)", flush=True)
    result = pd.DataFrame({"Date": dates, "Actual": actual})
    predictors = [("GRU", lambda: predict_gru(args, g, dates)),
                  ("HybridQNN", lambda: predict_hybrid(args, h, dates)),
                  ("VQC+XGBoost", lambda: predict_vqc(args, v, dates)),
                  ("QLSTM", lambda: predict_qlstm(args, q, qs, dates))]
    metrics = []
    for name, predict in predictors:
        print(f"Running {name}...", flush=True)
        pred = np.asarray(predict()).reshape(-1)
        if len(pred) != len(actual) or not np.isfinite(pred).all():
            raise ValueError(f"Invalid predictions from {name}")
        result[name] = pred
        err = pred - actual
        metrics.append({"Model": name, "RMSE": float(np.sqrt(np.mean(err ** 2))),
                        "MAE": float(np.mean(np.abs(err)))})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(15, 6), layout="constrained")
    ax.plot(dates, actual, color="black", linewidth=2.2, label="Actual Price", zorder=5)
    for (name, _), color, style in zip(predictors,
                                      ["#0072B2", "#D55E00", "#009E73", "#CC79A7"],
                                      ["--", "-", "-.", ":"]):
        ax.plot(dates, result[name], color=color, linestyle=style,
                linewidth=1.5, alpha=.9, label=name)
    ax.set(title="DJIA: Actual Price and Model Predictions on the Same Test Period",
           xlabel="Target Date", ylabel="Adjusted Closing Price (Index Points)")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(ncol=5, loc="best")
    ax.grid(alpha=.25)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"stock_model_comparison.{extension}", dpi=300)
    plt.close(fig)
    metadata = {"start": str(dates[0].date()), "end": str(dates[-1].date()),
                "samples": len(dates), "paths": {k: str(val) for k, val in vars(args).items()},
                "notes": ["Target dates are aligned across all four original test splits.",
                          "GRU reproduces historical full-data scaling and test selection.",
                          "QLSTM dimensions are loaded from checkpoint; preprocessing follows result.txt.",
                          "This is a checkpoint comparison, not an untouched holdout benchmark."]}
    (args.output_dir / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"Saved figure and predictions to: {args.output_dir}")


if __name__ == "__main__":
    main()
