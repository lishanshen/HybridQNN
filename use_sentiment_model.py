"""
Sentiment analysis pipeline: load DistilBERT model → infer on RedditNews → aggregate features → CSV output
Outputs:
  1. ./Dataset/DJIA/processed_data.csv    — per-news sentiment score
  2. ./Dataset/DJIA/sentiment_features.csv — date-aggregated feature matrix (up to 50 news items per day)
"""

import csv
import os
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────
MODEL_DIR = os.path.join(".", "sentiment_model")
# Fallback used when the (255 MB) weights are not present locally, e.g. right after a fresh clone.
HF_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
REDDIT_NEWS = os.path.join(".", "Dataset", "DJIA", "RedditNews.csv")
OUTPUT_SCORES = os.path.join(".", "Dataset", "DJIA", "processed_data.csv")
OUTPUT_FEATURES = os.path.join(".", "Dataset", "DJIA", "sentiment_features.csv")

BATCH_SIZE = 32          # Inference batch size
MAX_TOKENS = 512         # BERT maximum input length
POOL_OR_MULTI = "multi"  # "multi": independent feature per news item | "pool": mean/variance aggregation
MAX_NEWS_PER_DAY = 50    # Number of feature matrix columns (aligned with the max daily news count)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────


def load_model_and_tokenizer(model_dir: str):
    """Load the locally saved DistilBERT model and tokenizer.

    The binary weights (``model.safetensors``, ~255 MB) are **not** tracked by Git.
    If they are missing, the model is fetched from the Hugging Face Hub instead.
    """
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    print(f"[1/4] Loading model and tokenizer from: {model_dir}")
    print(f"     Device: {DEVICE.upper()}")

    local_weights = os.path.exists(os.path.join(model_dir, "model.safetensors")) or os.path.exists(
        os.path.join(model_dir, "pytorch_model.bin")
    )
    source = model_dir if local_weights else HF_MODEL_ID
    if not local_weights:
        print(f"     Local weights not found -> downloading '{HF_MODEL_ID}' from the Hugging Face Hub")

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = DistilBertForSequenceClassification.from_pretrained(source)
    model.to(DEVICE)
    model.eval()

    print(f"     Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"     Label mapping:    {model.config.id2label}")
    return model, tokenizer


def read_reddit_news(path: str) -> pd.DataFrame:
    """Read RedditNews.csv and return a DataFrame[date, news]."""
    print(f"[2/4] Reading news data: {path}")
    df = pd.read_csv(path, encoding="utf-8", quotechar='"', escapechar="\\")
    df.columns = ["date", "news"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"     Total news items: {len(df):,}")
    print(f"     Date range:       {df['date'].min()} ~ {df['date'].max()}")
    print(f"     News per day:     {df.groupby('date').size().describe()}")
    return df


@torch.no_grad()
def infer_sentiment(model, tokenizer, texts: list[str]) -> np.ndarray:
    """
    Batch inference, returning the POSITIVE probability for each text (shape: [n,])
    DistilBERT output logits: [NEGATIVE, POSITIVE]
    """
    model.eval()
    all_probs = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        logits = model(**encoded).logits            # [B, 2]
        probs = softmax(logits, dim=-1)[:, 1]        # POSITIVE probability
        all_probs.append(probs.cpu().numpy())

        if (i // BATCH_SIZE) % 50 == 0 and i > 0:
            print(f"     Inference progress: {i}/{len(texts)}")

    return np.concatenate(all_probs)


def build_features_multi(df_scores: pd.DataFrame, max_cols: int = 50) -> pd.DataFrame:
    """
    Treat each news item as an independent feature column.
    """
    print(f"[4/4] Building multi-feature matrix (one column per news item x{max_cols})")

    def expand(row):
        scores = row
        if len(scores) >= max_cols:
            scores = scores[:max_cols]
        else:
            scores = scores + [np.nan] * (max_cols - len(scores))
        return scores

    expanded = df_scores.groupby("date")["prob"].apply(list).reset_index()
    expanded[[f"pos_{i+1}" for i in range(max_cols)]] = (
        expanded["prob"].apply(expand).tolist()
    )
    expanded.drop(columns=["prob"], inplace=True)
    expanded.sort_values("date", inplace=True)
    expanded.reset_index(drop=True, inplace=True)
    return expanded


def main():
    t_start = time.time()

    # 1. Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR)

    # 2. Read data
    df_news = read_reddit_news(REDDIT_NEWS)
    texts = df_news["news"].tolist()

    # 3. Sentiment inference
    print(f"[3/4] Inferring sentiment scores ({len(texts)} items, batch_size={BATCH_SIZE})")
    probs = infer_sentiment(model, tokenizer, texts)
    df_news["prob"] = probs

    # Save per-item scores (for inspection)
    df_news.to_csv(OUTPUT_SCORES, index=False)
    print(f"     Per-item scores saved: {OUTPUT_SCORES}")

    # 4. Feature aggregation → CSV
    df_features = build_features_multi(df_news, max_cols=MAX_NEWS_PER_DAY)

    # Merge stock price labels (optional, to prepare for downstream DL)
    price_path = os.path.join(".", "Dataset", "DJIA", "upload_DJIA_table.csv")
    if os.path.exists(price_path):
        df_price = pd.read_csv(price_path)
        df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.date
        df_price.rename(columns={"Date": "date"}, inplace=True)
        df_features = df_features.merge(df_price, on="date", how="left")
        print(f"     Merged stock price data: {price_path}")

    df_features.to_csv(OUTPUT_FEATURES, index=False)
    print(f"     Feature matrix saved:    {OUTPUT_FEATURES}")
    print(f"     Matrix shape:            {df_features.shape}")

    elapsed = time.time() - t_start
    print(f"\n✅ All done! Elapsed {elapsed / 60:.1f} minutes")
    print(f"   → {OUTPUT_SCORES}")
    print(f"   → {OUTPUT_FEATURES}")


if __name__ == "__main__":
    main()
