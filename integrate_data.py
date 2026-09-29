import pandas as pd

# 1. Read the data
df = pd.read_csv('./Dataset/DJIA/sentiment_features.csv', parse_dates=['date'])

# 2. Drop dates with no trading session
price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
for col in price_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing prices
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
df = df[df['Volume'] != 0]

# Compute the mean news sentiment, pos_mean
pos_cols = [f'pos_{i}' for i in range(1, 51)]
# Convert the sentiment columns to numeric
df[pos_cols] = df[pos_cols].apply(pd.to_numeric, errors='coerce')
df['pos_mean'] = df[pos_cols].mean(axis=1, skipna=True)

# Sort by date
df = df.sort_values('date').reset_index(drop=True)

# Build the previous-day adjusted close and its moving averages
df['prev_adj_Close'] = df['Adj Close'].shift(1)
df['ma_5'] = df['prev_adj_Close'].rolling(window=5, min_periods=5).mean()
df['ma_20'] = df['prev_adj_Close'].rolling(window=20, min_periods=20).mean()

# Assemble the output columns
output_cols = [
    'date', 'Open', 'High', 'Low', 'Volume',
    'pos_mean', 'open_today', 'prev_adj_Close',
    'ma_5', 'ma_20', 'Close', 'Adj Close'
]
df_out = df.copy()
df_out['open_today'] = df_out['Open']
df_out = df_out[output_cols]

#Save as a new CSV
OUTPUT_PATH = './Dataset/DJIA/processed_data.csv'
df_out.to_csv(OUTPUT_PATH, index=False, date_format='%Y-%m-%d')
print(f"Processing complete; results saved as {OUTPUT_PATH}")
