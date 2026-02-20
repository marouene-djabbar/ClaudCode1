import pandas as pd
import glob

# --- Config ---
HORIZON = 10  # ticks ahead to predict
DATA_PATH = "C:/Users/marou/Documents/ClaudCode1/parquet_01/*.parquet"
OUTPUT_PATH = "C:/Users/marou/Documents/ClaudCode1/labeled_data.parquet"

# --- Load all files ---
print("Loading data...")
files = sorted(glob.glob(DATA_PATH))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"Loaded {len(df):,} rows from {len(files)} files")

# --- Sort by timestamp ---
df = df.sort_values("event_ts").reset_index(drop=True)

# --- Calculate future mid price change ---
df["future_mid"] = df["mid_price"].shift(-HORIZON)
df["price_change"] = df["future_mid"] - df["mid_price"]

# Drop last HORIZON rows (no future data available)
df = df.dropna(subset=["price_change"]).reset_index(drop=True)

# --- Quantile-based labeling for balanced classes ---
q33 = df["price_change"].quantile(0.333)
q67 = df["price_change"].quantile(0.667)

print(f"\nThresholds — lower: {q33:.6f}, upper: {q67:.6f}")

def assign_label(x):
    if x > q67:
        return 1    # up
    elif x < q33:
        return -1   # down
    else:
        return 0    # flat

df["label"] = df["price_change"].apply(assign_label)

# --- Class distribution ---
counts = df["label"].value_counts().sort_index()
total = len(df)
print("\nClass distribution:")
for label, count in counts.items():
    name = {1: "Up", 0: "Flat", -1: "Down"}[label]
    print(f"  {name:>5} ({label:>2}): {count:>8,}  ({100*count/total:.1f}%)")

# --- Save labeled data ---
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved labeled data to: {OUTPUT_PATH}")
