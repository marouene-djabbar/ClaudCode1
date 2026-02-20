import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def log(msg):
    print(msg, flush=True)

# --- Config ---
THRESHOLD  = 0.5   # 1 tick
HORIZONS   = [50]  # 5 seconds
SAMPLE     = 100_000
DATA_PATH  = "C:/Users/marou/Documents/ClaudCode1/parquet_01/*.parquet"

# --- Feature columns ---
BID_V = [f"bid_v_{i:02d}" for i in range(20)]
ASK_V = [f"ask_v_{i:02d}" for i in range(20)]
BID_P = [f"bid_p_{i:02d}" for i in range(20)]
ASK_P = [f"ask_p_{i:02d}" for i in range(20)]
BASE_FEATURES = ["spread_l1", "vol_imbalance_l1"] + BID_V + ASK_V

# --- Step 1: Load data ---
log("Step 1/6 — Loading data...")
files = sorted(glob.glob(DATA_PATH))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.sort_values("event_ts").reset_index(drop=True)
log(f"  Loaded {len(df):,} rows from {len(files)} files")

# --- Step 2: Sample ---
log(f"\nStep 2/6 — Sampling {SAMPLE:,} rows (chronological)...")
df = df.iloc[:SAMPLE].reset_index(drop=True)
log(f"  Sample size: {len(df):,} rows")

# --- Step 3: Build features ---
log("\nStep 3/6 — Building features...")
for col in BID_P:
    df[col + "_dist"] = df["mid_price"] - df[col]
for col in ASK_P:
    df[col + "_dist"] = df[col] - df["mid_price"]
DIST_FEATURES = [c + "_dist" for c in BID_P] + [c + "_dist" for c in ASK_P]
FEATURES = BASE_FEATURES + DIST_FEATURES
df = df.dropna(subset=FEATURES).reset_index(drop=True)
log(f"  Features: {len(FEATURES)} columns, {len(df):,} rows after NaN drop")

# --- Steps 4-6: Loop over horizons with auto-balanced thresholds ---
log(f"\n{'='*90}")
log(f"{'Horizon':>10} {'Seconds':>8} {'Threshold':>10} {'Down%':>7} {'Flat%':>7} {'Up%':>7} {'Train Acc':>10} {'Test Acc':>10}")
log(f"{'='*90}")

for HORIZON in HORIZONS:
    # Price change
    price_change = (df["mid_price"].shift(-HORIZON) - df["mid_price"]).dropna()

    # Auto threshold: 67th percentile gives ~33% up, ~34% flat, ~33% down
    upper = price_change.quantile(0.667)
    lower = price_change.quantile(0.333)
    # Use symmetric threshold (average of abs values)
    thresh = round((abs(upper) + abs(lower)) / 2, 2)

    # Label
    full_change = df["mid_price"].shift(-HORIZON) - df["mid_price"]
    label = np.where(full_change > thresh, 1,
            np.where(full_change < -thresh, -1, 0))
    valid = df.index[:-HORIZON]
    X = df.loc[valid, FEATURES].values
    y = label[valid]
    down_pct = 100 * (y == -1).mean()
    flat_pct = 100 * (y ==  0).mean()
    up_pct   = 100 * (y ==  1).mean()

    # Split & Scale
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train with class_weight='balanced' to handle imbalance
    model = SGDClassifier(loss="log_loss", max_iter=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc  = accuracy_score(y_test,  model.predict(X_test))  * 100

    log(f"{HORIZON:>10} {HORIZON/10:>8.1f}s {THRESHOLD:>10.2f} {down_pct:>7.1f}% {flat_pct:>7.1f}% {up_pct:>7.1f}% {train_acc:>9.2f}% {test_acc:>9.2f}%")

    log(f"\n--- Classification Report (Test set) ---")
    log(classification_report(y_test, model.predict(X_test),
        target_names=["Down(-1)", "Flat(0)", "Up(1)"]))

    log("--- Confusion Matrix ---")
    cm = pd.DataFrame(
        confusion_matrix(y_test, model.predict(X_test)),
        index=["Actual Down", "Actual Flat", "Actual Up"],
        columns=["Pred Down", "Pred Flat", "Pred Up"]
    )
    log(str(cm))

log(f"\nDone.")
