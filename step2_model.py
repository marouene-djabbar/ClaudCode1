import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def log(msg):
    print(msg, flush=True)

# --- Config ---
THRESHOLD  = 0.5   # 1 tick
HORIZONS   = [10]  # 1 second
SAMPLE     = None  # full data
BINARY     = True  # drop Flat, predict Up vs Down only
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
if SAMPLE:
    log(f"\nStep 2/6 — Sampling {SAMPLE:,} rows (chronological)...")
    df = df.iloc[:SAMPLE].reset_index(drop=True)
    log(f"  Sample size: {len(df):,} rows")
else:
    log(f"\nStep 2/6 — Using full dataset ({len(df):,} rows)...")
log(f"  Mode: {'Binary (Up vs Down only)' if BINARY else '3-class (Up/Flat/Down)'}")

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
    # Label using fixed threshold
    full_change = df["mid_price"].shift(-HORIZON) - df["mid_price"]
    label = np.where(full_change > THRESHOLD,  1,
            np.where(full_change < -THRESHOLD, -1, 0))
    valid = df.index[:-HORIZON]
    X = df.loc[valid, FEATURES].values
    y = label[valid]

    # Binary mode: drop Flat rows
    if BINARY:
        mask = y != 0
        X, y = X[mask], y[mask]
        log(f"  Binary mode — kept {mask.sum():,} rows (dropped Flat)")

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

    target_names = ["Down(-1)", "Up(1)"] if BINARY else ["Down(-1)", "Flat(0)", "Up(1)"]
    log(f"\n--- Classification Report (Test set) ---")
    log(classification_report(y_test, model.predict(X_test), target_names=target_names))

    log("--- Confusion Matrix ---")
    cm = pd.DataFrame(
        confusion_matrix(y_test, model.predict(X_test)),
        index=[f"Actual {t}" for t in target_names],
        columns=[f"Pred {t}" for t in target_names]
    )
    log(str(cm))

    # --- Save model and scaler ---
    model_path  = f"C:/Users/marou/Documents/ClaudCode1/model_h{HORIZON}.pkl"
    scaler_path = f"C:/Users/marou/Documents/ClaudCode1/scaler_h{HORIZON}.pkl"
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    log(f"\nModel  saved: {model_path}")
    log(f"Scaler saved: {scaler_path}")

log(f"\nDone.")
