import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def log(msg): print(msg, flush=True)

# --- Config ---
THRESHOLD  = 0.5
HORIZONS   = [50, 100]  # 5 seconds and 10 seconds
SAMPLE     = None       # full data
DATA_PATH  = "C:/Users/marou/Documents/ClaudCode1/parquet_01/*.parquet"

BID_V = [f"bid_v_{i:02d}" for i in range(20)]
ASK_V = [f"ask_v_{i:02d}" for i in range(20)]
BID_P = [f"bid_p_{i:02d}" for i in range(20)]
ASK_P = [f"ask_p_{i:02d}" for i in range(20)]
BASE_FEATURES = ["spread_l1", "vol_imbalance_l1"] + BID_V + ASK_V

# --- Step 1: Load data ---
log("Step 1/5 — Loading data...")
files = sorted(glob.glob(DATA_PATH))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.sort_values("event_ts").reset_index(drop=True)
log(f"  {len(df):,} rows loaded")

# --- Step 2: Sample ---
if SAMPLE:
    log(f"\nStep 2/5 — Sampling {SAMPLE:,} rows...")
    df = df.iloc[:SAMPLE].reset_index(drop=True)
else:
    log(f"\nStep 2/5 — Using full dataset ({len(df):,} rows)...")

# --- Step 3: Build features ---
log("\nStep 3/5 — Building features...")
for col in BID_P:
    df[col + "_dist"] = df["mid_price"] - df[col]
for col in ASK_P:
    df[col + "_dist"] = df[col] - df["mid_price"]
DIST_FEATURES = [c + "_dist" for c in BID_P] + [c + "_dist" for c in ASK_P]
FEATURES = BASE_FEATURES + DIST_FEATURES
df = df.dropna(subset=FEATURES).reset_index(drop=True)
log(f"  {len(FEATURES)} features, {len(df):,} rows")

# --- Steps 4-5: Loop over horizons ---
for HORIZON in HORIZONS:
    log(f"\n{'='*60}")
    log(f"HORIZON: {HORIZON} ticks ({HORIZON/10:.0f} seconds)")
    log(f"{'='*60}")

    # Label
    full_change = df["mid_price"].shift(-HORIZON) - df["mid_price"]
    label = np.where(np.abs(full_change) > THRESHOLD, 1, 0)
    valid = df.index[:-HORIZON]
    X     = df.loc[valid, FEATURES].values
    y     = label[valid]

    flat_pct = 100 * (y == 0).mean()
    move_pct = 100 * (y == 1).mean()
    log(f"  Flat (0): {(y==0).sum():>8,}  ({flat_pct:.1f}%)")
    log(f"  Move (1): {(y==1).sum():>8,}  ({move_pct:.1f}%)")

    # Split & Scale
    split   = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train
    model = SGDClassifier(loss="log_loss", max_iter=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc  = accuracy_score(y_test,  model.predict(X_test))  * 100
    log(f"\n  Train Accuracy: {train_acc:.2f}%")
    log(f"  Test  Accuracy: {test_acc:.2f}%")

    log("\n--- Classification Report ---")
    log(classification_report(y_test, model.predict(X_test),
        target_names=["Flat(0)", "Move(1)"]))

    log("--- Confusion Matrix ---")
    cm = pd.DataFrame(
        confusion_matrix(y_test, model.predict(X_test)),
        index=["Actual Flat", "Actual Move"],
        columns=["Pred Flat", "Pred Move"]
    )
    log(str(cm))

log("\nDone.")
