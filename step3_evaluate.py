import pandas as pd
import numpy as np
import glob
import joblib

def log(msg): print(msg, flush=True)

# --- Config ---
THRESHOLD   = 0.5
HORIZON     = 10
DATA_PATH   = "C:/Users/marou/Documents/ClaudCode1/parquet_01/*.parquet"
MODEL_PATH  = "C:/Users/marou/Documents/ClaudCode1/model_h10.pkl"
SCALER_PATH = "C:/Users/marou/Documents/ClaudCode1/scaler_h10.pkl"

BID_V = [f"bid_v_{i:02d}" for i in range(20)]
ASK_V = [f"ask_v_{i:02d}" for i in range(20)]
BID_P = [f"bid_p_{i:02d}" for i in range(20)]
ASK_P = [f"ask_p_{i:02d}" for i in range(20)]
BASE_FEATURES = ["spread_l1", "vol_imbalance_l1"] + BID_V + ASK_V

# --- Step 1: Load saved model and scaler ---
log("Step 1/4 — Loading saved model and scaler...")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
log("  Model and scaler loaded.")

# --- Step 2: Load full data ---
log("\nStep 2/4 — Loading full dataset...")
files = sorted(glob.glob(DATA_PATH))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.sort_values("event_ts").reset_index(drop=True)
log(f"  {len(df):,} rows loaded from {len(files)} files")

# --- Step 3: Build features and labels ---
log("\nStep 3/4 — Building features and 3-class labels...")
for col in BID_P:
    df[col + "_dist"] = df["mid_price"] - df[col]
for col in ASK_P:
    df[col + "_dist"] = df[col] - df["mid_price"]
DIST_FEATURES = [c + "_dist" for c in BID_P] + [c + "_dist" for c in ASK_P]
FEATURES = BASE_FEATURES + DIST_FEATURES
df = df.dropna(subset=FEATURES).reset_index(drop=True)

full_change = df["mid_price"].shift(-HORIZON) - df["mid_price"]
y_true = np.where(full_change >  THRESHOLD,  1,
         np.where(full_change < -THRESHOLD, -1, 0))

valid   = df.index[:-HORIZON]
X_all   = df.loc[valid, FEATURES].values
y_all   = y_true[valid]
log(f"  {len(X_all):,} rows ready for evaluation")

# --- Step 4: Predict and evaluate ---
log("\nStep 4/4 — Predicting on ALL rows...")
X_scaled = scaler.transform(X_all)
y_pred   = model.predict(X_scaled)
log("  Done predicting.")

total   = len(y_all)
correct = (y_pred == y_all).sum()

log(f"\n{'='*55}")
log(f"  OVERALL ACCURACY: {100*correct/total:.2f}%  ({correct:,} / {total:,})")
log(f"{'='*55}")

log("\n--- Breakdown by true class ---")
for label, name in [(-1, "Down"), (0, "Flat"), (1, "Up")]:
    mask     = y_all == label
    n        = mask.sum()
    acc      = (y_pred[mask] == label).mean() * 100
    pred_d   = (y_pred[mask] == -1).mean() * 100
    pred_f   = (y_pred[mask] ==  0).mean() * 100
    pred_u   = (y_pred[mask] ==  1).mean() * 100
    log(f"  {name:>5} ({n:>9,} rows) | Correct: {acc:5.1f}% | Pred as Down: {pred_d:5.1f}%  Flat: {pred_f:5.1f}%  Up: {pred_u:5.1f}%")

log("\n--- Summary ---")
down_n = (y_all == -1).sum()
flat_n = (y_all ==  0).sum()
up_n   = (y_all ==  1).sum()
log(f"  True Down: {100*down_n/total:.1f}% | True Flat: {100*flat_n/total:.1f}% | True Up: {100*up_n/total:.1f}%")
log(f"  Pred Down: {100*(y_pred==-1).mean():.1f}% | Pred Flat: {100*(y_pred==0).mean():.1f}% | Pred Up: {100*(y_pred==1).mean():.1f}%")

log("\nDone.")
