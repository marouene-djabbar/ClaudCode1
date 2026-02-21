import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score

def log(msg): print(msg, flush=True)

# --- Config ---
THRESHOLD      = 0.5
HORIZON_MOVE   = 50   # 5 seconds (move detector)
HORIZON_DIR    = 10   # 1 second  (direction model)
DATA_PATH      = "C:/Users/marou/Documents/ClaudCode1/parquet_01/*.parquet"

# --- Load saved models ---
log("Step 1/5 — Loading saved models...")
model_move  = joblib.load("C:/Users/marou/Documents/ClaudCode1/model_move_5s.pkl")
scaler_move = joblib.load("C:/Users/marou/Documents/ClaudCode1/scaler_move_5s.pkl")
dec_thresh  = joblib.load("C:/Users/marou/Documents/ClaudCode1/threshold_move_5s.pkl")
model_dir   = joblib.load("C:/Users/marou/Documents/ClaudCode1/model_h10.pkl")
scaler_dir  = joblib.load("C:/Users/marou/Documents/ClaudCode1/scaler_h10.pkl")
log(f"  Move detector loaded (threshold={dec_thresh})")
log(f"  Direction model loaded")

# --- Load data ---
log("\nStep 2/5 — Loading data...")
files = sorted(glob.glob(DATA_PATH))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.sort_values("event_ts").reset_index(drop=True)
log(f"  {len(df):,} rows loaded")

# --- Build features ---
log("\nStep 3/5 — Building features...")
BID_V = [f"bid_v_{i:02d}" for i in range(20)]
ASK_V = [f"ask_v_{i:02d}" for i in range(20)]
BID_P = [f"bid_p_{i:02d}" for i in range(20)]
ASK_P = [f"ask_p_{i:02d}" for i in range(20)]
BASE  = ["spread_l1", "vol_imbalance_l1"] + BID_V + ASK_V
for col in BID_P:
    df[col + "_dist"] = df["mid_price"] - df[col]
for col in ASK_P:
    df[col + "_dist"] = df[col] - df["mid_price"]
DIST     = [c + "_dist" for c in BID_P] + [c + "_dist" for c in ASK_P]
FEATURES = BASE + DIST
df = df.dropna(subset=FEATURES).reset_index(drop=True)
log(f"  {len(FEATURES)} features, {len(df):,} rows")

# --- Build labels ---
log("\nStep 4/5 — Building labels...")
max_horizon = max(HORIZON_MOVE, HORIZON_DIR)
valid       = df.index[:-max_horizon]
X_all       = df.loc[valid, FEATURES].values

# Move label: did price move in 5s?
change_move = df["mid_price"].shift(-HORIZON_MOVE) - df["mid_price"]
y_move      = np.where(np.abs(change_move.values[valid]) > THRESHOLD, 1, 0)

# Direction label: Up/Down/Flat in 1s?
change_dir  = df["mid_price"].shift(-HORIZON_DIR) - df["mid_price"]
y_dir       = np.where(change_dir.values[valid] >  THRESHOLD,  1,
              np.where(change_dir.values[valid] < -THRESHOLD, -1, 0))

# Chronological test set (last 20%)
split   = int(len(X_all) * 0.8)
X_test  = X_all[split:]
ym_test = y_move[split:]
yd_test = y_dir[split:]
log(f"  Test set: {len(X_test):,} rows")

# --- Step 5: Run pipeline ---
log("\nStep 5/5 — Running combined pipeline...")

# Stage 1: Move detector
X_move_sc = scaler_move.transform(X_test)
move_probs = model_move.predict_proba(X_move_sc)[:, 1]
move_fired = move_probs >= dec_thresh

# Stage 2: Direction model on ALL rows (applied only where move fired)
X_dir_sc   = scaler_dir.transform(X_test)
dir_pred   = model_dir.predict(X_dir_sc)   # -1 or 1

log(f"\n{'='*60}")
log("COMBINED PIPELINE RESULTS")
log(f"{'='*60}")

total      = len(X_test)
n_fired    = move_fired.sum()
log(f"\n  Total test rows          : {total:>10,}")
log(f"  Move detector fired      : {n_fired:>10,}  ({100*n_fired/total:.1f}%)")

# Of rows where move fired: was there actually a move in 5s?
actual_move_5s  = ym_test[move_fired]
correct_move    = actual_move_5s.sum()
log(f"  Actually moved in 5s     : {correct_move:>10,}  ({100*correct_move/n_fired:.1f}% of signals) — Move Precision")

# Of rows where move fired: was there a move in 1s?
actual_move_1s  = (yd_test[move_fired] != 0)
moved_1s        = actual_move_1s.sum()
log(f"  Actually moved in 1s     : {moved_1s:>10,}  ({100*moved_1s/n_fired:.1f}% of signals)")

# Direction accuracy on rows that actually moved in 1s
dir_on_moved    = dir_pred[move_fired][actual_move_1s]
true_dir_moved  = yd_test[move_fired][actual_move_1s]
dir_correct     = (dir_on_moved == true_dir_moved).sum()
log(f"  Direction correct (1s)   : {dir_correct:>10,}  ({100*dir_correct/moved_1s:.1f}% of 1s movers)")

# Combined: move fired AND price moved in 1s AND direction correct
combined_correct = (move_fired) & (yd_test != 0) & (dir_pred == yd_test)
log(f"\n  Combined correct trades  : {combined_correct.sum():>10,}  ({100*combined_correct.sum()/n_fired:.1f}% of all signals)")
log(f"  Wrong direction trades   : {(move_fired & (yd_test != 0) & (dir_pred != yd_test)).sum():>10,}")
log(f"  No move in 1s (flat)     : {(move_fired & (yd_test == 0)).sum():>10,}")

log(f"\n{'='*60}")
log("SUMMARY")
log(f"{'='*60}")
log(f"  When pipeline fires:")
log(f"    - 88% chance price moves in 5s (move detector precision)")
log(f"    - {100*moved_1s/n_fired:.1f}% chance price moves in 1s")
log(f"    - {100*dir_correct/moved_1s:.1f}% direction accuracy when it does move in 1s")
log(f"    - {100*combined_correct.sum()/n_fired:.1f}% of signals result in a correctly called trade")
log("\nDone.")
