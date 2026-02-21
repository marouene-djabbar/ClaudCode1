import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def log(msg): print(msg, flush=True)

# --- Config ---
THRESHOLD     = 0.5
HORIZON_S1    = 50   # Stage 1: 5 seconds (move detector)
HORIZON_S3    = 10   # Stage 3: 1 second  (direction)
DATA_PATH     = "C:/Users/marou/Documents/ClaudCode1/parquet_01/*.parquet"
MODEL_S3_PATH = "C:/Users/marou/Documents/ClaudCode1/model_h10.pkl"
SCALER_S3_PATH= "C:/Users/marou/Documents/ClaudCode1/scaler_h10.pkl"

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
log(f"  {len(df):,} rows loaded")

# --- Step 2: Build features ---
log("\nStep 2/6 — Building features...")
for col in BID_P:
    df[col + "_dist"] = df["mid_price"] - df[col]
for col in ASK_P:
    df[col + "_dist"] = df[col] - df["mid_price"]
DIST_FEATURES = [c + "_dist" for c in BID_P] + [c + "_dist" for c in ASK_P]
FEATURES = BASE_FEATURES + DIST_FEATURES
df = df.dropna(subset=FEATURES).reset_index(drop=True)
log(f"  {len(FEATURES)} features, {len(df):,} rows")

# Use only rows where both horizons have valid future data
max_horizon = max(HORIZON_S1, HORIZON_S3)
valid = df.index[:-max_horizon]
X_all = df.loc[valid, FEATURES].values

# --- Step 3: Build labels for both stages ---
log("\nStep 3/6 — Building labels...")
change_s1 = df["mid_price"].shift(-HORIZON_S1) - df["mid_price"]
change_s3 = df["mid_price"].shift(-HORIZON_S3) - df["mid_price"]

y_s1 = np.where(np.abs(change_s1) > THRESHOLD, 1, 0)[valid]   # Move(1) vs Flat(0) at 5s
y_s3 = np.where(change_s3 > THRESHOLD, 1,
       np.where(change_s3 < -THRESHOLD, -1, 0))[valid]         # Up/Flat/Down at 1s

log(f"  Stage 1 labels — Flat: {(y_s1==0).sum():,} ({100*(y_s1==0).mean():.1f}%)  Move: {(y_s1==1).sum():,} ({100*(y_s1==1).mean():.1f}%)")
log(f"  Stage 3 labels — Down: {(y_s3==-1).sum():,} ({100*(y_s3==-1).mean():.1f}%)  Flat: {(y_s3==0).sum():,} ({100*(y_s3==0).mean():.1f}%)  Up: {(y_s3==1).sum():,} ({100*(y_s3==1).mean():.1f}%)")

# --- Step 4: Train Stage 1 (80/20 chronological split) ---
log("\nStep 4/6 — Training Stage 1 (Move detector, 5s)...")
split = int(len(X_all) * 0.8)
X_train_s1 = X_all[:split]
y_train_s1 = y_s1[:split]

scaler_s1  = StandardScaler()
X_train_s1_sc = scaler_s1.fit_transform(X_train_s1)
X_all_s1_sc   = scaler_s1.transform(X_all)

model_s1 = SGDClassifier(loss="log_loss", max_iter=100, random_state=42, class_weight="balanced")
model_s1.fit(X_train_s1_sc, y_train_s1)
log(f"  Stage 1 trained on {len(X_train_s1):,} rows")

# --- Step 5: Load Stage 3 ---
log("\nStep 5/6 — Loading Stage 3 (direction model, 1s)...")
model_s3  = joblib.load(MODEL_S3_PATH)
scaler_s3 = joblib.load(SCALER_S3_PATH)
log("  Stage 3 loaded.")

# --- Step 6: Run pipeline on test set ---
log("\nStep 6/6 — Running full pipeline on test set...")
X_test    = X_all[split:]
y_s1_test = y_s1[split:]
y_s3_test = y_s3[split:]

# Stage 1 predictions
X_test_s1_sc = scaler_s1.transform(X_test)
pred_s1      = model_s1.predict(X_test_s1_sc)

# Rows Stage 1 flags as Move
move_mask    = pred_s1 == 1
X_move       = X_test[move_mask]
y_s3_move    = y_s3_test[move_mask]  # true direction labels for those rows
y_s1_move    = y_s1_test[move_mask]  # true move labels for those rows

log(f"\n  Test set: {len(X_test):,} rows")
log(f"  Stage 1 flagged as Move: {move_mask.sum():,} ({100*move_mask.mean():.1f}%)")
log(f"  Of those, actually had a move (5s): {y_s1_move.sum():,} ({100*y_s1_move.mean():.1f}%)")

# Stage 3 predictions on flagged rows
X_move_s3_sc = scaler_s3.transform(X_move)
pred_s3      = model_s3.predict(X_move_s3_sc)

log(f"\n{'='*60}")
log("PIPELINE RESULTS")
log(f"{'='*60}")

# Stage 1 standalone accuracy on test
s1_acc = accuracy_score(y_s1_test, pred_s1) * 100
log(f"\n  Stage 1 accuracy (Move vs Flat, 5s):  {s1_acc:.2f}%")

# Stage 3: evaluated only on non-flat ground truth rows
nflat_mask = y_s3_move != 0
if nflat_mask.sum() > 0:
    s3_acc = accuracy_score(y_s3_move[nflat_mask], pred_s3[nflat_mask]) * 100
    log(f"  Stage 3 accuracy (Up vs Down, 1s):    {s3_acc:.2f}%  (on {nflat_mask.sum():,} rows with actual moves)")

# Combined: Stage 1 correct AND Stage 3 correct direction
actually_moved_1s = y_s3_move != 0
correct_direction = pred_s3 == y_s3_move
combined_correct  = actually_moved_1s & correct_direction
log(f"\n  Combined pipeline:")
log(f"    Rows acted on (Stage 1 = Move):    {len(X_move):,}")
log(f"    Of those, price actually moved 1s: {actually_moved_1s.sum():,} ({100*actually_moved_1s.mean():.1f}%)")
log(f"    Of those, direction correct:       {combined_correct.sum():,} ({100*combined_correct.mean():.1f}%)")

log(f"\n--- Stage 3 Classification Report (on Stage 1 flagged rows) ---")
log(classification_report(y_s3_move, pred_s3,
    target_names=["Down(-1)", "Flat(0)", "Up(1)"],
    labels=[-1, 0, 1]))

log("\nDone.")
