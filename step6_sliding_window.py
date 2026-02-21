import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

def log(msg): print(msg, flush=True)

# --- Config ---
THRESHOLD   = 0.5
HORIZON     = 10       # 1 second
WINDOWS     = [10, 20, 30]
BATCH_SIZE  = 200_000
DATA_PATH   = "C:/Users/marou/Documents/ClaudCode1/parquet_01/*.parquet"

BID_V = [f"bid_v_{i:02d}" for i in range(20)]
ASK_V = [f"ask_v_{i:02d}" for i in range(20)]
BID_P = [f"bid_p_{i:02d}" for i in range(20)]
ASK_P = [f"ask_p_{i:02d}" for i in range(20)]
BASE_FEATURES = ["spread_l1", "vol_imbalance_l1"] + BID_V + ASK_V

# --- Step 1: Load data ---
log("Step 1/4 — Loading full data...")
files = sorted(glob.glob(DATA_PATH))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.sort_values("event_ts").reset_index(drop=True)
log(f"  {len(df):,} rows loaded")

# --- Step 2: Build base features ---
log("\nStep 2/4 — Building base features...")
for col in BID_P:
    df[col + "_dist"] = df["mid_price"] - df[col]
for col in ASK_P:
    df[col + "_dist"] = df[col] - df["mid_price"]
DIST_FEATURES = [c + "_dist" for c in BID_P] + [c + "_dist" for c in ASK_P]
FEATURES = BASE_FEATURES + DIST_FEATURES
df = df.dropna(subset=FEATURES).reset_index(drop=True)

# Labels for all rows
full_change = df["mid_price"].shift(-HORIZON) - df["mid_price"]
y_all = np.where(np.abs(full_change.values) > THRESHOLD, 1, 0)
feat_matrix = df[FEATURES].values.astype(np.float32)
log(f"  {len(FEATURES)} base features, {len(df):,} rows")

# Chronological split index
SPLIT = int(len(df) * 0.8)

# --- Step 3: Loop over window sizes ---
log(f"\nStep 3/4 — Testing window sizes: {WINDOWS}")
log(f"\n{'='*75}")
log(f"{'Window':>8} {'Features':>10} {'Train Acc':>10} {'Test Acc':>10} {'Move F1':>10} {'Move Rec':>10}")
log(f"{'='*75}")

for WINDOW in WINDOWS:
    n_features = WINDOW * len(FEATURES)

    # Fit scaler on a sample from training data
    sample_idx = np.linspace(WINDOW, SPLIT - HORIZON, 5000, dtype=int)
    sample_X = np.array([feat_matrix[i - WINDOW + 1: i + 1].flatten() for i in sample_idx])
    scaler = StandardScaler()
    scaler.fit(sample_X)

    # Compute class weights manually (required for partial_fit)
    y_train_all = y_all[WINDOW - 1: SPLIT - HORIZON]
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train_all)
    class_weights = {0: weights[0], 1: weights[1]}

    model = SGDClassifier(loss="log_loss", max_iter=1, random_state=42,
                          class_weight=class_weights, warm_start=True)

    # --- Train in batches ---
    starts = list(range(WINDOW - 1, SPLIT - HORIZON, BATCH_SIZE))
    for i, start in enumerate(starts):
        end = min(start + BATCH_SIZE, SPLIT - HORIZON)
        X_batch = np.array([feat_matrix[t - WINDOW + 1: t + 1].flatten()
                            for t in range(start, end)])
        y_batch = y_all[start:end]
        X_batch = scaler.transform(X_batch)
        if i == 0:
            model.partial_fit(X_batch, y_batch, classes=[0, 1])
        else:
            model.partial_fit(X_batch, y_batch)

    # --- Evaluate on train sample ---
    train_sample = np.array([feat_matrix[t - WINDOW + 1: t + 1].flatten()
                              for t in sample_idx])
    train_pred = model.predict(scaler.transform(train_sample))
    train_acc = accuracy_score(y_all[sample_idx], train_pred) * 100

    # --- Evaluate on full test set ---
    test_X, test_y = [], []
    for start in range(SPLIT, len(df) - HORIZON, BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(df) - HORIZON)
        X_batch = np.array([feat_matrix[t - WINDOW + 1: t + 1].flatten()
                            for t in range(start, end)])
        test_X.append(scaler.transform(X_batch))
        test_y.append(y_all[start:end])

    test_X = np.vstack(test_X)
    test_y = np.concatenate(test_y)
    test_pred = model.predict(test_X)
    test_acc  = accuracy_score(test_y, test_pred) * 100

    move_f1  = f1_score(test_y, test_pred, pos_label=1)
    move_rec = recall_score(test_y, test_pred, pos_label=1)

    log(f"{WINDOW:>8} {n_features:>10,} {train_acc:>9.2f}% {test_acc:>9.2f}% {move_f1:>10.3f} {move_rec:>10.3f}")

# --- Step 4: Detailed report for last window ---
log(f"\nStep 4/4 — Detailed report for window={WINDOW}...")
log(classification_report(test_y, test_pred, target_names=["Flat(0)", "Move(1)"]))
log("\nDone.")
