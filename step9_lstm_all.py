"""
LSTM Experiment Suite
=====================
Runs 7 experiments and saves all results to results_lstm.txt

Experiments:
  A) 3-class Up/Down/Flat  — horizons 5s, 10s  (all rows)
  B) Binary Up/Down        — horizons 1s, 5s, 10s  (moved rows only)
  C) Binary Move/No-Move   — horizons 5s, 10s  (all rows)
"""

import pandas as pd
import numpy as np
import glob
import os
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

SAVE_DIR  = "C:/Users/marou/Documents/ClaudCode1"
DATA_PATH = f"{SAVE_DIR}/parquet_01/*.parquet"
RESULTS_FILE = f"{SAVE_DIR}/results_lstm.txt"

SEQ_LEN    = 10
THRESHOLD  = 0.5
BATCH_SIZE = 1024
EPOCHS     = 10
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg): print(msg, flush=True)

# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=82, hidden_size=128, num_layers=2,
                 dropout=0.3, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ── Dataset ───────────────────────────────────────────────────────────────────
class LOBDataset(Dataset):
    def __init__(self, indices, labels, X_scaled):
        self.indices  = indices
        self.labels   = labels
        self.X_scaled = X_scaled

    def __len__(self): return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        seq = self.X_scaled[i - SEQ_LEN + 1 : i + 1]
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(self.labels[k], dtype=torch.long))

# ── Training loop ─────────────────────────────────────────────────────────────
def train_and_eval(train_loader, test_loader, n_classes, class_weights, tag):
    model     = LSTMModel(n_classes=n_classes).to(DEVICE)
    weights   = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc   = 0.0
    best_state = None
    best_preds = None
    best_true  = None

    log(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>9}")
    log(f"  {'-'*48}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total_n = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            out  = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            correct    += (out.argmax(1) == batch_y).sum().item()
            total_n    += len(batch_y)

        train_acc = correct / total_n
        avg_loss  = total_loss / total_n

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                p = model(batch_x).argmax(1).cpu().numpy()
                preds.extend(p)
                trues.extend(batch_y.numpy())

        test_acc = accuracy_score(trues, preds)
        log(f"  {epoch:>5} | {avg_loss:>10.4f} | {train_acc:>9.3f} | {test_acc:>9.3f}")

        if test_acc > best_acc:
            best_acc   = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = preds[:]
            best_true  = trues[:]

    # Save best model
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{SAVE_DIR}/model_{tag}.pt")
    log(f"  Best test acc: {best_acc:.3f} — saved model_{tag}.pt")
    return best_acc, best_preds, best_true, model

# ── Helper: build indices + labels for an experiment ─────────────────────────
def make_split(all_indices, all_labels, X_raw):
    split     = int(len(all_indices) * 0.8)
    train_idx = all_indices[:split]
    test_idx  = all_indices[split:]
    train_lbl = all_labels[:split]
    test_lbl  = all_labels[split:]

    # Scaler fit on training period only
    scaler = StandardScaler()
    scaler.fit(X_raw[: train_idx[-1] + 1])
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    return train_idx, test_idx, train_lbl, test_lbl, scaler, X_scaled

def make_loaders(train_idx, test_idx, train_lbl, test_lbl, X_scaled):
    train_ds = LOBDataset(train_idx, train_lbl, X_scaled)
    test_ds  = LOBDataset(test_idx,  test_lbl,  X_scaled)
    trl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=0, pin_memory=True)
    tel = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=0, pin_memory=True)
    return trl, tel

def class_weights_uniform(counts):
    total = sum(counts)
    n     = len(counts)
    return [total / (n * c) for c in counts]

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
log("="*65)
log("LSTM EXPERIMENT SUITE")
log("="*65)
log(f"Device : {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'})")
log(f"SEQ_LEN: {SEQ_LEN}  |  EPOCHS: {EPOCHS}  |  BATCH: {BATCH_SIZE}")

log("\n[Data] Loading parquet files...")
files = sorted(glob.glob(DATA_PATH))
df    = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df    = df.sort_values("event_ts").reset_index(drop=True)
log(f"  {len(df):,} rows")

log("[Data] Building features...")
BID_V = [f"bid_v_{i:02d}" for i in range(20)]
ASK_V = [f"ask_v_{i:02d}" for i in range(20)]
BID_P = [f"bid_p_{i:02d}" for i in range(20)]
ASK_P = [f"ask_p_{i:02d}" for i in range(20)]
BASE  = ["spread_l1", "vol_imbalance_l1"] + BID_V + ASK_V
for col in BID_P: df[col + "_dist"] = df["mid_price"] - df[col]
for col in ASK_P: df[col + "_dist"] = df[col] - df["mid_price"]
DIST     = [c + "_dist" for c in BID_P] + [c + "_dist" for c in ASK_P]
FEATURES = BASE + DIST
df = df.dropna(subset=FEATURES).reset_index(drop=True)
X_raw = df[FEATURES].values.astype(np.float32)
log(f"  {len(FEATURES)} features, {len(df):,} rows")

# Results accumulator
all_results = []

# =============================================================================
# EXPERIMENT GROUP A — 3-class Up / Flat / Down
# =============================================================================
log("\n" + "="*65)
log("GROUP A: 3-class (Up / Flat / Down)")
log("="*65)

for horizon_ticks in [50, 100]:
    horizon_sec = horizon_ticks // 10
    tag = f"3class_{horizon_sec}s"
    log(f"\n--- A{horizon_sec}: {horizon_sec}s horizon ---")

    price_change = df["mid_price"].shift(-horizon_ticks) - df["mid_price"]
    label = np.where(price_change >  THRESHOLD, 2,
            np.where(price_change < -THRESHOLD, 0, 1))  # 0=Down,1=Flat,2=Up

    valid   = np.arange(SEQ_LEN - 1, len(df) - horizon_ticks)
    lbls    = label[valid].astype(np.int64)
    n_down  = (lbls == 0).sum()
    n_flat  = (lbls == 1).sum()
    n_up    = (lbls == 2).sum()
    log(f"  Down: {n_down:,} ({100*n_down/len(lbls):.1f}%)  Flat: {n_flat:,} ({100*n_flat/len(lbls):.1f}%)  Up: {n_up:,} ({100*n_up/len(lbls):.1f}%)")

    tr_idx, te_idx, tr_lbl, te_lbl, scaler, X_sc = make_split(valid, lbls, X_raw)
    joblib.dump(scaler, f"{SAVE_DIR}/scaler_{tag}.pkl")
    trl, tel = make_loaders(tr_idx, te_idx, tr_lbl, te_lbl, X_sc)
    cw = class_weights_uniform([n_down, n_flat, n_up])

    best_acc, preds, trues, _ = train_and_eval(trl, tel, 3, cw, tag)
    report = classification_report(trues, preds, target_names=["Down","Flat","Up"])
    log("\n" + report)
    all_results.append(("A", tag, best_acc, report))

# =============================================================================
# EXPERIMENT GROUP B — Binary Up / Down  (moved rows only)
# =============================================================================
log("\n" + "="*65)
log("GROUP B: Binary Up/Down (moved rows only)")
log("="*65)

for horizon_ticks in [10, 50, 100]:
    horizon_sec = horizon_ticks // 10
    tag = f"updown_{horizon_sec}s"
    log(f"\n--- B{horizon_sec}: {horizon_sec}s horizon ---")

    price_change = df["mid_price"].shift(-horizon_ticks) - df["mid_price"]
    label_raw = np.where(price_change >  THRESHOLD,  1,
                np.where(price_change < -THRESHOLD,  0, -1))  # -1=flat

    valid   = np.arange(SEQ_LEN - 1, len(df) - horizon_ticks)
    mask    = label_raw[valid] != -1
    moved   = valid[mask]
    lbls    = label_raw[moved].astype(np.int64)
    n_down  = (lbls == 0).sum()
    n_up    = (lbls == 1).sum()
    log(f"  Moved rows: {len(lbls):,}  |  Down: {n_down:,} ({100*n_down/len(lbls):.1f}%)  Up: {n_up:,} ({100*n_up/len(lbls):.1f}%)")

    tr_idx, te_idx, tr_lbl, te_lbl, scaler, X_sc = make_split(moved, lbls, X_raw)
    joblib.dump(scaler, f"{SAVE_DIR}/scaler_{tag}.pkl")
    trl, tel = make_loaders(tr_idx, te_idx, tr_lbl, te_lbl, X_sc)
    cw = class_weights_uniform([n_down, n_up])

    best_acc, preds, trues, _ = train_and_eval(trl, tel, 2, cw, tag)
    report = classification_report(trues, preds, target_names=["Down","Up"])
    log("\n" + report)
    all_results.append(("B", tag, best_acc, report))

# =============================================================================
# EXPERIMENT GROUP C — Binary Move / No-Move
# =============================================================================
log("\n" + "="*65)
log("GROUP C: Binary Move/No-Move")
log("="*65)

for horizon_ticks in [50, 100]:
    horizon_sec = horizon_ticks // 10
    tag = f"move_{horizon_sec}s"
    log(f"\n--- C{horizon_sec}: {horizon_sec}s horizon ---")

    price_change = df["mid_price"].shift(-horizon_ticks) - df["mid_price"]
    label = np.where(np.abs(price_change) > THRESHOLD, 1, 0)  # 1=Move,0=NoMove

    valid  = np.arange(SEQ_LEN - 1, len(df) - horizon_ticks)
    lbls   = label[valid].astype(np.int64)
    n_flat = (lbls == 0).sum()
    n_move = (lbls == 1).sum()
    log(f"  No-Move: {n_flat:,} ({100*n_flat/len(lbls):.1f}%)  Move: {n_move:,} ({100*n_move/len(lbls):.1f}%)")

    tr_idx, te_idx, tr_lbl, te_lbl, scaler, X_sc = make_split(valid, lbls, X_raw)
    joblib.dump(scaler, f"{SAVE_DIR}/scaler_{tag}.pkl")
    trl, tel = make_loaders(tr_idx, te_idx, tr_lbl, te_lbl, X_sc)
    cw = class_weights_uniform([n_flat, n_move])

    best_acc, preds, trues, _ = train_and_eval(trl, tel, 2, cw, tag)
    report = classification_report(trues, preds, target_names=["No-Move","Move"])
    log("\n" + report)
    all_results.append(("C", tag, best_acc, report))

# =============================================================================
# SAVE RESULTS SUMMARY
# =============================================================================
log("\n" + "="*65)
log("ALL RESULTS SUMMARY")
log("="*65)

lines = []
lines.append("LSTM EXPERIMENT RESULTS")
lines.append(f"SEQ_LEN={SEQ_LEN}  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}")
lines.append(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'})")
lines.append("")
lines.append("Baselines (logistic regression):")
lines.append("  Binary Up/Down 1s : 77%")
lines.append("  Move/No-Move 5s   : 64% (88% precision at threshold=0.85)")
lines.append("")
lines.append("="*65)
lines.append(f"{'Group':<8} {'Experiment':<20} {'Best Test Acc':>13}")
lines.append("-"*45)

for grp, tag, acc, _ in all_results:
    lines.append(f"{grp:<8} {tag:<20} {acc:>12.1%}")
    log(f"  {grp}  {tag:<20}  {acc:.1%}")

lines.append("")
lines.append("="*65)
lines.append("DETAILED CLASSIFICATION REPORTS")
lines.append("="*65)
for grp, tag, acc, report in all_results:
    lines.append(f"\n[{grp}] {tag}  (best acc: {acc:.1%})")
    lines.append(report)

with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(lines))

log(f"\nResults saved to: results_lstm.txt")
log("Done.")
