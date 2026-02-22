"""
DeepLOB Experiment Suite
========================
CNN (spatial) + Inception + LSTM (temporal) architecture for LOB data.
Based on Zhang et al. 2019 "DeepLOB: Deep Convolutional Neural Networks
for Limit Order Books", adapted for 20-level order book.

Feature arrangement for CNN (80 features, 20 levels):
  For each level i: [bid_p_dist, bid_vol, ask_p_dist, ask_vol]
  This lets the CNN learn (price, volume) pairs and (bid, ask) pairs
  via small 1x2 kernels before aggregating across all levels.

Experiments:
  A) Binary Up/Down  — 1s, 5s  (moved rows only)
  B) 3-class Up/Flat/Down — 5s (all rows)
"""

import pandas as pd
import numpy as np
import glob
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

SAVE_DIR  = "C:/Users/marou/Documents/ClaudCode1"
DATA_PATH = f"{SAVE_DIR}/parquet_01/*.parquet"
RESULTS_FILE = f"{SAVE_DIR}/results_deeplob.txt"

SEQ_LEN    = 10
THRESHOLD  = 0.5
BATCH_SIZE = 512     # smaller than LSTM — model is bigger
EPOCHS     = 15
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(msg): print(msg, flush=True)

# ── DeepLOB Model ─────────────────────────────────────────────────────────────
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 4 parallel branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # concat along channels


class DeepLOB(nn.Module):
    """
    Input: (batch, 1, SEQ_LEN, 80)
      - 80 = 20 levels × [bid_p_dist, bid_vol, ask_p_dist, ask_vol]

    CNN block:
      Conv(1, 32, 1×2, stride 1×2) → pairs (price, volume)       → (batch, 32, T, 40)
      Conv(32, 32, 1×2, stride 1×2) → pairs (bid, ask) per level → (batch, 32, T, 20)
      Conv(32, 32, 1×20)            → aggregate all levels        → (batch, 32, T, 1)

    Inception: (batch, 32, T, 1) → (batch, 128, T, 1)

    Reshape: (batch, T, 128)

    LSTM(128, 64)

    FC(64, n_classes)
    """
    def __init__(self, n_levels=20, n_classes=2, lstm_hidden=64, inception_ch=32):
        super().__init__()
        n_lob_features = n_levels * 4   # 80

        # CNN block — extracts spatial features from the LOB structure
        self.cnn = nn.Sequential(
            # Step 1: pair (price, volume) within bid and ask sides
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),   # → (32, T, 40)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            # Step 2: pair (bid, ask) at the same price level
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),  # → (32, T, 20)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            # Step 3: aggregate across all 20 levels
            nn.Conv2d(32, 32, kernel_size=(1, n_levels)),           # → (32, T, 1)
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # Inception module — captures patterns at different temporal scales
        self.inception = InceptionModule(32, inception_ch)
        inception_out = inception_ch * 4   # 4 branches concatenated = 128

        # LSTM — temporal modelling
        self.lstm = nn.LSTM(inception_out, lstm_hidden,
                            num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)

        # Output
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        # x: (batch, 1, T, 80)
        x = self.cnn(x)                     # → (batch, 32, T, 1)
        x = self.inception(x)               # → (batch, 128, T, 1)
        x = x.squeeze(-1)                   # → (batch, 128, T)
        x = x.permute(0, 2, 1)             # → (batch, T, 128)
        out, _ = self.lstm(x)               # → (batch, T, 64)
        out = self.dropout(out[:, -1, :])   # last time step → (batch, 64)
        return self.fc(out)                 # → (batch, n_classes)


# ── Dataset ───────────────────────────────────────────────────────────────────
class DeepLOBDataset(Dataset):
    def __init__(self, indices, labels, X_lob_scaled):
        self.indices       = indices
        self.labels        = labels
        self.X_lob_scaled  = X_lob_scaled

    def __len__(self): return len(self.indices)

    def __getitem__(self, k):
        i   = self.indices[k]
        # seq: (SEQ_LEN, 80) → add channel dim → (1, SEQ_LEN, 80)
        seq = self.X_lob_scaled[i - SEQ_LEN + 1 : i + 1]
        x   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        lbl = torch.tensor(self.labels[k], dtype=torch.long)
        return x, lbl


# ── Helpers ───────────────────────────────────────────────────────────────────
def class_weights_uniform(counts):
    total = sum(counts)
    return [total / (len(counts) * c) for c in counts]

def make_split(indices, labels, X_raw_lob):
    split     = int(len(indices) * 0.8)
    tr_idx    = indices[:split]
    te_idx    = indices[split:]
    tr_lbl    = labels[:split]
    te_lbl    = labels[split:]
    scaler    = StandardScaler()
    scaler.fit(X_raw_lob[: tr_idx[-1] + 1])
    X_sc      = scaler.transform(X_raw_lob).astype(np.float32)
    return tr_idx, te_idx, tr_lbl, te_lbl, scaler, X_sc

def make_loaders(tr_idx, te_idx, tr_lbl, te_lbl, X_sc):
    trd = DeepLOBDataset(tr_idx, tr_lbl, X_sc)
    ted = DeepLOBDataset(te_idx, te_lbl, X_sc)
    trl = DataLoader(trd, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    tel = DataLoader(ted, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    return trl, tel

def train_and_eval(trl, tel, n_classes, class_weights, tag, class_names):
    model     = DeepLOB(n_classes=n_classes).to(DEVICE)
    weights   = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=3, factor=0.5)

    n_params  = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {n_params:,}")
    log(f"  {'Epoch':>5} | {'Loss':>8} | {'Train Acc':>9} | {'Test Acc':>9}")
    log(f"  {'-'*46}")

    best_acc   = 0.0
    best_state = None
    best_preds = None
    best_true  = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total_n = 0.0, 0, 0

        for batch_x, batch_y in trl:
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
        scheduler.step(avg_loss)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in tel:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                p = model(batch_x).argmax(1).cpu().numpy()
                preds.extend(p)
                trues.extend(batch_y.numpy())

        test_acc = accuracy_score(trues, preds)
        log(f"  {epoch:>5} | {avg_loss:>8.4f} | {train_acc:>9.3f} | {test_acc:>9.3f}")

        if test_acc > best_acc:
            best_acc   = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = preds[:]
            best_true  = trues[:]

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{SAVE_DIR}/deeplob_{tag}.pt")
    report = classification_report(best_true, best_preds, target_names=class_names)
    log(f"  Best test acc: {best_acc:.3f} — saved deeplob_{tag}.pt")
    log("\n" + report)
    return best_acc, report


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
log("="*65)
log("DeepLOB EXPERIMENT SUITE")
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

for col in BID_P: df[col + "_dist"] = df["mid_price"] - df[col]
for col in ASK_P: df[col + "_dist"] = df[col] - df["mid_price"]

# LOB features arranged for CNN: per level [bid_p_dist, bid_v, ask_p_dist, ask_v]
LOB_FEATURES = []
for i in range(20):
    LOB_FEATURES.append(f"bid_p_{i:02d}_dist")
    LOB_FEATURES.append(f"bid_v_{i:02d}")
    LOB_FEATURES.append(f"ask_p_{i:02d}_dist")
    LOB_FEATURES.append(f"ask_v_{i:02d}")
# = 80 features: 20 levels × 4

df = df.dropna(subset=LOB_FEATURES).reset_index(drop=True)
X_raw = df[LOB_FEATURES].values.astype(np.float32)
log(f"  {len(LOB_FEATURES)} LOB features (20 levels x 4), {len(df):,} rows")
log(f"  Feature order: [bid_p_dist, bid_vol, ask_p_dist, ask_vol] x 20 levels")

all_results = []

# =============================================================================
# EXPERIMENT A — Binary Up/Down  (moved rows)
# =============================================================================
log("\n" + "="*65)
log("GROUP A: Binary Up/Down (moved rows only)")
log("="*65)

for horizon_ticks in [10, 50]:
    horizon_sec = horizon_ticks // 10
    tag = f"updown_{horizon_sec}s"
    log(f"\n--- A: {horizon_sec}s horizon ---")

    price_change = df["mid_price"].shift(-horizon_ticks) - df["mid_price"]
    label_raw = np.where(price_change >  THRESHOLD,  1,
                np.where(price_change < -THRESHOLD,  0, -1))

    valid   = np.arange(SEQ_LEN - 1, len(df) - horizon_ticks)
    mask    = label_raw[valid] != -1
    moved   = valid[mask]
    lbls    = label_raw[moved].astype(np.int64)
    n_down  = (lbls == 0).sum()
    n_up    = (lbls == 1).sum()
    log(f"  Moved rows: {len(lbls):,}  |  Down: {n_down:,} ({100*n_down/len(lbls):.1f}%)  Up: {n_up:,} ({100*n_up/len(lbls):.1f}%)")

    tr_idx, te_idx, tr_lbl, te_lbl, scaler, X_sc = make_split(moved, lbls, X_raw)
    joblib.dump(scaler, f"{SAVE_DIR}/scaler_deeplob_{tag}.pkl")
    trl, tel = make_loaders(tr_idx, te_idx, tr_lbl, te_lbl, X_sc)
    cw = class_weights_uniform([n_down, n_up])

    best_acc, report = train_and_eval(trl, tel, 2, cw, tag, ["Down", "Up"])
    all_results.append(("A", tag, best_acc, report))

# =============================================================================
# EXPERIMENT B — 3-class Up/Flat/Down
# =============================================================================
log("\n" + "="*65)
log("GROUP B: 3-class Up/Flat/Down (all rows)")
log("="*65)

for horizon_ticks in [50, 100]:
    horizon_sec = horizon_ticks // 10
    tag = f"3class_{horizon_sec}s"
    log(f"\n--- B: {horizon_sec}s horizon ---")

    price_change = df["mid_price"].shift(-horizon_ticks) - df["mid_price"]
    label = np.where(price_change >  THRESHOLD, 2,
            np.where(price_change < -THRESHOLD, 0, 1))

    valid  = np.arange(SEQ_LEN - 1, len(df) - horizon_ticks)
    lbls   = label[valid].astype(np.int64)
    n_down = (lbls == 0).sum()
    n_flat = (lbls == 1).sum()
    n_up   = (lbls == 2).sum()
    log(f"  Down: {n_down:,} ({100*n_down/len(lbls):.1f}%)  Flat: {n_flat:,} ({100*n_flat/len(lbls):.1f}%)  Up: {n_up:,} ({100*n_up/len(lbls):.1f}%)")

    tr_idx, te_idx, tr_lbl, te_lbl, scaler, X_sc = make_split(valid, lbls, X_raw)
    joblib.dump(scaler, f"{SAVE_DIR}/scaler_deeplob_{tag}.pkl")
    trl, tel = make_loaders(tr_idx, te_idx, tr_lbl, te_lbl, X_sc)
    cw = class_weights_uniform([n_down, n_flat, n_up])

    best_acc, report = train_and_eval(trl, tel, 3, cw, tag, ["Down", "Flat", "Up"])
    all_results.append(("B", tag, best_acc, report))

# =============================================================================
# SAVE RESULTS
# =============================================================================
log("\n" + "="*65)
log("ALL RESULTS SUMMARY")
log("="*65)

lines = []
lines.append("DeepLOB EXPERIMENT RESULTS")
lines.append(f"SEQ_LEN={SEQ_LEN}  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}")
lines.append(f"Device: {DEVICE}")
lines.append("")
lines.append("Architecture: CNN(1x2 x2, 1x20) + Inception(4 branches) + LSTM(128->64) + FC")
lines.append("Features: 80 LOB features = 20 levels x [bid_p_dist, bid_vol, ask_p_dist, ask_vol]")
lines.append("")
lines.append("LSTM baselines:")
lines.append("  Binary Up/Down 1s : 77.5%")
lines.append("  Binary Up/Down 5s : 68.2%")
lines.append("  3-class 5s        : 53.1%")
lines.append("  3-class 10s       : 51.5%")
lines.append("")
lines.append("="*65)
lines.append(f"{'Group':<8} {'Experiment':<20} {'Best Test Acc':>13}")
lines.append("-"*45)

for grp, tag, acc, _ in all_results:
    lines.append(f"{grp:<8} {tag:<20} {acc:>12.1%}")
    log(f"  {grp}  {tag:<20}  {acc:.1%}")

lines.append("")
lines.append("="*65)
lines.append("DETAILED REPORTS")
for grp, tag, acc, report in all_results:
    lines.append(f"\n[{grp}] {tag}  (best acc: {acc:.1%})")
    lines.append(report)

with open(RESULTS_FILE, "w") as f:
    f.write("\n".join(lines))

log(f"\nResults saved to: results_deeplob.txt")
log("Done.")
