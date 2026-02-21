import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def log(msg): print(msg, flush=True)

# --- Config ---
SEQ_LEN    = 10    # 1 second of history (best from step7)
THRESHOLD  = 0.5
HORIZONS   = [50, 100]   # 5 seconds and 10 seconds
DATA_PATH  = "C:/Users/marou/Documents/ClaudCode1/parquet_01/*.parquet"
BATCH_SIZE = 1024
EPOCHS     = 10
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Load data once ---
log("\nLoading data...")
files = sorted(glob.glob(DATA_PATH))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.sort_values("event_ts").reset_index(drop=True)
log(f"  {len(df):,} rows loaded")

# --- Build features once ---
log("Building features...")
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
X_raw = df[FEATURES].values.astype(np.float32)
log(f"  {len(FEATURES)} features, {len(df):,} rows")

# --- Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=82, hidden_size=128, num_layers=2, dropout=0.3, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- Dataset ---
class LOBDataset(Dataset):
    def __init__(self, indices, labels, X_scaled):
        self.indices  = indices
        self.labels   = labels
        self.X_scaled = X_scaled

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i   = self.indices[k]
        seq = self.X_scaled[i - SEQ_LEN + 1 : i + 1]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.labels[k], dtype=torch.long)

# --- Run for each horizon ---
results_summary = []

for HORIZON in HORIZONS:
    horizon_sec = HORIZON // 10
    log(f"\n{'='*65}")
    log(f"HORIZON = {HORIZON} ticks ({horizon_sec} seconds)")
    log(f"{'='*65}")

    # Labels: 0=Down, 1=Flat, 2=Up — ALL rows (3 classes)
    price_change = df["mid_price"].shift(-HORIZON) - df["mid_price"]
    label = np.where(price_change >  THRESHOLD,  2,
            np.where(price_change < -THRESHOLD,  0, 1))  # 2=Up,1=Flat,0=Down

    # Valid range: enough history AND enough future
    valid_range = np.arange(SEQ_LEN - 1, len(df) - HORIZON)
    all_labels  = label[valid_range].astype(np.int64)
    all_indices = valid_range

    n_down = (all_labels == 0).sum()
    n_flat = (all_labels == 1).sum()
    n_up   = (all_labels == 2).sum()
    total  = len(all_labels)
    log(f"  Down: {n_down:,} ({100*n_down/total:.1f}%) | Flat: {n_flat:,} ({100*n_flat/total:.1f}%) | Up: {n_up:,} ({100*n_up/total:.1f}%)")

    # Chronological 80/20 split
    split      = int(len(all_indices) * 0.8)
    train_idx  = all_indices[:split]
    test_idx   = all_indices[split:]
    train_lbl  = all_labels[:split]
    test_lbl   = all_labels[split:]
    log(f"  Train: {len(train_idx):,} | Test: {len(test_idx):,}")

    # Scaler: fit on training period only
    train_end = train_idx[-1] + 1
    scaler    = StandardScaler()
    scaler.fit(X_raw[:train_end])
    X_scaled  = scaler.transform(X_raw).astype(np.float32)

    # Datasets and loaders
    train_ds = LOBDataset(train_idx, train_lbl, X_scaled)
    test_ds  = LOBDataset(test_idx,  test_lbl,  X_scaled)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = LSTMModel(n_classes=3).to(DEVICE)

    # Class weights
    w_down = total / (3 * n_down)
    w_flat = total / (3 * n_flat)
    w_up   = total / (3 * n_up)
    weights   = torch.tensor([w_down, w_flat, w_up], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    log(f"\n  Training {EPOCHS} epochs...")
    log(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>9}")
    log(f"  {'-'*50}")

    best_test_acc = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total_n = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            output = model(batch_x)
            loss   = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            pred        = output.argmax(dim=1)
            correct    += (pred == batch_y).sum().item()
            total_n    += len(batch_y)

        train_acc = correct / total_n
        avg_loss  = total_loss / total_n

        # Eval
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                pred = model(batch_x).argmax(dim=1).cpu().numpy()
                all_preds.extend(pred)
                all_true.extend(batch_y.numpy())

        test_acc = accuracy_score(all_true, all_preds)
        best_test_acc = max(best_test_acc, test_acc)
        log(f"  {epoch:>5} | {avg_loss:>10.4f} | {train_acc:>9.3f} | {test_acc:>9.3f}")

    log(f"\n  Classification report (test set, epoch {EPOCHS}):")
    log(classification_report(all_true, all_preds, target_names=["Down", "Flat", "Up"]))

    results_summary.append((horizon_sec, best_test_acc))

# --- Final summary ---
log("\n" + "="*65)
log("SUMMARY")
log("="*65)
log(f"  {'Horizon':>8} | {'Best Test Acc':>13}")
log(f"  {'-'*30}")
for h, acc in results_summary:
    log(f"  {h:>5}s     | {acc:>12.1%}")
log("")
log("  Baseline (logistic regression, binary 1s): 77%")
