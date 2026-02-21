import pandas as pd
import numpy as np
import glob
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def log(msg): print(msg, flush=True)

# --- Config ---
SEQ_LEN   = 50    # 5 seconds of history (50 snapshots @ 100ms)
HORIZON   = 10    # predict 1 second ahead
THRESHOLD = 0.5
DATA_PATH = "C:/Users/marou/Documents/ClaudCode1/parquet_01/*.parquet"
BATCH_SIZE = 1024
EPOCHS     = 10
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Load data ---
log("\nStep 1 — Loading data...")
files = sorted(glob.glob(DATA_PATH))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df = df.sort_values("event_ts").reset_index(drop=True)
log(f"  {len(df):,} rows loaded")

# --- Build features ---
log("\nStep 2 — Building features...")
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
log(f"  {len(FEATURES)} features, {len(df):,} rows after dropna")

# --- Build labels ---
log("\nStep 3 — Building labels...")
price_change = df["mid_price"].shift(-HORIZON) - df["mid_price"]
label = np.where(price_change >  THRESHOLD,  1,
        np.where(price_change < -THRESHOLD,  0, -1))  # 1=Up, 0=Down, -1=Flat

# Valid range: enough history for sequence AND enough future for label
valid_range  = np.arange(SEQ_LEN - 1, len(df) - HORIZON)
moved_mask   = label[valid_range] != -1
moved_indices = valid_range[moved_mask]
moved_labels  = label[moved_indices]

n_up   = (moved_labels == 1).sum()
n_down = (moved_labels == 0).sum()
log(f"  Moved rows: {len(moved_indices):,}")
log(f"  Up: {n_up:,} ({100*n_up/len(moved_indices):.1f}%) | Down: {n_down:,} ({100*n_down/len(moved_indices):.1f}%)")

# --- Extract feature array ---
X_all = df[FEATURES].values.astype(np.float32)

# --- Chronological train/test split (80/20) ---
split     = int(len(moved_indices) * 0.8)
train_idx = moved_indices[:split]
test_idx  = moved_indices[split:]
train_lbl = moved_labels[:split].astype(np.int64)
test_lbl  = moved_labels[split:].astype(np.int64)
log(f"  Train: {len(train_idx):,} | Test: {len(test_idx):,}")

# --- Scale features (fit on training period rows only) ---
log("\nStep 4 — Fitting scaler on training data only...")
train_end = train_idx[-1] + 1   # last row index in training period
scaler = StandardScaler()
scaler.fit(X_all[:train_end])   # only rows up to train boundary
X_scaled = scaler.transform(X_all).astype(np.float32)
log("  Scaler fitted and applied")

# --- Dataset ---
class LOBDataset(Dataset):
    def __init__(self, indices, labels):
        self.indices = indices
        self.labels  = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i   = self.indices[k]
        seq = X_scaled[i - SEQ_LEN + 1 : i + 1]   # shape (SEQ_LEN, 82)
        lbl = self.labels[k]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)

train_ds = LOBDataset(train_idx, train_lbl)
test_ds  = LOBDataset(test_idx,  test_lbl)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# --- Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=82, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # use last time step output

model = LSTMModel().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
log(f"\nModel: LSTM(82 -> 128 x 2 layers -> 2)  |  {n_params:,} parameters")

# --- Class weights (handle imbalance) ---
w_down = len(train_lbl) / (2 * (train_lbl == 0).sum())
w_up   = len(train_lbl) / (2 * (train_lbl == 1).sum())
weights   = torch.tensor([w_down, w_up], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Train ---
log(f"\nTraining for {EPOCHS} epochs on {DEVICE}...")
log("-" * 65)
log(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>9}")
log("-" * 65)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

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
        total      += len(batch_y)

    train_acc = correct / total
    avg_loss  = total_loss / total

    # --- Evaluate ---
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            out  = model(batch_x)
            pred = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_true.extend(batch_y.numpy())

    test_acc = accuracy_score(all_true, all_preds)
    log(f"{epoch:>6} | {avg_loss:>10.4f} | {train_acc:>9.3f} | {test_acc:>9.3f}")

log("-" * 65)
log("\nFinal classification report (test set):")
log(classification_report(all_true, all_preds, target_names=["Down", "Up"]))

# --- Save ---
torch.save(model.state_dict(), "C:/Users/marou/Documents/ClaudCode1/model_lstm.pt")
joblib.dump(scaler, "C:/Users/marou/Documents/ClaudCode1/scaler_lstm.pkl")
log("Saved: model_lstm.pt, scaler_lstm.pkl")
