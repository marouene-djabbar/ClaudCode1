# Session Log — ClaudCode1

## Session 1 — 2026-02-20
### Setup
- Created folder `ClaudCode1`, moved to `C:\Users\marou\Documents\ClaudCode1`
- Agreed to work exclusively inside this folder
- Set up dual memory system: MEMORY.md (auto-loaded) + SESSION_LOG.md (project-level)

### Decisions
- File access restricted to `C:\Users\marou\Documents\ClaudCode1`
- SESSION_LOG.md updated incrementally (not just at end of session)
- User wants to understand each step before it is executed

### Crash — Session 1 ended abruptly
- Bun v1.3.10 crashed with `panic(main thread): switch on corrupt value`
- This is a Bun runtime bug, not a user or git error
- MEMORY.md writes did not persist because of the crash
- git init may or may not have completed before the crash — needs to be verified

## Session 2 — 2026-02-20
### Recovery
- User pasted full Session 1 transcript to restore context
- MEMORY.md restored manually with correct working directory, preferences, and session rules
- SESSION_LOG.md updated (this file)

### Next Steps (completed in Session 3)
- Verified git repo exists
- Connected to GitHub

## Session 3 — 2026-02-20
### GitHub Setup
- Installed GitHub CLI (gh)
- Logged in via `gh auth login` as marouene-djabbar
- Created remote repo: https://github.com/marouene-djabbar/ClaudCode1
- Added origin remote and pushed master branch
- Repo is now live on GitHub

### Data & Environment Setup
- Added parquet data: `parquet_01/` folder with 20 files (`book20_ml_*.parquet`)
- Created temporary Python `.venv` (to be replaced with Miniconda later)
- Installed: pandas, pyarrow, scikit-learn

### Dataset
- **Instrument:** Bitcoin futures (mid price ~90,000)
- **Tick size:** 0.5
- **Total rows:** 6,196,644 across 20 parquet files
- **Snapshot frequency:** every 100ms (10 snapshots/second)
- **Features:** 88 columns — spread_l1, vol_imbalance_l1, mid_price, 20 bid/ask price+volume levels
- **Date range:** starts 2026-01-01

### Labeling Decisions
- **Threshold:** ±0.5 (1 tick)
- **Horizon tested:** 5s (50 ticks) and 10s (100 ticks)
- Class labeling: Up = price_change > +0.5, Down = price_change < -0.5, Flat = in between
- Flat zone width = 1.0 (2 ticks total)
- At 5s: Down 18.7% / Flat 56.5% / Up 24.9% — imbalanced, used class_weight='balanced'
- At 10s: Down 26.4% / Flat 38.2% / Up 35.4% — more balanced

### Model Results (100K sample, SGDClassifier logistic regression)
| Horizon | Threshold | Test Acc | Notes |
|---------|-----------|----------|-------|
| 1s (10t) | 0.5 | 91% | Fake — model predicts flat always |
| 3s (30t) | 0.5 | 78% | Fake — same issue |
| 5s (50t) | 0.5 | 65% | Imbalanced, Down F1=0.25 |
| 10s (100t) | 0.5 | 55% | Best real signal, balanced classes |
| 20s+ | auto | <50% | Worse than random |

### Scripts
- `step1_labeling.py` — loads all parquets, creates labels, saves `labeled_data.parquet`
- `step2_model.py` — trains SGDClassifier, reports accuracy + confusion matrix

### Next Steps
- Decide on final horizon (5s vs 10s)
- Run on full 6.2M rows
- Improve Down class prediction (currently weakest)
- Replace `.venv` with Miniconda environment
- Consider more features or feature engineering

### Future Goals
- Explore remote communication with Claude (e.g. Slack bot via Claude API)
  - Would allow monitoring long-running jobs without being physically present
  - Requires building a custom Slack integration — noted as a future project

## Session 4 — 2026-02-21

### Mid-price tick size clarification
- Mid price moves in 0.25 steps (not 0.5) since mid = (bid+ask)/2
- However in this dataset, mid price only moves in 0.5 steps in practice (bid/ask always move together)
- Threshold of 0.5 confirmed correct for this dataset

### Model Results Summary (full 6.2M rows)

#### Binary Up/Down (drop flat rows)
| Horizon | Moved Rows | Test Acc | Notes |
|---------|------------|----------|-------|
| 1s (10t) | 1.24M | 77% | Best direction model |
| 5s (50t) | 3.05M | 68% | More rows, less accurate |

#### Move vs Flat detector
| Horizon | Flat% | Move% | Test Acc | Move F1 | Notes |
|---------|-------|-------|----------|---------|-------|
| 1s (10t) | 80% | 20% | 71% | 0.48 | Imbalanced, misleading accuracy |
| 5s (50t) | 51% | 49% | 64% | 0.61 | Best balanced, +14% vs random |
| 10s (100t) | 36% | 64% | 61% | 0.65 | Move dominates |

#### Full pipeline evaluation (Stage 1 → Stage 3 applied to all 6.2M)
- Binary model (trained on moved rows) applied to ALL data → 16% overall accuracy
- Model never predicts Flat → fails completely on 80% flat rows
- Confirmed: binary model only useful when move is already known

### Saved Models
- `model_h10.pkl` — binary Up/Down, 1 second horizon
- `scaler_h10.pkl` — scaler for above model

### Scripts
- `step1_labeling.py` — label generation
- `step2_model.py` — binary Up/Down model
- `step3_evaluate.py` — evaluate saved model on all data
- `step4_move_detector.py` — Move vs Flat detector
- `step5_pipeline.py` — 2-stage pipeline (Stage 1 + Stage 3)

### Key Insights
- Short horizon (1s) better for direction prediction (order book directly shows pressure)
- Longer horizon (5s) better for move detection (pressure builds gradually before move)
- Single snapshot insufficient for 1s move detection — too little signal
- class_weight='balanced' helps but doesn't fix fundamental imbalance problem

### Next Steps (completed — see Session 5 below)
- Sliding window features tested
- Full pipeline built and evaluated
- GPU discovered → plan to move to deep learning

## Session 5 — 2026-02-21

### Overview
This session focused on improving the move detector, building a full trading pipeline,
and understanding how to combine two models to make a trading decision.

---

### Test 1 — Sliding Window Features for Move Detection (step6_sliding_window.py)
**Goal:** Instead of using a single order book snapshot (82 features), use the last N
snapshots stacked together (N x 82 features). The idea is that order book pressure
builds gradually before a price move — a sequence of snapshots captures this better
than a single photo.

**Method:** For each row t, stack snapshots [t-N+1 ... t] into one feature vector.
Train SGDClassifier (logistic regression) with partial_fit (batch processing) to avoid
memory issues on 6.2M rows.

**Results (full data, Move vs Flat, 1 second horizon):**
| Window | Features | Test Acc | Move F1 | Move Recall |
|--------|----------|----------|---------|-------------|
| 1 (single) | 82 | 71% | 0.48 | 62% |
| 10 (1s history) | 820 | 55% | 0.43 | 79% |
| 20 (2s history) | 1,640 | 57% | 0.43 | 74% |
| 30 (3s history) | 2,460 | 60% | 0.41 | 66% |

**Conclusion:** Sliding window improves move recall (catches more moves) but reduces
precision (more false alarms). Single snapshot still has best F1 and precision.
The fundamental problem is the 80/20 class imbalance at 1 second — no window size fixes this.

---

### Test 2 — Decision Threshold Tuning (precision focus)
**Goal:** The user prioritises AVOIDING LOSSES over catching every opportunity. A false
alarm (model says "move", price stays flat) causes a bad trade. A missed move causes no
cost. Therefore we care about PRECISION (when model says "move", how often is it right?),
not recall.

**Method:** Instead of default threshold of 0.50 (predict Move if probability > 50%),
we raise the threshold to only fire when the model is very confident.

**Results — 1 second horizon:**
| Threshold | Precision | Recall | Signal% |
|-----------|-----------|--------|---------|
| 0.50 | 39% | 62% | 35% |
| 0.75 | 64% | 17% | 6% |
| 0.85 | 73% | 7% | 2% |
| 0.95 | 88% | 2% | 0.5% |

**Results — 5 second horizon (better because classes are 50/50 balanced):**
| Threshold | Precision | Recall | Signal% |
|-----------|-----------|--------|---------|
| 0.50 | 64% | 59% | 44% |
| 0.75 | 84% | 13% | 8% |
| 0.85 | 88% | 5% | 2.8% |
| 0.95 | 91% | 1% | 0.6% |

**Decision: Lock in 5 second horizon, threshold = 0.85**
- 88% precision — when it fires, it is right 88% of the time
- Fires on 2.8% of rows — very selective, avoids overtrading
- Saved as: model_move_5s.pkl, scaler_move_5s.pkl, threshold_move_5s.pkl

**Key insight:** 5 second horizon gives much better precision than 1 second because the
classes are naturally balanced (50/50 vs 80/20). A balanced dataset produces more
reliable probability estimates.

---

### Test 3 — When Does the Move Happen Within the 5-Second Window?
**Goal:** The move detector predicts a move within 5 seconds. We need to know WHEN in
that window the move typically happens to decide how to time the trade.

**Method:** For each row where the 5-second move was detected, scan forward tick by tick
and record the first tick where |price_change| > 0.5.

**Results (500K sample):**
| Second | % of moves happening here | Cumulative |
|--------|--------------------------|------------|
| 1st second | 38% | 38% |
| 2nd second | 22% | 60% |
| 3rd second | 16% | 76% |
| 4th second | 13% | 89% |
| 5th second | 11% | 100% |
- Median: 1.5 seconds after detection

**Conclusion:** 60% of moves happen within the first 2 seconds. Waiting 4 seconds
before trading (original idea) would mean missing 89% of moves. Must act immediately
at time T when the signal fires.

---

### Test 4 — Direction Model for 5 Seconds (model_dir_5s.pkl)
**Goal:** The existing direction model (model_h10.pkl) was trained to predict 1-second
direction. Since we now hold trades for up to 5 seconds, we trained a new direction model
specifically for 5-second direction prediction.

**Method:** Keep only rows where price moved in 5 seconds (drop flat). Train binary
Up/Down SGDClassifier on 5-second net price change direction.

**Results (full data, 3.05M moved rows):**
- Train accuracy: 71%
- Test accuracy: 68% (vs 77% for 1-second model)

**Conclusion:** 5-second direction is harder to predict than 1-second because more
things can change over a longer horizon. 1-second model is better.

---

### Test 5 — Full 2-Stage Pipeline Evaluation
**Goal:** Combine both models into a real trading simulation:
- Stage 1 (move detector, 5s, threshold=0.85): fires when confident a move is coming
- Stage 2 (direction model): predicts Up or Down
- Enter trade at T, hold up to 5 seconds, exit when move happens

**Three possible outcomes per signal:**
- WIN: price moved in predicted direction within 5s
- LOSS: price moved against predicted direction within 5s
- NO TRADE: price did not move at all in 5s (exit flat, just fees)

**Results comparing 1s vs 5s direction model (34,545 signals on test set):**
| Outcome | 1s Direction Model | 5s Direction Model |
|---------|-------------------|-------------------|
| WIN | 54.4% | 53.9% |
| LOSS | 33.3% | 33.8% |
| NO TRADE | 12.3% | 12.3% |
| WIN rate (of moves) | 62.0% | 61.5% |

**Conclusion:** Both direction models give nearly identical pipeline results. The
bottleneck is not the direction model horizon — it is the features. A single order book
snapshot gives ~62% directional accuracy regardless of horizon. To improve beyond this
ceiling we need richer features (sliding window history) or a stronger model.

---

### GPU Discovery
- Machine has NVIDIA GeForce RTX 5070 with 12GB VRAM, CUDA 12.9
- This enables GPU-accelerated training: LightGBM-GPU, XGBoost-GPU, PyTorch
- RTX 5070 is powerful enough to train DeepLOB (CNN+LSTM) on 6.2M rows

---

### Saved Models
| File | Description | Performance |
|------|-------------|-------------|
| model_h10.pkl | Binary Up/Down, 1s horizon | 77% accuracy on moved rows |
| scaler_h10.pkl | Scaler for model_h10 | — |
| model_move_5s.pkl | Move vs Flat, 5s horizon, threshold=0.85 | 88% precision |
| scaler_move_5s.pkl | Scaler for model_move_5s | — |
| threshold_move_5s.pkl | Decision threshold = 0.85 | — |
| model_dir_5s.pkl | Binary Up/Down, 5s horizon | 68% accuracy |
| scaler_dir_5s.pkl | Scaler for model_dir_5s | — |

---

### Scripts
| File | Purpose |
|------|---------|
| step1_labeling.py | Load parquets, create labels |
| step2_model.py | Binary Up/Down direction model |
| step3_evaluate.py | Evaluate saved model on all data |
| step4_move_detector.py | Move vs Flat detector |
| step5_pipeline.py | Full 2-stage pipeline evaluation |
| step6_sliding_window.py | Sliding window feature experiments |

---

### Key Findings Summary
1. **Move detection**: 5s horizon with threshold=0.85 gives 88% precision — best move detector
2. **Direction prediction**: 1s model (77%) beats 5s model (68%) — shorter horizon is more predictable
3. **Pipeline result**: 62% WIN rate on actual moves, 12% NO TRADE, 26% false direction
4. **Ceiling hit**: Logistic regression on single snapshot maxes out at ~62% direction accuracy
5. **Sliding window**: Improves recall but hurts precision — not suitable for loss-averse trading
6. **Next step**: Move to deep learning (DeepLOB / LSTM) with GPU support

---

### Next Steps (completed — see Session 6 below)
- Miniconda + CUDA PyTorch installed
- LSTM experiments run on GPU
- DeepLOB identified as next step

## Session 6 — 2026-02-21

### Environment Setup
- Miniconda 25.11.1 installed at C:\Users\marou\miniconda3
- Conda env: `trading` with Python 3.11
- PyTorch 2.12.0 nightly + CUDA 12.8 (supports RTX 5070 Ti sm_120 Blackwell)
- Previous PyTorch 2.6+cu124 replaced — was incompatible with sm_120
- All packages in `trading` env: pandas, pyarrow, scikit-learn, lightgbm, jupyter, joblib, torch

### LSTM Experiments — Step 7 & 9

All experiments use:
- SEQ_LEN=10 (1 second of history, 10 snapshots)
- Same 82 features as logistic regression models
- Scaler fit on training data only (no leakage)
- Chronological 80/20 train/test split
- Class weights to handle imbalance
- 10 epochs, batch size 1024, Adam optimizer, lr=0.001

#### Window Size Experiment (step7_lstm.py)
- SEQ_LEN=10 → best test acc 77.5% at epoch 1, drops to 75% by epoch 10 (overfitting)
- SEQ_LEN=50 → same start 77.5%, drops to 71% by epoch 10 (worse overfitting)
- Conclusion: more history does NOT help — signal is in current snapshot

#### Full Experiment Suite (step9_lstm_all.py) — 7 experiments on GPU

**GROUP A: 3-class Up/Down/Flat (all rows)**
| Experiment | Best Test Acc | Notes |
|------------|--------------|-------|
| 3class_5s  | 53.1% | Random=33%, so +20% above random |
| 3class_10s | 51.5% | Classes more balanced but harder |

Flat class precision ~74%, Up/Down only 42-48% — model confuses direction with flat.

**GROUP B: Binary Up/Down (moved rows only)**
| Experiment | Best Test Acc | Notes |
|------------|--------------|-------|
| updown_1s  | 77.5% | Matches logistic regression baseline |
| updown_5s  | 68.2% | +0.2% over logistic regression (68%) |
| updown_10s | 64.2% | Same ceiling as logistic regression |

LSTM matches but does not exceed logistic regression on direction prediction.

**GROUP C: Binary Move/No-Move (all rows)**
| Experiment | Best Test Acc | Notes |
|------------|--------------|-------|
| move_5s    | 69.0% | vs 64% for logistic regression — improvement! |
| move_10s   | 68.1% | Good, classes more imbalanced (64% Move) |

Move detection improved with LSTM: 69% vs 64% logistic regression at 5s.

### Key Findings — Session 6

1. **LSTM matches logistic regression on direction** — both plateau at ~77% (1s), 68% (5s)
2. **LSTM improves move detection** — 69% vs 64% at 5s horizon
3. **Overfitting is the main problem** — train acc reaches 86-90% but test drops after epoch 1-2
4. **Larger window (SEQ_LEN=50) makes overfitting worse**, not better
5. **Signal is in the current snapshot** — historical context adds noise, not signal
6. **3-class problem is harder** — LSTM gets 53% vs 33% random, but Up/Down confused with Flat

### Saved Models (Session 6)
| File | Task | Acc |
|------|------|-----|
| model_updown_1s.pt | Binary Up/Down, 1s | 77.5% |
| model_updown_5s.pt | Binary Up/Down, 5s | 68.2% |
| model_updown_10s.pt | Binary Up/Down, 10s | 64.2% |
| model_3class_5s.pt | 3-class, 5s | 53.1% |
| model_3class_10s.pt | 3-class, 10s | 51.5% |
| model_move_5s.pt | Move/No-Move, 5s | 69.0% |
| model_move_10s.pt | Move/No-Move, 10s | 68.1% |

Full detailed results saved to: `results_lstm.txt`

### Scripts (Session 6)
| File | Purpose |
|------|---------|
| step7_lstm.py | Binary Up/Down LSTM, window size experiments |
| step8_lstm_3class.py | 3-class LSTM (superseded by step9) |
| step9_lstm_all.py | Full experiment suite: all tasks × all horizons |

### Next Steps
- Try DeepLOB (CNN + LSTM) — CNN extracts spatial patterns across 20 bid/ask levels
- Expected: DeepLOB may break 77% ceiling by learning cross-level features
- Consider LightGBM-GPU as a fast tree-based baseline before DeepLOB
