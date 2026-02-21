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

### Next Steps
- Implement sliding window features (Option 5):
  - Use last 10 snapshots (1 second history) as features instead of single snapshot
  - 10 snapshots x 82 features = 820 features per row
  - Same logistic regression model, much richer features
  - Expected to improve Move detection at 1 second
- After sliding window: revisit full 2-stage pipeline
- Then move to LightGBM for further improvement
