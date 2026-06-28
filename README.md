# MLB First-Pitch Outcome Prediction

A machine learning system for predicting the outcome of the **first pitch** of Major League Baseball at-bats — before any pitch is thrown. Models classify first-pitch outcomes as either **hit into play vs. not in play** (two-class) or **ball / strike / hit into play** (three-class), using only information available prior to the start of each at-bat.

> **Why first pitches?** The first pitch (0 balls, 0 strikes) is the cleanest signal in baseball: the pitcher sets the tone of the at-bat, and the batter decides whether to swing or take with no count leverage on either side. First pitches have well-defined strategic patterns driven by pitcher tendencies and batter approach — patterns that can be learned from historical data without leaking any pitch-sequence information.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Design Decisions](#key-design-decisions)
3. [Features](#features)
4. [Models](#models)
5. [File Structure](#file-structure)
6. [Quickstart](#quickstart)
7. [Notebook Descriptions](#notebook-descriptions)
8. [Script Descriptions](#script-descriptions)
9. [Results](#results)
10. [Requirements](#requirements)

---

## Project Overview

The system is built around three progressively broader modeling architectures:

| Architecture | Description |
|---|---|
| **Single-Batter** | One model per batter, trained on that batter's first-pitch history only |
| **Multi-Batter** | One model per batter from a pool of 300 top hitters, evaluated in a unified loop |
| **Global** | One model trained across all 300 batters simultaneously, with `batter_id` as a native categorical feature |

Each architecture is implemented in both **two-class** (hit into play vs. not) and **three-class** (ball / strike / hit into play) variants. Six classifiers are compared across all architectures: Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, and ANN.

**Best result**: The global CatBoost model achieved the highest F1 score for predicting first-pitch hit-into-play events.

---

## Key Design Decisions

### Strictly Predictive Features Only

All pitch outcome features (launch angle, spin rate, plate location, exit velocity, etc.) are excluded. The models use only what a manager, analyst, or betting market participant would know *before* the pitch is thrown: pitcher tendencies, batter approach, count situation, base runners, and game context.

### Temporal Decay Weighting

Training rows are weighted by recency using exponential decay:

```
weight = exp(-DECAY_RATE × days_before_TRAIN_END)
```

With the default `DECAY_RATE = 0.004`, at-bats from 6 months prior receive approximately half the weight of the most recent at-bats (`exp(-0.004 × 180) ≈ 0.49`). This allows the model to train on a long history while biasing learning toward recent tendencies — particularly important for pitchers and batters who evolve their approach across seasons.

### SMOTE for Class Imbalance

First-pitch hit-into-play events are the minority class (roughly 10–15% of first pitches). SMOTE (Synthetic Minority Oversampling Technique) is applied to the training set to balance class distributions before model fitting. Decay weights are propagated through SMOTE: original rows retain their computed weights and synthetic minority rows receive the mean weight of the real minority-class rows they were interpolated from.

### Sidecar Context Features

Sequential in-game and prior-game context features are precomputed once across each Statcast parquet file and stored as a separate sidecar parquet. This avoids recomputation on every notebook run and ensures the sequential walk-through of each game's pitch order is only performed once.

### Leakage Prevention

The validation window is always fully held out — no data from the validation period is used to compute any feature, fit any encoder, or tune any hyperparameter. Pitcher statistics are always computed from the training window only.

---

## Features

### Game Situation (known pre-pitch)

| Feature | Description |
|---|---|
| `stand` | Batter handedness (L/R), or inferred from pitcher hand for switch hitters |
| `p_throws` | Pitcher handedness (L/R) |
| `outs_when_up` | Outs at the start of the at-bat (0/1/2) |
| `on_1b`, `on_2b`, `on_3b` | Base-runner flags (binarised) |
| `at_bat_number` | Position of this at-bat in the game |
| `bat_win_exp` | Batting team win expectancy at the moment of the pitch |
| `inning` | Inning number |
| `inning_topbot` | Top or bottom of the inning |

### Pitcher Tendency Features (computed from training-window first pitches)

| Feature | Description |
|---|---|
| `strike_percent` | Fraction of first pitches that result in a strike |
| `swing_percent_on_strikes` | Fraction of strikes that are swung at |
| `contact_percent_on_strikes` | Fraction of strikes that result in contact |
| `in_play_percent_on_strikes` | Fraction of strikes put in play |
| `FF_percent`, `SL_percent`, ... | First-pitch usage rate for each of 19 pitch types |

### Sequential Context Features (from `build_context_features.py`)

| Feature | Description |
|---|---|
| `pitcher_pitch_count_in_game` | Pitches thrown by the pitcher in the current game before this pitch |
| `batter_prior_hip_count_in_game` | First-pitch HIPs by this batter earlier in the current game |
| `pitcher_pitch_count_prior_game` | Total pitches the pitcher threw in their most recent prior game |
| `batter_prior_game_hip_count` | First-pitch HIPs by this batter in their most recent prior game |

### Previous At-Bat Context (computed in-notebook)

| Feature | Description |
|---|---|
| `prev_pitch_result` | Outcome of the preceding at-bat (mapped to: single, field_out, strikeout, walk, extra_base_hit, start_of_game, start_of_inning) |
| `prev_ab_pitch_count` | Number of pitches in the preceding at-bat |

### Global Model Additional Features

| Feature | Description |
|---|---|
| `batter` | MLBAM batter ID as a native CatBoost categorical (no one-hot encoding) |
| `batter_first_pitch_swing_pct` | Batter's historical first-pitch swing rate |
| `batter_first_pitch_contact_pct` | Batter's historical contact rate on first-pitch swings |
| `batter_first_pitch_hip_pct` | Batter's historical first-pitch hit-into-play rate |

---

## Models

### Algorithm Comparison

| Model | Encoding | Class Balancing |
|---|---|---|
| Decision Tree | One-hot | SMOTE + sample weights |
| Random Forest | One-hot | SMOTE + sample weights |
| XGBoost | Native categorical (`enable_categorical=True`) | sample weights |
| LightGBM | Native categorical | `is_unbalance=True` + sample weights |
| CatBoost | Native categorical (ordered target statistics) | `auto_class_weights='Balanced'` + sample weights |
| ANN | One-hot | SMOTE + sample weights |

### Why CatBoost for the Global Model?

The global model trains on 300 batters simultaneously with `batter_id` as a high-cardinality categorical feature (300+ categories). One-hot encoding would add 300+ sparse binary columns, creating a feature matrix where batter identity dwarfs all other signals. CatBoost's ordered target statistics handle high-cardinality categoricals natively — without sparsity, without one-hot expansion, and without requiring SMOTE (class balancing is handled internally). This makes CatBoost the natural choice for the global architecture.

---

## File Structure

```
├── Notebooks
│   ├── First_Pitch_Single_Batter.ipynb          Two-class, single batter
│   ├── First_Pitch_Multi_Batter.ipynb           Two-class, per-batter loop
│   ├── First_Pitch_Global.ipynb                 Two-class, global model
│   ├── First_Pitch_Single_Batter_Multiclass.ipynb   Three-class, single batter
│   ├── First_Pitch_Multi_Batter_Multiclass.ipynb    Three-class, per-batter loop
│   └── First_Pitch_Global_Multiclass.ipynb          Three-class, global model
│
├── Scripts
│   ├── statcast_loader.py          Download and cache Statcast data as parquet
│   ├── build_context_features.py   Compute sequential context features per parquet file
│   ├── train_global_model.py       Train and save global CatBoost model artifacts
│   ├── predict_first_pitch_global.py   Interactive CLI prediction tool
│   └── evaluate_game_global.py     Evaluate model against all first pitches in a game
│
└── Generated (not committed)
    ├── statcast_YYYYMMDD_YYYYMMDD.parquet          Cached Statcast data
    ├── statcast_YYYYMMDD_YYYYMMDD_context_features.parquet   Sidecar context features
    └── global_model/                               Saved model artifacts
        ├── catboost_model/
        ├── label_encoder.pkl
        ├── cat_categories.pkl
        ├── feature_columns.pkl
        ├── batter_features.pkl
        ├── pitcher_features.pkl
        ├── meta.pkl
        └── first_pitch_meta.pkl
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install pybaseball scikit-learn imbalanced-learn catboost xgboost lightgbm tensorflow pandas numpy matplotlib seaborn
```

### 2. Cache Statcast data

The first run downloads data from Baseball Savant and saves it as a parquet file. All subsequent runs load from cache (instant).

```python
from statcast_loader import load_statcast

table = load_statcast('2024-03-28', '2026-04-15')   # training window
```

Or directly:

```bash
python statcast_loader.py
```

### 3. Build context feature sidecars

Run once per parquet file. The sidecar is saved alongside the input file automatically.

```bash
python build_context_features.py statcast_20240328_20260415.parquet
python build_context_features.py statcast_20260416_20260609.parquet
```

### 4. Train the global model

```bash
python train_global_model.py
```

Artifacts are saved to `./global_model/`. Edit `TRAIN_START`, `TRAIN_END`, and `DECAY_RATE` at the top of the file to change the training window.

### 5. Make a prediction

```bash
python predict_first_pitch_global.py
```

The CLI prompts for batter name, pitcher name, and game situation. Batter handedness and pitcher handedness are looked up automatically from historical data. A **grid of predictions** is returned across combinations of previous at-bat result and pitch count — no need to re-run for each scenario.

```
── Batter ─────────────────────────────────────────────────
  Last name : judge
  First name: aaron
  Batter stands: R

── Pitcher ────────────────────────────────────────────────
  Last name : cole
  First name: gerrit
  Pitcher throws: R

── Game Situation ─────────────────────────────────────────
  Outs [0/1/2]: 1
  Prev at-bat result(s): field_out, strikeout, single
  Prev at-bat pitch count(s) (range or list): 1-5
  ...

══════════════════════════════════════════════════════════════════
  PREDICTION GRID  (Global CatBoost)
══════════════════════════════════════════════════════════════════
  Prev Result              PC=1   PC=2   PC=3   PC=4   PC=5
──────────────────────────────────────────────────────────────────
  field_out            NIP 0.31  NIP 0.30  NIP 0.29  ...
  strikeout            NIP 0.28  NIP 0.27  NIP 0.26  ...
  single               HIP 0.52  HIP 0.50  NIP 0.48  ...
```

### 6. Evaluate a game

Edit `GAME_DATE`, `HOME_TEAM`, and `AWAY_TEAM` at the top of `evaluate_game_global.py`, then:

```bash
python evaluate_game_global.py
```

Outputs a per-AB prediction table and a summary with accuracy, precision, recall, and F1 split by whether each batter was in the training pool.

---

## Notebook Descriptions

### Two-Class Notebooks (`hit_into_play` vs. `not_in_play`)

**`First_Pitch_Single_Batter.ipynb`**
Trains all six classifiers on one batter's first-pitch history. Used primarily for feature development and validation. Key configuration variables at the top: `TRAIN_START`, `TRAIN_END`, `VAL_START`, `VAL_END`, `DECAY_RATE`, and the batter lookup (by name via `playerid_lookup`).

**`First_Pitch_Multi_Batter.ipynb`**
Iterates over the top 300 batters by first-pitch hit-into-play rate (minimum 100 first-pitch PA in the training window). For each batter: builds a feature matrix, applies SMOTE with decay weights, trains all six classifiers, and evaluates on the held-out validation window. Outputs `multi_batter_results.csv` with per-batter per-model metrics and a confusion table for the Decision Tree.

**`First_Pitch_Global.ipynb`**
Trains one model per algorithm across all 300 batters combined. `batter_id` is retained as a feature using native categorical handling (CatBoost, XGBoost, LightGBM) or one-hot encoding (Decision Tree, Random Forest, ANN). Three additional batter-level tendency features are added: first-pitch swing rate, contact rate, and hit-into-play rate. Outputs `global_model_per_batter_results.csv` with per-batter breakdowns of the best global model's performance.

### Three-Class Notebooks (`ball` / `strike` / `hit_into_play`)

**`First_Pitch_Single_Batter_Multiclass.ipynb`**
Analogous to the two-class single-batter notebook. All pitch descriptions are consolidated into three classes: `ball` (ball, blocked_ball, hit_by_pitch, pitchout), `strike` (called_strike, foul, swinging_strike, and variants), and `hit_into_play`. ANN output layer uses `softmax` with `categorical_crossentropy`; XGBoost uses `multi:softmax`; CatBoost uses `MultiClass`.

**`First_Pitch_Multi_Batter_Multiclass.ipynb`**
Analogous to the two-class multi-batter notebook. Per-class precision, recall, and F1 are reported for all three classes. The ANN confusion table is output as `ann_multiclass_hip_confusion_per_batter.csv`.

**`First_Pitch_Global_Multiclass.ipynb`**
Analogous to the two-class global notebook with three-class adaptations. `classification_report` is included in the evaluation helper to show per-class breakdowns for all three classes alongside overall accuracy.

---

## Script Descriptions

### `statcast_loader.py`

Downloads pitch-by-pitch Statcast data from Baseball Savant via pybaseball and caches it as a parquet file. Subsequent loads are instant. Cache files are named `statcast_YYYYMMDD_YYYYMMDD.parquet` and stored in the same directory as the script.

```python
from statcast_loader import load_statcast
table = load_statcast('2024-03-28', '2026-04-15')
```

Pass `force=True` to force a fresh download even if a cache file exists.

### `build_context_features.py`

Walks every pitch in each game in strict chronological order (`game_pk` → `at_bat_number` → `pitch_number`) and computes four sequential features:

- **`pitcher_pitch_count_in_game`** — running pitch count for the pitcher in the current game at the moment each pitch is delivered
- **`batter_prior_hip_count_in_game`** — first-pitch HIPs by the batter so far in the current game, before the current at-bat
- **`pitcher_pitch_count_prior_game`** — total pitches thrown by the pitcher in their most recent prior game (proxy for workload/fatigue)
- **`batter_prior_game_hip_count`** — first-pitch HIPs by the batter in their most recent prior game (momentum signal)

Output is a four-column sidecar parquet joined to the main table by `pitch_id`.

```bash
python build_context_features.py statcast_20240328_20260415.parquet
# → saves statcast_20240328_20260415_context_features.parquet
```

### `train_global_model.py`

Trains the global CatBoost two-class model on the top 300 batters by first-pitch hit-into-play rate (minimum 100 PA). Saves eight artifacts to `./global_model/` that are loaded instantly at prediction time:

| Artifact | Contents |
|---|---|
| `catboost_model/` | CatBoost native model directory |
| `label_encoder.pkl` | Maps `hit_into_play` / `not_in_play` ↔ integer labels |
| `cat_categories.pkl` | Categorical column category lists (ensures prediction-time alignment) |
| `feature_columns.pkl` | Ordered feature column names |
| `batter_features.pkl` | Batter-level tendency features (swing/contact/HIP rates) |
| `pitcher_features.pkl` | First-pitch stats for all pitchers in training data |
| `meta.pkl` | Minority label index, training dates, `top_batter_ids` set |
| `first_pitch_meta.pkl` | Batter/pitcher handedness lookup table |

Key parameters at the top of the file: `TRAIN_START`, `TRAIN_END`, `DECAY_RATE`, `MIN_PA`, `TOP_N_BATTERS`.

### `predict_first_pitch_global.py`

Interactive CLI that loads saved model artifacts (no retraining) and returns a **grid of predictions** across combinations of:

- **Previous at-bat results** — enter one or more comma-separated values (e.g. `field_out, strikeout, single`)
- **Previous at-bat pitch counts** — enter a range (e.g. `1-6`) or comma-separated list (e.g. `3,5,7`)

Outs and base runners for the current at-bat are derived automatically from the previous result (e.g. `field_out` → outs +1; `single` → runner on 1B, push existing runners; `extra_base_hit` → runner on 2B). Batter handedness and pitcher handedness are looked up from historical data automatically.

For batters outside the top-300 training pool, a notice is displayed with an option to continue (predictions will rely on pitcher stats and batter tendency features rather than batter ID signal).

### `evaluate_game_global.py`

Evaluates the trained global model against all first-pitch PAs in a specified game. Configure `GAME_DATE`, `HOME_TEAM`, and `AWAY_TEAM` at the top of the file. Game context (previous at-bat results, pitch counts, runners, outs, win expectancy) is derived from the game's actual Statcast data in pitch order.

Outputs:
- Per-AB table with batter, pitcher, actual result, predicted result, and P(hit_into_play)
- `game_evaluation_YYYY-MM-DD.csv`
- Summary split three ways: all batters, batters in the training pool, and batters outside the training pool

---

## Results

The global CatBoost model trained on two seasons of data (2024–2026) with exponential decay weighting achieved the highest F1 score for predicting first-pitch hit-into-play events. The three-class models generally show lower hit-into-play F1 than their two-class counterparts due to the harder decision boundary — the model must distinguish not only whether contact is made but also whether a non-contact pitch was a ball or strike.

Early-season validation windows (April–June) consistently show lower performance than mid-to-late season validation windows (August–September). The most likely causes are roster instability (trades, call-ups creating unseen pitcher matchups), adjustments batters and pitchers make as the season develops, and smaller per-batter PA counts in validation reducing metric stability.

---

## Requirements

```
pybaseball>=2.2.5
scikit-learn>=1.3
imbalanced-learn>=0.11
catboost>=1.2
xgboost>=2.0
lightgbm>=4.0
tensorflow>=2.13
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
```

---

## Author
Robert Brydon — for inquiries or suggestions please connect on [LinkedIn](https://www.linkedin.com/in/robert-brydon-phd-0241b5186/).

