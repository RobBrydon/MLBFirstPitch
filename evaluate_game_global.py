"""
evaluate_game.py
────────────────
Evaluates the global CatBoost model (trained by train_global_model.py)
against every first-pitch at-bat in a real game.

For each first-pitch PA in the specified game:
  1. Loads the pre-trained global CatBoost model from MODEL_DIR
  2. Builds a prediction row from the actual game context (batter_id,
     pitcher stats, runners, outs, win expectancy, etc.)
  3. Compares the prediction to the actual first-pitch result

Outputs:
  • Per-AB prediction table printed to console
  • game_evaluation_<GAME_DATE>.csv saved to disk
  • Summary: TP / FP / FN / TN, accuracy, precision, recall, F1

Usage
─────
    python evaluate_game.py

Run train_global_model.py first to generate the model artifacts.
Edit GAME_DATE, HOME_TEAM, AWAY_TEAM at the top to change the game.

Default game: Minnesota Twins vs Detroit Tigers, 2026-06-10.
"""

import os, sys, warnings, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pybaseball import statcast, playerid_reverse_lookup, cache

cache.enable()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
GAME_DATE  = '2026-06-10'
HOME_TEAM  = 'DET'
AWAY_TEAM  = 'MIN'
MODEL_DIR  = './global_model'

CATEGORICAL_COLS_NATIVE = ['p_throws', 'outs_when_up', 'stand', 'prev_pitch_result', 'batter']

pitcher_cols_static = (
    ['pitcher', 'strike_percent', 'swing_percent_on_strikes',
     'contact_percent_on_strikes', 'in_play_percent_on_strikes']
    + [f'{pt}_percent' for pt in
       ['FF','SI','SL','CH','FC','ST','CU','FS','KC','None',
        'SV','KN','FA','EP','FO','CS','SC','PO','UN']]
)

EVENT_MAP = {
    'hit_by_pitch': 'walk',    'intent_walk': 'walk',    'catcher_interf': 'walk',
    'home_run': 'extra_base_hit', 'triple': 'extra_base_hit', 'double': 'extra_base_hit',
    'force_out': 'field_out',  'fielders_choice': 'field_out',
    'grounded_into_double_play': 'field_out', 'double_play': 'field_out',
    'fielders_choice_out': 'field_out', 'sac_bunt': 'field_out',
    'sac_fly': 'field_out',    'field_error': 'single',
    'strikeout_double_play': 'strikeout',
}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
def load_artifacts():
    required = ['catboost_model', 'label_encoder.pkl', 'cat_categories.pkl',
                'feature_columns.pkl', 'batter_features.pkl',
                'pitcher_features.pkl', 'meta.pkl', 'first_pitch_meta.pkl']
    missing = [f for f in required
               if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        print(f'ERROR: Missing model artifacts in {MODEL_DIR}/: {missing}')
        print('Run train_global_model.py first.')
        sys.exit(1)

    def load(fname):
        with open(os.path.join(MODEL_DIR, fname), 'rb') as f:
            return pickle.load(f)

    cat_model = CatBoostClassifier()
    cat_model.load_model(os.path.join(MODEL_DIR, 'catboost_model'))

    return {
        'model':           cat_model,
        'le':              load('label_encoder.pkl'),
        'cat_categories':  load('cat_categories.pkl'),
        'feature_columns': load('feature_columns.pkl'),
        'batter_feats':    load('batter_features.pkl'),
        'pitcher_feats':   load('pitcher_features.pkl'),
        'meta':            load('meta.pkl'),
        'fp_meta':         load('first_pitch_meta.pkl'),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_pitcher_stats(pitcher_id, pitcher_feats_df):
    row = pitcher_feats_df[pitcher_feats_df['pitcher'] == pitcher_id]
    if len(row) == 0:
        blank = pd.DataFrame([{'pitcher': pitcher_id}])
        for col in pitcher_cols_static[1:]:
            blank[col] = 0.0
        return blank
    return row.reset_index(drop=True)


def get_batter_tendencies(batter_id, batter_feats_df):
    row = batter_feats_df[batter_feats_df['batter'] == batter_id]
    if len(row) == 0:
        return 0.0, 0.0, 0.0
    return (float(row['batter_first_pitch_swing_pct'].iloc[0]),
            float(row['batter_first_pitch_contact_pct'].iloc[0]),
            float(row['batter_first_pitch_hip_pct'].iloc[0]))


def predict_ab(row, batter_id, pitcher_id, art):
    """Build prediction row from a game Statcast row and return (outcome, hip_prob)."""
    feature_columns = art['feature_columns']
    cat_categories  = art['cat_categories']
    cat_model       = art['model']
    le              = art['le']
    minority_label  = art['meta']['minority_label']

    pred = pd.DataFrame([{col: 0.0 for col in feature_columns}])

    # Game-state values from the actual Statcast row
    for col in ['stand', 'p_throws', 'outs_when_up',
                'prev_pitch_result', 'prev_ab_pitch_count',
                'pitcher_pitch_count_in_game', 'batter_prior_hip_count_in_game',
                'pitcher_pitch_count_prior_game', 'batter_prior_game_hip_count',
                'bat_win_exp', 'at_bat_number']:
        if col in pred.columns and col in row.index:
            pred[col] = row[col]

    # Base runners
    for col in ['on_1b', 'on_2b', 'on_3b']:
        if col in pred.columns:
            val = row.get(col, 0)
            pred[col] = 1 if pd.notna(val) and val not in (0, '0', '', False) else 0

    # batter_id as string
    if 'batter' in pred.columns:
        pred['batter'] = str(batter_id)

    # Pitcher stats
    pitcher_stats = get_pitcher_stats(pitcher_id, art['pitcher_feats'])
    for col in pitcher_cols_static[1:]:
        if col in pitcher_stats.columns and col in pred.columns:
            pred[col] = pitcher_stats[col].iloc[0]

    # Batter-level tendency features
    swing_pct, contact_pct, hip_pct = get_batter_tendencies(batter_id, art['batter_feats'])
    for col, val in [('batter_first_pitch_swing_pct',   swing_pct),
                     ('batter_first_pitch_contact_pct', contact_pct),
                     ('batter_first_pitch_hip_pct',     hip_pct)]:
        if col in pred.columns:
            pred[col] = val

    # Align categoricals to training categories
    for col in CATEGORICAL_COLS_NATIVE:
        if col not in pred.columns or col not in cat_categories:
            continue
        train_cats = cat_categories[col]
        val_str    = str(pred[col].iloc[0])
        if val_str not in train_cats:
            val_str = train_cats[0]
        pred[col] = pd.Categorical([val_str], categories=train_cats)

    try:
        pred_encoded = cat_model.predict(pred)[0]
        pred_proba   = cat_model.predict_proba(pred)[0]
    except Exception as e:
        return None, None

    hip_prob = pred_proba[minority_label]
    outcome  = le.inverse_transform([int(pred_encoded)])[0]
    return outcome, hip_prob


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 68)
    print(f'  Game Evaluation: {AWAY_TEAM} @ {HOME_TEAM}  —  {GAME_DATE}')
    print(f'  Model: Global CatBoost  ({MODEL_DIR}/)')
    print('=' * 68)

    # ── Load model artifacts ──────────────────────────────────────────────────
    print(f'\nLoading model artifacts from {MODEL_DIR}/ ...')
    art  = load_artifacts()
    meta = art['meta']
    print(f"  Trained on {meta['train_start']} → {meta['train_end']}")
    print(f"  Top {meta.get('top_n_batters','?')} batters by first-pitch HIP rate")

    trained_ids = meta.get('top_batter_ids', set())

    # ── Fetch game data ───────────────────────────────────────────────────────
    print(f'\nFetching game data for {GAME_DATE} ({AWAY_TEAM} @ {HOME_TEAM})...')
    try:
        all_day = statcast(start_dt=GAME_DATE, end_dt=GAME_DATE)
    except Exception as e:
        print(f'  ERROR fetching game: {e}')
        sys.exit(1)

    if all_day is None or len(all_day) == 0:
        print('  No game data returned for that date.')
        sys.exit(1)

    game_df = all_day[
        ((all_day['home_team'] == HOME_TEAM) & (all_day['away_team'] == AWAY_TEAM)) |
        ((all_day['home_team'] == AWAY_TEAM) & (all_day['away_team'] == HOME_TEAM))
    ].copy()

    if len(game_df) == 0:
        print(f'  No data found for {AWAY_TEAM} @ {HOME_TEAM} on {GAME_DATE}.')
        print(f'  Games on that date: '
              f'{all_day[["home_team","away_team"]].drop_duplicates().values.tolist()}')
        sys.exit(1)

    game_df['pitch_id'] = game_df.index

    # Default columns that may not be in Statcast data
    for col in ['pitcher_pitch_count_in_game', 'batter_prior_hip_count_in_game',
                'pitcher_pitch_count_prior_game', 'batter_prior_game_hip_count']:
        game_df[col] = 0
    if 'bat_win_exp' not in game_df.columns:
        game_df['bat_win_exp'] = 0.5

    # Load game context sidecar if available
    game_context_path = f'statcast_{GAME_DATE.replace("-","")}_context_features.parquet'
    if os.path.exists(game_context_path):
        ctx_game = pd.read_parquet(game_context_path)
        game_df  = game_df.merge(ctx_game, on='pitch_id', how='left')
        for col in ['pitcher_pitch_count_in_game', 'batter_prior_hip_count_in_game',
                    'pitcher_pitch_count_prior_game', 'batter_prior_game_hip_count']:
            if col in game_df.columns:
                game_df[col] = game_df[col].fillna(0).astype(int)
            else:
                game_df[col] = 0
        print('  Game context sidecar loaded.')

    # Compute game-state context sequentially
    game_df = game_df.sort_values(['at_bat_number', 'pitch_number']).reset_index(drop=True)
    game_df['pitch_id'] = game_df.index
    game_df['prev_pitch_result']   = 'start_of_game'
    game_df['prev_ab_pitch_count'] = 0

    for i in range(len(game_df)):
        select_id   = game_df['pitch_id'].iloc[i]
        select_game = game_df['game_pk'].iloc[i]
        select_inn  = game_df['inning'].iloc[i]
        select_top  = game_df['inning_topbot'].iloc[i]
        prev = game_df[game_df['pitch_id'] == (select_id + 1)]
        if prev.empty:
            continue
        if prev['game_pk'].values[0] != select_game:
            pass
        elif (prev['inning'].values[0] != select_inn or
              prev['inning_topbot'].values[0] != select_top):
            game_df.at[i, 'prev_pitch_result'] = 'start_of_inning'
        else:
            raw_event = prev['events'].values[0]
            game_df.at[i, 'prev_pitch_result'] = (
                EVENT_MAP.get(raw_event, raw_event)
                if pd.notna(raw_event) else 'start_of_inning'
            )
            game_df.at[i, 'prev_ab_pitch_count'] = prev['pitch_number'].values[0]

    game_first = game_df[(game_df['balls'] == 0) & (game_df['strikes'] == 0)].copy()
    print(f'  {len(game_df)} total pitches  |  {len(game_first)} first-pitch PAs\n')

    # ── Build name map ────────────────────────────────────────────────────────
    all_ids = (list(game_first['batter'].unique()) +
               list(game_first['pitcher'].unique()))
    try:
        lookup = playerid_reverse_lookup(
            [int(x) for x in all_ids if pd.notna(x)], key_type='mlbam'
        )
        name_map = {
            int(r['key_mlbam']): f"{r['name_first'].title()} {r['name_last'].title()}"
            for _, r in lookup.iterrows()
        }
    except Exception:
        name_map = {}

    def get_name(mid):
        return name_map.get(int(mid), str(mid))

    # ── Run predictions ───────────────────────────────────────────────────────
    print(f'Running predictions on {len(game_first)} first-pitch PAs...\n')
    records = []

    for _, row in game_first.iterrows():
        batter_id  = row['batter']
        pitcher_id = row['pitcher']

        actual_label = ('hit_into_play'
                        if row['description'] == 'hit_into_play' else 'not_in_play')

        rec = {
            'inning':    row.get('inning', '?'),
            'top_bot':   row.get('inning_topbot', '?'),
            'batter':    get_name(batter_id),
            'pitcher':   get_name(pitcher_id),
            'stand':     row.get('stand', '?'),
            'p_throws':  row.get('p_throws', '?'),
            'outs':      row.get('outs_when_up', '?'),
            'in_model':  batter_id in trained_ids,
            'actual':    actual_label,
            'predicted': 'N/A',
            'p_hip':     None,
            'correct':   None,
        }

        predicted, hip_prob = predict_ab(row, batter_id, pitcher_id, art)

        if predicted is None:
            rec['predicted'] = 'ERROR'
        else:
            rec['predicted'] = predicted
            rec['p_hip']     = round(hip_prob, 3)
            rec['correct']   = (predicted == actual_label)

        records.append(rec)

    # ── Results table ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    print(results_df[[
        'inning','top_bot','batter','pitcher','stand','p_throws',
        'outs','in_model','actual','predicted','p_hip','correct'
    ]].to_string(index=False))

    csv_path = f'game_evaluation_{GAME_DATE}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    # ── Summary ───────────────────────────────────────────────────────────────
    scored     = results_df[results_df['correct'].notna()]
    in_model   = results_df[results_df['in_model'] == True]
    out_model  = results_df[results_df['in_model'] == False]

    if len(scored) == 0:
        print('No scoreable predictions.')
        return

    def confusion_block(df, label):
        s = df[df['correct'].notna()]
        if len(s) == 0:
            return
        tp  = int(((s['predicted'] == 'hit_into_play') & (s['actual'] == 'hit_into_play')).sum())
        fp  = int(((s['predicted'] == 'hit_into_play') & (s['actual'] == 'not_in_play')).sum())
        fn  = int(((s['predicted'] == 'not_in_play')   & (s['actual'] == 'hit_into_play')).sum())
        tn  = int(((s['predicted'] == 'not_in_play')   & (s['actual'] == 'not_in_play')).sum())
        acc = s['correct'].mean()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f'\n  {label} ({len(s)} at-bats)')
        print(f'  Accuracy   : {acc:.3f}')
        print(f'  TP / FP / FN / TN : {tp} / {fp} / {fn} / {tn}')
        print(f'  Precision  : {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}')

    print('\n' + '=' * 68)
    print('  SUMMARY  (Global CatBoost)')
    print('=' * 68)
    confusion_block(results_df, 'All batters')
    if len(in_model)  > 0: confusion_block(in_model,  'In training pool')
    if len(out_model) > 0: confusion_block(out_model, 'Outside training pool')
    print('=' * 68)


if __name__ == '__main__':
    main()
