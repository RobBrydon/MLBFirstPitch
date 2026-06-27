"""
predict_first_pitch_global.py
─────────────────────────────
Loads a pre-trained global CatBoost model (saved by train_global_model.py)
and makes an instant first-pitch hit-into-play prediction for any batter /
pitcher / game situation.

Run train_global_model.py first to generate the model artifacts, then:
    python predict_first_pitch_global.py

The model is loaded from MODEL_DIR (default: ./global_model/) in under
a second — no retraining required between predictions.
"""

import os, sys, warnings, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pybaseball import playerid_lookup, cache

cache.enable()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR = './global_model'

PITCH_TYPES = ['FF','SI','SL','CH','FC','ST','CU','FS','KC','None',
               'SV','KN','FA','EP','FO','CS','SC','PO','UN']

pitcher_cols = (
    ['pitcher', 'strike_percent', 'swing_percent_on_strikes',
     'contact_percent_on_strikes', 'in_play_percent_on_strikes']
    + [f'{pt}_percent' for pt in PITCH_TYPES]
)

CATEGORICAL_COLS_NATIVE = ['p_throws', 'outs_when_up', 'stand', 'prev_pitch_result', 'batter']

PREV_RESULT_OPTIONS = [
    'start_of_game', 'start_of_inning', 'single', 'strikeout',
    'field_out', 'walk', 'extra_base_hit',
]

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
        print('Run train_global_model.py first to generate them.')
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
def get_batter_stand(batter_id, fp_meta):
    rows = fp_meta[fp_meta['batter'] == batter_id]['stand'].dropna()
    if rows.empty:
        return None
    unique = rows.unique()
    return unique[0] if len(unique) == 1 else 'S'


def get_pitcher_throws(pitcher_id, fp_meta):
    rows = fp_meta[fp_meta['pitcher'] == pitcher_id]['p_throws'].dropna()
    if rows.empty:
        return None
    return rows.mode().iloc[0]


def get_batter_tendencies(batter_id, batter_feats_df):
    """Return (swing_pct, contact_pct, hip_pct) for a batter, or (0,0,0) if unseen."""
    row = batter_feats_df[batter_feats_df['batter'] == batter_id]
    if len(row) == 0:
        return 0.0, 0.0, 0.0
    return (float(row['batter_first_pitch_swing_pct'].iloc[0]),
            float(row['batter_first_pitch_contact_pct'].iloc[0]),
            float(row['batter_first_pitch_hip_pct'].iloc[0]))


def get_pitcher_stats(pitcher_id, pitcher_feats_df):
    """Return pitcher feature row, or zeros if pitcher not seen in training."""
    row = pitcher_feats_df[pitcher_feats_df['pitcher'] == pitcher_id]
    if len(row) == 0:
        blank = pd.DataFrame([{'pitcher': pitcher_id}])
        for col in pitcher_cols[1:]:
            blank[col] = 0.0
        return blank
    return row.reset_index(drop=True)


def lookup_player_id(last, first):
    result = playerid_lookup(last, first)
    result = result[result['key_mlbam'].notna()].reset_index(drop=True)
    if result.empty:
        print(f"  No MLBAM ID found for '{first} {last}'.")
        mlbam = input('  Enter MLBAM ID directly (or press Enter to skip): ').strip()
        if mlbam.isdigit():
            return int(mlbam)
        return None
    if len(result) == 1:
        return int(result['key_mlbam'].iloc[0])
    print(f"  Multiple matches for '{first} {last}':")
    for i, row in result.iterrows():
        print(f"    [{i}] {row['name_first']} {row['name_last']}  "
              f"born {row.get('birth_year','?')}  mlbam={int(row['key_mlbam'])}")
    choice = input("  Enter index (or an MLBAM ID directly): ").strip()
    if choice.isdigit() and int(choice) not in result.index.tolist():
        return int(choice)   # treated as a raw MLBAM ID
    return int(result['key_mlbam'].iloc[int(choice)])


def prompt(label, options=None, default=None):
    opts_str = f" [{'/'.join(options)}]" if options else ''
    dflt_str = f" (default: {default})" if default is not None else ''
    while True:
        val = input(f'  {label}{opts_str}{dflt_str}: ').strip()
        if val == '' and default is not None:
            return default
        if options and val not in options:
            print(f"    Please enter one of: {', '.join(options)}")
            continue
        return val


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 62)
    print('  First-Pitch HIP Predictor — Global CatBoost')
    print('=' * 62)

    # ── Load artifacts ─────────────────────────────────────────────────────────
    print(f'\nLoading model from {MODEL_DIR}/ ...')
    art = load_artifacts()
    meta = art['meta']
    print(f"  Model trained on {meta['train_start']} → {meta['train_end']}")
    print(f"  Classes: {meta['classes']}")

    # ── Player lookup ──────────────────────────────────────────────────────────
    print('\n── Batter ─────────────────────────────────────────────────')
    b_last    = input('  Last name : ').strip()
    b_first   = input('  First name: ').strip()
    batter_id = lookup_player_id(b_last, b_first)
    if batter_id is None:
        sys.exit(1)

    batter_stand = get_batter_stand(batter_id, art['fp_meta'])
    swing_pct, contact_pct, hip_pct = get_batter_tendencies(batter_id, art['batter_feats'])

    top_n       = meta.get('top_n_batters', 300)
    trained_ids = meta.get('top_batter_ids', set())
    if batter_id not in trained_ids:
        print(f'\n  *** NOTICE ***')
        print(f'  {b_first} {b_last} is not among the top {top_n} batters by')
        print(f'  first-pitch HIP rate that this model was trained on.')
        print(f'  Predictions for this batter may be unreliable.')
        print(f'  Run train_global_model.py with a larger TOP_N_BATTERS')
        print(f'  or a lower MIN_PA to include this batter.')
        print()
        cont = input('  Continue anyway? [y/n] (default: n): ').strip().lower()
        if cont != 'y':
            print('  Exiting.')
            sys.exit(0)

    if batter_stand is None:
        print('  WARNING: batter not seen in training data — handedness unknown.')
        batter_stand = prompt('  Enter manually', options=['L','R'])
    elif batter_stand == 'S':
        print('  Switch hitter detected — will be set to opposite of pitcher hand.')
    else:
        print(f'  Batter stands: {batter_stand}')

    print('\n── Pitcher ────────────────────────────────────────────────')
    p_last     = input('  Last name : ').strip()
    p_first    = input('  First name: ').strip()
    pitcher_id = lookup_player_id(p_last, p_first)
    if pitcher_id is None:
        sys.exit(1)

    p_throws = get_pitcher_throws(pitcher_id, art['fp_meta'])
    if p_throws is None:
        print('  WARNING: pitcher not seen in training data — handedness unknown.')
        p_throws = prompt('  Enter manually', options=['L','R'])
    else:
        print(f'  Pitcher throws: {p_throws}')

    if batter_stand == 'S':
        stand = 'R' if p_throws == 'L' else 'L'
        print(f'  Switch hitter vs {p_throws}HP → batting {stand}')
    else:
        stand = batter_stand

    print('\n── Game Situation ─────────────────────────────────────────')
    outs_when_up    = prompt('Outs at start of this AB',          options=['0','1','2'])
    on_1b           = prompt('Runner on 1B at start of this AB',  options=['0','1'], default='0')
    on_2b           = prompt('Runner on 2B at start of this AB',  options=['0','1'], default='0')
    on_3b           = prompt('Runner on 3B at start of this AB',  options=['0','1'], default='0')
    # ── Prev AB result: one or more (comma-separated) ─────────────────────────
    print(f'  Prev at-bat result options: {", ".join(PREV_RESULT_OPTIONS)}')
    prev_results_raw = input(
        '  Prev at-bat result(s) — comma-separated for grid '
        f'(default: start_of_game): '
    ).strip()
    if not prev_results_raw:
        prev_results_raw = 'start_of_game'
    prev_results_list = [r.strip() for r in prev_results_raw.split(',')]
    invalid = [r for r in prev_results_list if r not in PREV_RESULT_OPTIONS]
    if invalid:
        print(f'  WARNING: unknown result(s) ignored: {invalid}')
        prev_results_list = [r for r in prev_results_list if r in PREV_RESULT_OPTIONS] or ['start_of_game']

    # ── Prev AB pitch count: a range or list ───────────────────────────────────
    print('  Prev at-bat pitch count — enter a range (e.g. 1-6) or '
          'comma-separated values (e.g. 3,5,7).')
    prev_pitches_raw = input('  Prev at-bat pitch count(s) (default: 0): ').strip()
    if not prev_pitches_raw:
        prev_pitches_list = [0]
    elif '-' in prev_pitches_raw and ',' not in prev_pitches_raw:
        lo, hi = prev_pitches_raw.split('-', 1)
        prev_pitches_list = list(range(int(lo.strip()), int(hi.strip()) + 1))
    else:
        prev_pitches_list = [int(x.strip()) for x in prev_pitches_raw.split(',')]

    # ── Fixed game-state inputs ─────────────────────────────────────────────────
    batter_hip      = prompt('Batter first-pitch HIPs in game',            default='0')
    bat_win_exp     = prompt('Batting team win expectancy (0–1)',          default='0.5')
    at_bat_number   = prompt('At-bat number in the game',                  default='1')
    pitcher_pitches_base = int(prompt(
        'Pitcher pitches thrown BEFORE the previous at-bat',               default='0'
    ))
    pitcher_prior_game  = prompt('Pitcher total pitches in prior game',    default='0')
    batter_prior_hip    = prompt('Batter first-pitch HIPs in prior game',  default='0')

    # ── Shared prediction row template (filled once, varied per grid cell) ──────
    pitcher_stats  = get_pitcher_stats(pitcher_id, art['pitcher_feats'])
    minority_label = meta['minority_label']
    le             = art['le']
    cat_model      = art['model']
    cat_categories = art['cat_categories']

    # Base runners at START of previous AB (derive_state adjusts per result)
    base_runners = []
    if int(on_1b): base_runners.append('1B')
    if int(on_2b): base_runners.append('2B')
    if int(on_3b): base_runners.append('3B')

    # ── Helper: derive outs and runners for this AB from the previous AB result
    def derive_state(base_outs, base_1b, base_2b, base_3b, prev_result):
        """
        Given the game state AT THE START of the PREVIOUS at-bat and its result,
        return the outs and base runners AT THE START of THIS at-bat.

        Rules applied:
          field_out / strikeout → outs + 1 (capped at 2 for prediction purposes;
                                            3 would start a new inning)
          walk / hit_by_pitch   → runner on 1B (push existing runners)
          single                → runner on 1B, previous 1B runner moves to 2B
          extra_base_hit        → runner on 2B, bases otherwise cleared
          start_of_game /
          start_of_inning       → use base state as entered (no change)
          other (strikeout_dp   
          etc.)                 → outs + 1
        """
        outs  = int(base_outs)
        b_1b  = int(base_1b)
        b_2b  = int(base_2b)
        b_3b  = int(base_3b)

        if prev_result in ('start_of_game', 'start_of_inning'):
            return str(outs), b_1b, b_2b, b_3b

        if prev_result in ('field_out', 'strikeout'):
            return str(min(outs + 1, 2)), b_1b, b_2b, b_3b

        if prev_result in ('walk', 'hit_by_pitch'):
            # Force batter to 1B; push 1B runner to 2B if occupied
            new_2b = 1 if b_1b else b_2b
            new_3b = 1 if b_2b else b_3b
            return str(outs), 1, new_2b, new_3b

        if prev_result == 'single':
            new_2b = 1 if b_1b else b_2b
            new_3b = 1 if b_2b else b_3b
            return str(outs), 1, new_2b, new_3b

        if prev_result == 'extra_base_hit':
            return str(outs), 0, 1, b_3b

        # Catch-all (e.g. strikeout_double_play already mapped to strikeout)
        return str(min(outs + 1, 2)), b_1b, b_2b, b_3b

    def make_prediction(prev_result, prev_ab_count, pitcher_total_pitches):
        """Build one prediction row and return (outcome, hip_prob)."""
        pred_row = pd.DataFrame([{col: 0.0 for col in art['feature_columns']}])

        # Derive this AB's outs and runners from the previous AB result
        this_outs, this_1b, this_2b, this_3b = derive_state(
            outs_when_up, on_1b, on_2b, on_3b, prev_result
        )

        pred_row['batter']            = str(batter_id)
        pred_row['stand']             = stand
        pred_row['p_throws']          = p_throws
        pred_row['outs_when_up']      = this_outs
        pred_row['prev_pitch_result'] = prev_result

        for col in pitcher_cols[1:]:
            if col in pitcher_stats.columns and col in pred_row.columns:
                pred_row[col] = pitcher_stats[col].iloc[0]

        pred_row['batter_first_pitch_swing_pct']   = swing_pct
        pred_row['batter_first_pitch_contact_pct'] = contact_pct
        pred_row['batter_first_pitch_hip_pct']     = hip_pct

        pred_row['on_1b']                           = this_1b
        pred_row['on_2b']                           = this_2b
        pred_row['on_3b']                           = this_3b
        pred_row['prev_ab_pitch_count']             = int(prev_ab_count)
        pred_row['pitcher_pitch_count_in_game']     = int(pitcher_total_pitches)
        pred_row['batter_prior_hip_count_in_game']  = int(batter_hip)
        pred_row['pitcher_pitch_count_prior_game']  = int(pitcher_prior_game)
        pred_row['batter_prior_game_hip_count']     = int(batter_prior_hip)
        pred_row['bat_win_exp']                     = float(bat_win_exp)
        pred_row['at_bat_number']                   = int(at_bat_number)

        for col in CATEGORICAL_COLS_NATIVE:
            if col not in pred_row.columns or col not in cat_categories:
                continue
            train_cats = cat_categories[col]
            val_str    = str(pred_row[col].iloc[0])
            if val_str not in train_cats:
                val_str = train_cats[0]
            pred_row[col] = pd.Categorical([val_str], categories=train_cats)

        pred_encoded = cat_model.predict(pred_row)[0]
        pred_proba   = cat_model.predict_proba(pred_row)[0]
        hip_prob     = pred_proba[minority_label]
        outcome      = le.inverse_transform([int(pred_encoded)])[0]
        return outcome, hip_prob

    # ── Build and print the prediction grid ────────────────────────────────────
    print('\n' + '=' * 78)
    print('  PREDICTION GRID  (Global CatBoost)')
    print('=' * 78)
    print(f'  Batter   : {b_first} {b_last}  (bats {stand})   '
          f'Pitcher: {p_first} {p_last}  (throws {p_throws})')
    print(f'  Base outs: {outs_when_up}   Base runners (prev AB start): '
          f'{", ".join(base_runners) if base_runners else "none"}   '
          f'Win exp: {bat_win_exp}   AB#: {at_bat_number}')
    print(f'  Outs/runners for THIS AB are derived from the previous AB result.')
    print(f'  Batter swing/contact/HIP %: '
          f'{swing_pct:.3f} / {contact_pct:.3f} / {hip_pct:.3f}')
    print(f'  Pitcher prior game pitches : {pitcher_prior_game}   '
          f'Batter prior game HIPs: {batter_prior_hip}')
    print('-' * 78)

    # Header row
    col_w = 14
    header = f'  {"Prev Result":<20}' + ''.join(
        f'{f"PC={pc}":>{col_w}}' for pc in prev_pitches_list
    )
    print(header)
    print('-' * 78)

    for prev_result in prev_results_list:
        row_str = f'  {prev_result:<20}'
        for prev_ab_count in prev_pitches_list:
            # Total pitcher pitches = pitches before this AB + this AB's prev pitch count
            pitcher_total = pitcher_pitches_base + prev_ab_count
            outcome, hip_prob = make_prediction(prev_result, prev_ab_count, pitcher_total)
            d_outs, d_1b, d_2b, d_3b = derive_state(
                outs_when_up, on_1b, on_2b, on_3b, prev_result
            )
            label = 'HIP' if outcome == 'hit_into_play' else 'NIP'
            cell  = f'{label} {hip_prob:.2f}'
            row_str += f'{cell:>{col_w}}'
        print(row_str)

    print('-' * 78)
    print('  HIP = Hit Into Play predicted   NIP = Not In Play predicted')
    print('  PC  = Prev at-bat pitch count   probability shown beside label')
    print('=' * 78)


if __name__ == '__main__':
    main()
