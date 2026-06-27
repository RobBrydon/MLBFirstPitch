"""
train_global_model.py
─────────────────────
Trains a global CatBoost model across the top 300 batters by first-pitch
hit-into-play rate (min 100 PA) and saves all artifacts needed
for fast prediction.

Run once (or whenever you want to refresh with new data):
    python train_global_model.py

Artifacts saved to MODEL_DIR (default: ./global_model/):
    catboost_model/        — CatBoost native model directory
    label_encoder.pkl      — sklearn LabelEncoder (hit_into_play / not_in_play)
    cat_categories.pkl     — dict of {col: [category_list]} for each categorical
    feature_columns.pkl    — ordered list of X column names
    batter_features.pkl    — DataFrame of batter-level tendency features
    pitcher_features.pkl   — DataFrame of pitcher first-pitch stats
    first_pitch_meta.pkl   — minority_label index, classes list, train date range
    first_pitch_data.pkl   — first_pitch DataFrame (for stand/p_throws lookups)

These artifacts are loaded by predict_first_pitch_global.py at prediction time
with no retraining required.
"""

import os, sys, warnings, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from pybaseball import playerid_reverse_lookup, cache

cache.enable()
sys.path.insert(0, '.')
from statcast_loader import load_statcast

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_START = '2024-03-28'
TRAIN_END   = '2026-06-23'
RANDOM_SEED = 0
DECAY_RATE  = 0.004
MIN_PA        = 100
TOP_N_BATTERS = 300
MODEL_DIR     = './global_model'

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
NOT_IN_PLAY_OUTCOMES = [
    'called_strike', 'foul', 'swinging_strike', 'swinging_strike_blocked',
    'foul_tip', 'foul_bunt', 'missed_bunt', 'blocked_ball', 'ball',
    'bunt_foul_tip', 'automatic_strike', 'automatic_ball', 'intent_ball',
    'pitchout', 'hit_by_pitch',
]

SWING_SET   = {'swinging_strike','swinging_strike_blocked','foul','foul_bunt',
               'hit_into_play','foul_tip','bunt_foul_tip','missed_bunt'}
CONTACT_SET = {'foul','foul_bunt','hit_into_play','foul_tip','bunt_foul_tip'}

PITCH_TYPES = ['FF','SI','SL','CH','FC','ST','CU','FS','KC','None',
               'SV','KN','FA','EP','FO','CS','SC','PO','UN']

pitcher_cols = (
    ['pitcher', 'strike_percent', 'swing_percent_on_strikes',
     'contact_percent_on_strikes', 'in_play_percent_on_strikes']
    + [f'{pt}_percent' for pt in PITCH_TYPES]
)

DROP_COLS_1 = [
    'game_date','spin_dir','zone','spin_axis','spin_rate_deprecated',
    'break_angle_deprecated','break_length_deprecated','des','game_type','type',
    'hit_location','bb_type','balls','strikes','game_year','hc_x','hc_y',
    'tfs_deprecated','tfs_zulu_deprecated','umpire','sv_id','player_name',
    'hit_distance_sc','launch_speed','launch_angle',
    'fielder_2','fielder_3','fielder_4','fielder_5','fielder_6','fielder_7',
    'fielder_8','fielder_9',
    'home_team','away_team','home_score','away_score','bat_score','fld_score',
    'post_away_score','post_home_score','post_bat_score','post_fld_score',
    'release_pos_y','delta_home_win_exp','delta_run_exp','bat_speed','swing_length',
    'estimated_slg_using_speedangle','delta_pitcher_run_exp','hyper_speed',
    'home_score_diff','home_win_exp','age_pit_legacy','age_bat_legacy',
    'age_pit','age_bat','pitcher_days_since_prev_game','batter_days_since_prev_game',
    'pitcher_days_until_next_game','batter_days_until_next_game','pitch_name',
    # 'batter' intentionally NOT in this list — kept as a feature
]
DROP_COLS_2 = [
    'if_fielding_alignment','of_fielding_alignment','pitch_type','release_speed',
    'release_pos_x','release_pos_z','vx0','vy0','vz0','ax','ay','az','sz_top',
    'effective_speed','release_spin_rate','release_extension','sz_bot','pfx_x','pfx_z',
    'arm_angle','plate_x','plate_z','api_break_z_with_gravity','api_break_x_arm',
    'api_break_x_batter_in','estimated_ba_using_speedangle','estimated_woba_using_speedangle',
    'woba_value','woba_denom','babip_value','iso_value','launch_speed_angle',
]
DROP_COLS_3 = [
    'bat_score_diff','n_thruorder_pitcher','n_priorpa_thisgame_player_at_bat',
    'attack_angle','attack_direction','swing_path_tilt',
    'intercept_ball_minus_batter_pos_x_inches','intercept_ball_minus_batter_pos_y_inches',
    'miss_distance',
]

CATEGORICAL_COLS_NATIVE = ['p_throws', 'outs_when_up', 'stand', 'prev_pitch_result', 'batter']
DROP_BEFORE_MODEL       = ['pitcher', 'game_pk', 'inning', 'inning_topbot',
                            'pitch_number', 'pitch_id']

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
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def compute_pitch_counts(desc_series):
    vc = desc_series.value_counts()
    return {k: vc.get(k, 0) for k in
            ['ball','called_strike','swinging_strike','foul','foul_bunt',
             'hit_into_play','blocked_ball','foul_tip','bunt_foul_tip',
             'swinging_strike_blocked','hit_by_pitch','missed_bunt',
             'pitchout','automatic_strike','automatic_ball','intent_ball']}


def build_pitcher_features(pitcher_ids, source_first_pitch):
    pdf = pd.DataFrame(pitcher_ids, columns=['pitcher'])
    for col in pitcher_cols[1:]:
        pdf[col] = 0.0
    for i in range(len(pdf)):
        pid = pdf['pitcher'].iloc[i]
        sel = source_first_pitch[source_first_pitch['pitcher'] == pid]
        c   = compute_pitch_counts(sel['description'])
        total_p   = sum(c.values())
        t_strikes = (c['called_strike'] + c['swinging_strike'] + c['foul'] +
                     c['foul_bunt'] + c['hit_into_play'] + c['foul_tip'] +
                     c['swinging_strike_blocked'] + c['missed_bunt'] +
                     c['bunt_foul_tip'] + c['automatic_strike'])
        t_contact = (c['foul'] + c['foul_bunt'] + c['hit_into_play'] +
                     c['foul_tip'] + c['bunt_foul_tip'])
        t_swings  = (c['swinging_strike'] + c['swinging_strike_blocked'] +
                     c['foul'] + c['foul_bunt'] + c['hit_into_play'] +
                     c['foul_tip'] + c['bunt_foul_tip'])
        if total_p > 0:
            pdf.at[i, 'strike_percent'] = t_strikes / total_p
            if t_strikes > 0:
                pdf.at[i, 'swing_percent_on_strikes']  = t_swings  / t_strikes
                pdf.at[i, 'contact_percent_on_strikes'] = t_contact / t_strikes
                pdf.at[i, 'in_play_percent_on_strikes'] = c['hit_into_play'] / t_strikes
        sel_pt   = sel['pitch_type'].fillna('None')
        pt_vc    = sel_pt.value_counts()
        total_pt = pt_vc.sum()
        if total_pt > 0:
            for pt in PITCH_TYPES:
                pdf.at[i, f'{pt}_percent'] = pt_vc.get(pt, 0) / total_pt
    return pdf


def build_batter_features(batter_ids, source_first_pitch):
    bdf = pd.DataFrame(batter_ids, columns=['batter'])
    bdf['batter_first_pitch_swing_pct']   = 0.0
    bdf['batter_first_pitch_contact_pct'] = 0.0
    bdf['batter_first_pitch_hip_pct']     = 0.0
    for i in range(len(bdf)):
        bid  = bdf['batter'].iloc[i]
        sel  = source_first_pitch[source_first_pitch['batter'] == bid]
        n    = len(sel)
        if n == 0:
            continue
        desc      = sel['description']
        n_swing   = desc.isin(SWING_SET).sum()
        n_contact = desc.isin(CONTACT_SET).sum()
        n_hip     = (desc == 'hit_into_play').sum()
        bdf.at[i, 'batter_first_pitch_swing_pct']   = n_swing / n
        bdf.at[i, 'batter_first_pitch_contact_pct'] = (n_contact / n_swing) if n_swing > 0 else 0.0
        bdf.at[i, 'batter_first_pitch_hip_pct']     = n_hip / n
    return bdf


def add_game_state_context(merged_df, source_table):
    merged_df = merged_df.reset_index(drop=True).copy()
    merged_df['prev_pitch_result']   = 'start_of_game'
    merged_df['prev_ab_pitch_count'] = 0
    for i in range(len(merged_df)):
        select_id   = merged_df['pitch_id'].iloc[i]
        select_game = merged_df['game_pk'].iloc[i]
        select_inn  = merged_df['inning'].iloc[i]
        select_top  = merged_df['inning_topbot'].iloc[i]
        prev = source_table[source_table['pitch_id'] == (select_id + 1)]
        if prev.empty:
            continue
        if prev['game_pk'].values[0] != select_game:
            pass
        elif (prev['inning'].values[0] != select_inn or
              prev['inning_topbot'].values[0] != select_top):
            merged_df.at[i, 'prev_pitch_result'] = 'start_of_inning'
        else:
            merged_df.at[i, 'prev_pitch_result']   = prev['events'].values[0]
            merged_df.at[i, 'prev_ab_pitch_count'] = prev['pitch_number'].values[0]
    merged_df['prev_pitch_result'] = merged_df['prev_pitch_result'].replace(EVENT_MAP)
    desc_col = merged_df.pop('description')
    merged_df.insert(len(merged_df.columns), 'description', desc_col)
    return merged_df


def clean_rows(rows):
    clean = (
        rows
        .drop(columns=DROP_COLS_1, errors='ignore')
        .drop(columns=DROP_COLS_2, errors='ignore')
        .drop(columns=DROP_COLS_3, errors='ignore')
    )
    for col in ['on_1b', 'on_2b', 'on_3b']:
        clean[col] = clean[col].fillna(0)
        clean.loc[clean[col] > 0, col] = 1
    return clean


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'  Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print('=' * 62)
    print('  Global CatBoost — Training Script')
    print(f'  {TRAIN_START} → {TRAIN_END}')
    print('=' * 62)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f'\n[1/6] Loading Statcast data...')
    table = load_statcast(TRAIN_START, TRAIN_END)
    table['pitch_id'] = table.index

    context_path = (f'statcast_{TRAIN_START.replace("-","")}_{TRAIN_END.replace("-","")}'
                    f'_context_features.parquet')
    CONTEXT_COLS = ['pitcher_pitch_count_in_game', 'batter_prior_hip_count_in_game',
                    'pitcher_pitch_count_prior_game', 'batter_prior_game_hip_count']

    if os.path.exists(context_path):
        ctx   = pd.read_parquet(context_path)
        table = table.drop(columns=[c for c in CONTEXT_COLS if c in table.columns], errors='ignore')
        table = table.merge(ctx, on='pitch_id', how='left')
        for col in CONTEXT_COLS:
            if col in table.columns:
                table[col] = table[col].fillna(0).astype(int)
            else:
                table[col] = 0
                print(f'  NOTE: {col} not in sidecar — set to 0. Regenerate sidecar.')
        print('  Context sidecar loaded.')
    else:
        for col in CONTEXT_COLS:
            table[col] = 0
        print(f'  WARNING: {context_path} not found — context features set to 0.')

    table['pitch_id'] = table.index
    first_pitch = table[(table['balls'] == 0) & (table['strikes'] == 0)]
    print(f'  {len(table):,} pitches  |  {len(first_pitch):,} first-pitch PAs')

    # ── Select batters ────────────────────────────────────────────────────────
    print(f'\n[2/6] Selecting top {TOP_N_BATTERS} batters by first-pitch HIP rate '
          f'(min {MIN_PA} PA)...')
    pa_counts  = first_pitch['batter'].value_counts()
    hip_counts = (
        first_pitch[first_pitch['description'] == 'hit_into_play']
        ['batter'].value_counts()
    )
    hip_rate_df = (
        pd.DataFrame({'pa': pa_counts, 'hip': hip_counts})
        .fillna(0)
        .astype({'pa': int, 'hip': int})
        .assign(hip_rate=lambda df: df['hip'] / df['pa'])
        .loc[lambda df: df['pa'] >= MIN_PA]
        .sort_values('hip_rate', ascending=False)
    )
    top_batter_ids = list(hip_rate_df.head(TOP_N_BATTERS).index)
    print(f'  {len(top_batter_ids)} batters selected.')
    print(f'  HIP rate range: {hip_rate_df["hip_rate"].iloc[len(top_batter_ids)-1]:.3f} '
          f'– {hip_rate_df["hip_rate"].iloc[0]:.3f}')

    # ── Build combined dataset ────────────────────────────────────────────────
    print(f'\n[3/6] Building combined training dataset...')
    TRAIN_END_DT = pd.Timestamp(TRAIN_END)
    all_merged, all_weights, skipped = [], [], []

    for idx, bid in enumerate(top_batter_ids):
        rows = first_pitch[first_pitch['batter'] == bid].copy()
        if len(rows) < 5:
            skipped.append(bid)
            continue

        labeled = rows.drop('events', axis=1)
        labeled['description'] = labeled['description'].replace(
            {o: 'not_in_play' for o in NOT_IN_PLAY_OUTCOMES}
        )
        if 'hit_into_play' not in labeled['description'].values:
            skipped.append(bid)
            continue

        raw_dates = pd.to_datetime(labeled['game_date'], errors='coerce')
        days_ago  = (TRAIN_END_DT - raw_dates).dt.days.clip(lower=0).fillna(0)
        weight_by_pid = pd.Series(
            np.exp(-DECAY_RATE * days_ago.values),
            index=labeled['pitch_id'].values
        )

        pitcher_feats = build_pitcher_features(
            labeled['pitcher'].unique().tolist(), first_pitch
        )
        clean  = clean_rows(labeled)
        merged = pd.merge(clean, pitcher_feats, on='pitcher', how='left')
        for col in pitcher_cols[1:]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

        merged = add_game_state_context(merged, table)

        aligned_w = (
            weight_by_pid.reindex(merged['pitch_id'].values)
            .fillna(weight_by_pid.mean()).values
        )
        all_merged.append(merged)
        all_weights.append(aligned_w)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(top_batter_ids):
            print(f'  Processed {idx+1}/{len(top_batter_ids)} batters...')

    train_combined = pd.concat(all_merged, ignore_index=True)
    train_weights  = np.concatenate(all_weights)
    print(f'  Combined shape: {train_combined.shape}  ({len(skipped)} batters skipped)')

    # ── Batter-level features ─────────────────────────────────────────────────
    print(f'\n[4/6] Computing batter-level tendency features...')
    batter_feats   = build_batter_features(top_batter_ids, first_pitch)
    train_combined = train_combined.merge(batter_feats, on='batter', how='left')
    for col in ['batter_first_pitch_swing_pct',
                'batter_first_pitch_contact_pct',
                'batter_first_pitch_hip_pct']:
        train_combined[col] = train_combined[col].fillna(0.0)

    # Pitcher features for ALL pitchers seen in training (for lookup at predict time)
    all_pitcher_ids  = first_pitch['pitcher'].unique().tolist()
    pitcher_feats_all = build_pitcher_features(all_pitcher_ids, first_pitch)

    # ── Encode ────────────────────────────────────────────────────────────────
    print(f'\n[5/6] Encoding and training CatBoost...')
    dataset = train_combined.drop(columns=DROP_BEFORE_MODEL, errors='ignore')
    y_raw   = dataset['description']
    X_raw   = dataset.drop(columns=['description'])

    le      = LabelEncoder()
    y_enc   = le.fit_transform(y_raw)
    classes = list(le.classes_)
    minority_label = classes.index('hit_into_play')

    X_train        = X_raw.copy()
    cat_categories = {}
    for col in CATEGORICAL_COLS_NATIVE:
        if col not in X_train.columns:
            continue
        X_train[col]        = X_train[col].astype(str).astype('category')
        cat_categories[col] = list(X_train[col].cat.categories)

    cat_feature_idx = [X_train.columns.get_loc(c)
                       for c in CATEGORICAL_COLS_NATIVE if c in X_train.columns]

    cat_model = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.1,
        auto_class_weights='Balanced',
        cat_features=cat_feature_idx,
        random_seed=RANDOM_SEED, verbose=0,
    )
    cat_model.fit(X_train, y_enc, sample_weight=train_weights)
    print(f'  Training complete. ({len(train_combined):,} rows)')

    # ── Save artifacts ────────────────────────────────────────────────────────
    print(f'\n[6/6] Saving artifacts to {MODEL_DIR}/ ...')

    cat_model.save_model(os.path.join(MODEL_DIR, 'catboost_model'))
    print(f'  Saved: {MODEL_DIR}/catboost_model')

    save(le,                                       os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    save(cat_categories,                           os.path.join(MODEL_DIR, 'cat_categories.pkl'))
    save(list(X_train.columns),                    os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    save(batter_feats,                             os.path.join(MODEL_DIR, 'batter_features.pkl'))
    save(pitcher_feats_all,                        os.path.join(MODEL_DIR, 'pitcher_features.pkl'))
    save({'minority_label': minority_label,
          'classes': classes,
          'train_start': TRAIN_START,
          'train_end': TRAIN_END,
          'min_pa': MIN_PA,
          'top_n_batters': TOP_N_BATTERS,
          'top_batter_ids': set(top_batter_ids),
          'cat_feature_idx': cat_feature_idx},    os.path.join(MODEL_DIR, 'meta.pkl'))

    # Save a lean version of first_pitch for stand/p_throws lookups
    # (only the columns needed at prediction time)
    fp_slim = first_pitch[['batter', 'pitcher', 'stand', 'p_throws']].drop_duplicates()
    save(fp_slim,                                  os.path.join(MODEL_DIR, 'first_pitch_meta.pkl'))

    print(f'\n✓  All artifacts saved to {MODEL_DIR}/')
    print(f'   Run predict_first_pitch_global.py to make predictions.')


if __name__ == '__main__':
    main()
