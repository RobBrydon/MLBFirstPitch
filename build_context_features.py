"""
build_context_features.py
─────────────────────────
Builds two sequential in-game context features for every pitch in a
Statcast parquet file and saves the result as a sidecar parquet:

  pitcher_pitch_count_in_game
      Running count of pitches the PITCHER has thrown so far
      in the current game, at the moment this pitch is delivered.
      Resets to 0 at the start of each new game.

  batter_prior_hip_count_in_game
      Running count of first-pitch hit-into-play events the BATTER
      has recorded so far in the current game, before this at-bat.
      Only counts first-pitch (balls==0, strikes==0) HIPs.
      Resets to 0 at the start of each new game.

Usage
─────
    python build_context_features.py <input_parquet> [output_parquet]

If output_parquet is omitted the sidecar is written next to the input
file with the suffix _context_features.parquet.

The output contains only three columns:
    pitch_id, pitcher_pitch_count_in_game, batter_prior_hip_count_in_game

Join it back onto your main table with:
    table = table.merge(context_df, on='pitch_id', how='left')
"""

import sys
import os
import pandas as pd
from collections import defaultdict

def build_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk every pitch in chronological order within each game and compute
    sequential in-game counts.

    Parameters
    ----------
    df : pd.DataFrame
        Full Statcast table.  Must contain:
            pitch_id, game_pk, at_bat_number, pitch_number,
            pitcher, batter, balls, strikes, description

    Returns
    -------
    pd.DataFrame with columns:
        pitch_id, pitcher_pitch_count_in_game, batter_prior_hip_count_in_game,
        pitcher_pitch_count_prior_game, batter_prior_game_hip_count
    """
    # Sort into strict pitch order within each game
    df_sorted = df.sort_values(
        ['game_pk', 'at_bat_number', 'pitch_number'],
        ascending=True
    ).reset_index(drop=True)

    n = len(df_sorted)
    pitcher_counts      = [0] * n   # pitches thrown by this pitcher so far this game
    batter_hip          = [0] * n   # first-pitch HIPs by this batter so far this game
    pitcher_prior_game  = [0] * n   # pitcher's total pitches in their prior game
    batter_prior_game   = [0] * n   # batter's first-pitch HIPs in their prior game

    # Running state keyed by game_pk (in-game)
    pitcher_thrown    = defaultdict(lambda: defaultdict(int))
    batter_hip_so_far = defaultdict(lambda: defaultdict(int))
    seen_games        = set()

    # Prior-game state: total pitches / HIPs from the PREVIOUS game
    # pitcher_game_totals[pitcher_id] = total pitches in last completed game
    # batter_game_hip_totals[batter_id] = first-pitch HIPs in last completed game
    pitcher_game_totals    = defaultdict(int)  # last game's total
    batter_game_hip_totals = defaultdict(int)  # last game's HIP total
    # Accumulate CURRENT game totals to shift into prior on game change
    pitcher_current_game   = defaultdict(lambda: defaultdict(int))  # game -> pitcher -> count
    batter_current_game    = defaultdict(lambda: defaultdict(int))  # game -> batter -> HIP count

    game_col   = df_sorted['game_pk'].values
    ab_col     = df_sorted['at_bat_number'].values
    pitch_col  = df_sorted['pitch_number'].values
    pitcher_col = df_sorted['pitcher'].values
    batter_col  = df_sorted['batter'].values
    balls_col   = df_sorted['balls'].values
    strikes_col = df_sorted['strikes'].values
    desc_col    = df_sorted['description'].values

    prev_game = None
    prev_ab   = None

    for idx in range(n):
        game    = game_col[idx]
        ab      = ab_col[idx]
        pitcher = pitcher_col[idx]
        batter  = batter_col[idx]
        balls   = balls_col[idx]
        strikes = strikes_col[idx]
        desc    = desc_col[idx]

        # ── Prior-game totals: assign before updating current game ────────────
        pitcher_prior_game[idx] = pitcher_game_totals[pitcher]
        batter_prior_game[idx]  = batter_game_hip_totals[batter]

        # ── Pitcher pitch count (in-game) ─────────────────────────────────────
        pitcher_counts[idx] = pitcher_thrown[game][pitcher]
        pitcher_thrown[game][pitcher] += 1
        pitcher_current_game[game][pitcher] += 1

        # ── Batter prior-game HIP count (in-game) ────────────────────────────
        batter_hip[idx] = batter_hip_so_far[game][batter]

        is_first_pitch = (balls == 0 and strikes == 0)
        is_new_ab      = (game != prev_game or ab != prev_ab)

        if is_first_pitch and is_new_ab and desc == 'hit_into_play':
            batter_hip_so_far[game][batter] += 1
            batter_current_game[game][batter] += 1

        # ── When game changes: shift current game totals into prior-game ──────
        if game != prev_game and prev_game is not None:
            for pid, count in pitcher_current_game[prev_game].items():
                pitcher_game_totals[pid] = count
            for bid, count in batter_current_game[prev_game].items():
                batter_game_hip_totals[bid] = count

        prev_game = game
        prev_ab   = ab

    result = pd.DataFrame({
        'pitch_id':                          df_sorted['pitch_id'].values,
        'pitcher_pitch_count_in_game':       pitcher_counts,
        'batter_prior_hip_count_in_game':    batter_hip,
        'pitcher_pitch_count_prior_game':    pitcher_prior_game,
        'batter_prior_game_hip_count':       batter_prior_game,
    })

    return result


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else (
        os.path.splitext(input_path)[0] + '_context_features.parquet'
    )

    print(f'Loading  : {input_path}')
    df = pd.read_parquet(input_path)

    # Ensure pitch_id is set
    if 'pitch_id' not in df.columns:
        df['pitch_id'] = df.index

    print(f'Rows     : {len(df):,}')
    print('Building sequential context features...')

    result = build_context_features(df)

    print(f'Saving   : {output_path}')
    result.to_parquet(output_path, index=False)

    print('Done.')
    print(result.describe().to_string())


if __name__ == '__main__':
    main()
