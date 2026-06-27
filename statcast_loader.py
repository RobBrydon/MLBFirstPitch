"""
statcast_loader.py
──────────────────
Download, cache, and load Statcast pitch-by-pitch data.

Usage
-----
From a Jupyter notebook or script:

    from statcast_loader import load_statcast

    # Training window
    table = load_statcast('2024-03-28', '2025-06-01')

    # Validation window
    table_val = load_statcast('2025-06-02', '2025-09-28')

How it works
------------
On the first call for a given date range the function fetches from Baseball
Savant via pybaseball and saves the result as a Parquet file next to this
script.  Every subsequent call loads from that file instead of hitting the
network, so it is instant regardless of how many times you restart the kernel.

Parquet is used instead of CSV because:
  • ~5-10× smaller on disk (columnar compression)
  • Preserves dtypes exactly (no silent int→float coercion on reload)
  • Loads ~10× faster than pd.read_csv for a file this size

Cache files are named:
    statcast_YYYYMMDD_YYYYMMDD.parquet

Force a fresh download by passing force=True or by deleting the .parquet file.
"""

import re
import warnings
from pathlib import Path

import pandas as pd

# Directory where .parquet cache files are stored (same folder as this script)
_CACHE_DIR = Path(__file__).parent


def _cache_path(start_dt: str, end_dt: str) -> Path:
    """Return the parquet cache file path for this date range."""
    start_slug = re.sub(r'[^0-9]', '', start_dt)
    end_slug   = re.sub(r'[^0-9]', '', end_dt)
    return _CACHE_DIR / f'statcast_{start_slug}_{end_slug}.parquet'


def load_statcast(
    start_dt: str,
    end_dt: str,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Return a DataFrame of all Statcast pitches between start_dt and end_dt.

    Parameters
    ----------
    start_dt : str
        First date to include, 'YYYY-MM-DD'.
    end_dt : str
        Last date to include, 'YYYY-MM-DD'.
    force : bool
        If True, re-download even if a cached file exists.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        Full pitch-by-pitch Statcast table for the requested window.
    """
    cache_file = _cache_path(start_dt, end_dt)

    # ── Load from cache if available ─────────────────────────────────────
    if cache_file.exists() and not force:
        if verbose:
            size_mb = cache_file.stat().st_size / 1_048_576
            print(f'Loading from cache: {cache_file.name}  ({size_mb:.1f} MB)')
        table = pd.read_parquet(cache_file)
        if verbose:
            print(f'Loaded {len(table):,} pitches  ({start_dt} → {end_dt})')
        return table

    # ── Download from Baseball Savant ────────────────────────────────────
    if verbose:
        print(f'No cache found for {start_dt} → {end_dt}.')
        print('Downloading from Baseball Savant — this may take several minutes ...')

    warnings.filterwarnings('ignore')

    try:
        from pybaseball import statcast, cache as pyb_cache
        pyb_cache.enable()
    except ImportError as exc:
        raise ImportError(
            'pybaseball is required. Install it with: pip install pybaseball'
        ) from exc

    table = statcast(start_dt, end_dt)

    if verbose:
        print(f'Downloaded {len(table):,} pitches.')

    # ── Save to parquet ───────────────────────────────────────────────────
    table.to_parquet(cache_file, index=False)
    if verbose:
        size_mb = cache_file.stat().st_size / 1_048_576
        print(f'Saved to cache: {cache_file.name}  ({size_mb:.1f} MB)')

    return table


def list_cached_ranges(verbose: bool = True) -> list[dict]:
    """
    Show all cached date ranges available in the cache directory.

    Returns a list of dicts with keys: file, start, end, size_mb, rows.
    """
    files = sorted(_CACHE_DIR.glob('statcast_*.parquet'))
    results = []
    for f in files:
        match = re.search(r'statcast_(\d{8})_(\d{8})\.parquet', f.name)
        if not match:
            continue
        start = f'{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}'
        end   = f'{match.group(2)[:4]}-{match.group(2)[4:6]}-{match.group(2)[6:]}'
        size_mb = f.stat().st_size / 1_048_576
        results.append({'file': f.name, 'start': start, 'end': end, 'size_mb': size_mb})
        if verbose:
            print(f'  {f.name}  {start} → {end}  ({size_mb:.1f} MB)')
    if not results and verbose:
        print('  No cached files found.')
    return results


if __name__ == '__main__':
    # Run as a script to pre-download both windows used by the model notebooks.
    # Example:
    #   python statcast_loader.py
    print('=== Statcast Data Loader ===')
    print()

    # Training window
    print('── Training window ──')
    load_statcast('2024-03-28', '2025-06-01')
    print()

    # Validation window
    print('── Validation window ──')
    load_statcast('2025-06-02', '2025-09-28')
    print()

    print('── Cached files ──')
    list_cached_ranges()
