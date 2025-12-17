#!/usr/bin/env python3
"""
plot_training_curves.py

example（cooperates with run_plot.sh）：
  python3 plot_training_curves.py --metric reward --smooth 0.9 --out /path/to/out.png file1.txt file2.txt ...

input file format（each line）:
  <episode>,<value>
example:
  0,12.5
  1,13.1
  2,11.8
blank lines or lines starting with # will be skipped.

smoothing (EMA) implementation:
  s[0] = x[0]
  s[t] = alpha * s[t-1] + (1-alpha) * x[t]
parameter --smooth (alpha) ranges [0,1], default 0.9 (strong smoothing)
"""
import argparse
import os
import re
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# --- friendly name mapping (can be extended as needed) ---
FRIENDLY = {
    'tosrl_td3': 'TOSRL_TD3',
    'tosrl_sac': 'TOSRL_SAC',
    'tosrl_td3bc': 'TOSRL_TD3BC',
    'lstm_td3': 'LSTM_TD3',
    'lstm_sac': 'LSTM_SAC',
    'itemarl_td3': 'ITEMARL_TD3(ours)',
}

# regex to strip suffix like _L8_P0.8 etc.
SUFFIX_RE = re.compile(r'(_L\d+_P[0-9.]+)|(_L\d+)|(_P[0-9.]+)', flags=re.IGNORECASE)


def parse_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse file of lines "ep,val" -> return (episodes, values) as numpy arrays.
    Skips blank lines and comments (#).
    """
    eps = []
    vals = []
    with open(path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            # tolerate "ep,val" or single value per line
            parts = [p.strip() for p in ln.split(',') if p.strip() != '']
            if len(parts) == 0:
                continue
            try:
                if len(parts) == 1:
                    # assume sequential episodes if single column
                    vals.append(float(parts[0]))
                else:
                    eps.append(int(float(parts[0])))
                    vals.append(float(parts[1]))
            except Exception:
                # tolerate lines like "0 12.3"
                sp = re.split(r'\s+', ln)
                if len(sp) >= 2:
                    try:
                        eps.append(int(float(sp[0])))
                        vals.append(float(sp[1]))
                    except Exception:
                        continue
                else:
                    continue

    if len(vals) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if len(eps) == 0:
        # only values present -> use 0..N-1
        eps = list(range(len(vals)))

    return np.array(eps, dtype=int), np.array(vals, dtype=float)


def ema_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average smoothing with weight alpha on previous (0..1)."""
    if x.size == 0:
        return x
    s = np.empty_like(x, dtype=float)
    s[0] = x[0]
    for i in range(1, len(x)):
        s[i] = alpha * s[i - 1] + (1.0 - alpha) * x[i]
    return s


def clean_name_from_path(p: str) -> str:
    # basename
    b = os.path.basename(p)
    # strip extension
    b = os.path.splitext(b)[0]
    # if path contains folder like .../itemarl_td3_L8_P0.8/logs/reward.txt, try to pick parent folder name
    # inspect parent folder
    parent = os.path.basename(os.path.dirname(os.path.dirname(p)))  # go up two levels: .../algo_L.../logs/file
    if parent:
        # choose algorithm-like name if parent contains underscore + L/P
        if '_' in parent:
            b = parent

    # remove suffix patterns like _L8_P0.8 or _L8 or _P0.8
    b_clean = SUFFIX_RE.sub('', b)
    b_clean = b_clean.strip('_- ')
    # attempt friendly map
    key = b_clean.lower()
    if key in FRIENDLY:
        nice = FRIENDLY[key]
    else:
        # fallback: replace _ with space and uppercase
        nice = b_clean.replace('_', ' ').upper()
    # final uppercase as requested
    return nice


def choose_yticks_for_metric(vals: np.ndarray, metric: str) -> List[float]:
    """Simple heuristic for y ticks (not overcomplicated)."""
    if vals.size == 0:
        return []
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    rng = vmax - vmin
    if rng <= 0:
        # flat line
        return [vmin]
    # choose number of ticks like 5
    step = rng / 4.0
    ticks = [vmin + i * step for i in range(5)]
    return ticks


def main():
    parser = argparse.ArgumentParser(description="Plot smoothed training curves from metric files.")
    parser.add_argument('--metric', required=True, choices=['reward', 'track_rate', 'tracking_rate', 'trackrate'],
                        help='metric name (affects ylabel)')
    parser.add_argument('--smooth', type=float, default=0.9, help='EMA alpha for smoothing (0..1). default 0.9')
    parser.add_argument('--out', required=True, help='Output PNG file path')
    parser.add_argument('inputs', nargs='+', help='Input metric files (positional). Use run_plot.sh to collect them.')
    args = parser.parse_args()

    metric = args.metric
    alpha = float(args.smooth)
    outpath = args.out
    input_files = args.inputs

    if alpha < 0.0 or alpha > 1.0:
        print("smooth must be in [0,1].")
        sys.exit(2)

    plt.figure(figsize=(10, 6), dpi=300)

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    additional_colors = [ '#8a2be2', '#e41a1c', '#ffd700','#333333']
    custom_colors = default_colors + additional_colors

    plt.gca().set_prop_cycle(color=custom_colors)

    all_eps_min = None
    all_eps_max = None
    plotted = 0

    for fpath in input_files:
        if not os.path.isfile(fpath):
            print(f"[WARN] missing file: {fpath}  (skipping)")
            continue
        eps, vals = parse_file(fpath)
        if vals.size == 0:
            print(f"[WARN] no data in: {fpath} (skipping)")
            continue

        # Ensure eps monotonic increasing; if not, sort by eps
        if not np.all(np.diff(eps) >= 0):
            order = np.argsort(eps)
            eps = eps[order]
            vals = vals[order]

        sm = ema_smooth(vals, alpha=alpha)

        # normalize x axis to episode indices (we assume eps are already episode numbers)
        x = eps

        # plot only smoothed curve
        plt.plot(x, sm, linewidth=2, label=clean_name_from_path(fpath))

        if all_eps_min is None:
            all_eps_min = int(np.min(x))
            all_eps_max = int(np.max(x))
        else:
            all_eps_min = min(all_eps_min, int(np.min(x)))
            all_eps_max = max(all_eps_max, int(np.max(x)))
        plotted += 1

    if plotted == 0:
        print("[ERROR] No valid files to plot.")
        sys.exit(3)

    # x-axis padding (5% each side, at least 1)
    x_range = max(1, all_eps_max - all_eps_min)
    pad = max(1, int(round(x_range * 0.05)))
    x_min = all_eps_min - pad
    x_max = all_eps_max + pad
    plt.xlim(x_min, x_max)

    # x ticks every 200 (or adapt)
    if x_max - x_min <= 200:
        xtick_step = max(1, int(round((x_max - x_min) / 5)))
    else:
        xtick_step = 200
    xticks = list(range(max(0, x_min), x_max + 1, xtick_step))
    if len(xticks) < 2:
        xticks = [x_min, x_max]
    plt.xticks(xticks)

    # labels & grid
    ylabel = "Reward" if metric.startswith('reward') else "Tracking Rate"
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(os.path.splitext(os.path.basename(outpath))[0])

    plt.grid(True, linestyle='--', alpha=0.4)

    # legend top-left, small font, uppercase names already applied in clean_name
    plt.legend(loc='upper left', fontsize='small')

    # autoscale y with small margin
    ymin, ymax = plt.ylim()
    yrng = ymax - ymin
    if yrng <= 0:
        ymin = ymin - 1.0
        ymax = ymax + 1.0
    else:
        ymin = ymin - 0.05 * yrng
        ymax = ymax + 0.05 * yrng
    plt.ylim(ymin, ymax)

    # save
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.tight_layout()
    # plt.savefig(outpath, bbox_inches='tight')
    # print(f"[plot_training_curves] Saved: {outpath}")
    base_path = os.path.splitext(outpath)[0]

    png_path = base_path + '.png'
    plt.savefig(png_path, bbox_inches='tight')
    print(f"[plot_training_curves] Saved: {png_path}")

    pdf_path = base_path + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"[plot_training_curves] Saved: {pdf_path}")


if __name__ == '__main__':
    main()
