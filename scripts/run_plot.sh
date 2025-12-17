#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: bash run_plot.sh SEQ PDROP METRIC
  SEQ     sequence length (e.g. 8)
  PDROP   p_drop value (e.g. 0.8 or 0.0)
  METRIC  metric name file to plot (reward or track_rate)

Optional environment variables:
  RESULTS_DIR  (default: /home/ubuntu/stu_data/wszhuan_projects/ITEMARL/results)
  PYTHON       (default: python3)
EOF
  exit 1
}

if [ "$#" -lt 3 ]; then
  echo "Error: missing args."
  usage
fi

SEQ=$1
PDROP=$2
METRIC=$3

# configurable
RESULTS_DIR=${RESULTS_DIR:-/home/ubuntu/stu_data/wszhuan_projects/ITEMARL/results}
PYTHON=${PYTHON:-python3}
PLOT_SCRIPT=${PLOT_SCRIPT:-utils/plot_training_curves.py}
SMOOTH=${SMOOTH:-0.99}  # EMA alpha default (can override in env)

# list of algorithm folder name prefixes to check (edit as needed)
ALGS=(
  "itemarl_td3"
  "tosrl_td3" "tosrl_sac" "tosrl_td3bc"
  "lstm_td3" "lstm_sac"
)

OUTFILE="${RESULTS_DIR}/L${SEQ}P${PDROP}${METRIC}.png"

echo "[run_plot] SEQ=${SEQ}, PDROP=${PDROP}, METRIC=${METRIC}"
echo "[run_plot] RESULTS_DIR=${RESULTS_DIR}"
echo "[run_plot] Output: ${OUTFILE}"
echo "[run_plot] Looking for metric files..."

# collect existing metric files
INPUTS=()
for a in "${ALGS[@]}"; do
  f="${RESULTS_DIR}/${a}_L${SEQ}_P${PDROP}/logs/${METRIC}.txt"
  if [ -f "${f}" ]; then
    INPUTS+=("${f}")
    echo "  found: ${f}"
  else
    echo "  missing: ${f}"
  fi
done

if [ "${#INPUTS[@]}" -eq 0 ]; then
  echo "[run_plot] No input metric files found for seq=${SEQ}, p_drop=${PDROP}, metric=${METRIC}."
  exit 2
fi

# Ensure python script exists (relative to current working dir)
if [ ! -f "${PLOT_SCRIPT}" ]; then
  echo "[run_plot] ERROR: plotting script '${PLOT_SCRIPT}' not found in current dir."
  echo "Place the plotting script 'plot_training_curves.py' in this directory or set PLOT_SCRIPT env var."
  exit 3
fi

# Run the plotting script, passing files as positional args (Scheme B)
echo "[run_plot] Running: ${PYTHON} ${PLOT_SCRIPT} --metric ${METRIC} --smooth ${SMOOTH} --out ${OUTFILE} ${INPUTS[*]}"
# Use -- so that any filenames starting with - won't be treated as options (defensive)
${PYTHON} "${PLOT_SCRIPT}" --metric "${METRIC}" --smooth "${SMOOTH}" --out "${OUTFILE}" -- "${INPUTS[@]}"

echo "[run_plot] Done. Output saved to: ${OUTFILE}"
