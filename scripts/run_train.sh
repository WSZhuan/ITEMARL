#!/usr/bin/env bash
# Usage:
#   bash scripts/run_train.sh <seq_len> <p_drop> [agent_type]
# Examples:
#   bash scripts/run_train.sh 8 0.5 itemarl_td3
#   bash scripts/run_train.sh 12 0.8 itemarl_td3
# Arguments:
#   $1: observation_seq_len (e.g. 8 or 12)
#   $2: p_drop (e.g. 0.0, 0.5, 0.8)
#   $3: agent_type (e.g. itemarl_td3)

set -e

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <seq_len> <p_drop> [agent_type]"
  echo "Example: $0 8 0.5 itemarl_td3"
  exit 1
fi

SEQ_LEN=$1
P_DROP=$2
AGENT_TYPE=${3:-itemarl_td3}

# Optional: validate AGENT_TYPE against allowed list
ALLOWED_AGENTS=("tosrl_sac" "tosrl_td3" "tosrl_td3bc" \
                "itemarl_td3" "lstm_td3" "lstm_sac"  )

if [[ ! " ${ALLOWED_AGENTS[@]} " =~ " ${AGENT_TYPE} " ]]; then
  echo "ERROR: Unknown agent_type '${AGENT_TYPE}'"
  echo "Allowed: ${ALLOWED_AGENTS[*]}"
  exit 1
fi

# Ensure project root on PYTHONPATH
export PYTHONPATH="$(cd "$(dirname "$0")"/.. && pwd)"

BASE_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
OUTPUT_DIR="${BASE_DIR}/results/${AGENT_TYPE}_L${SEQ_LEN}_P${P_DROP}"

echo "============================================"
echo " Training ${AGENT_TYPE} with seq_len=${SEQ_LEN}, p_drop=${P_DROP}"
echo " Output dir: ${OUTPUT_DIR}"
echo "============================================"

python3 -u "${BASE_DIR}/trainers/train.py" \
  --env_config    "${BASE_DIR}/configs/env.yaml" \
  --agent_config  "${BASE_DIR}/configs/agent.yaml" \
  --train_config  "${BASE_DIR}/configs/train.yaml" \
  --output_dir    "${OUTPUT_DIR}" \
  --seq_len       "${SEQ_LEN}" \
  --p_drop        "${P_DROP}" \
  --agent_type    "${AGENT_TYPE}"