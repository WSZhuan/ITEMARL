#!/usr/bin/env bash
# Usage:
#   bash scripts/run_eval.sh <seq_len> <p_drop> [agent_type]
# Examples:
#   bash scripts/run_eval.sh 8 0.5 itemarl_td3
#   bash scripts/run_eval.sh 12 0.8 itemarl_td3
#
# Arguments:
#   $1: seq_len        observation sequence length
#   $2: p_drop        information dropout probability
#   $3: agent_type    optional, default itemarl_td3


set -e

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <seq_len> <p_drop> [agent_type]"
  echo "  agent_type: optional, default itemarl_td3"
  exit 1
fi

SEQ_LEN=$1
P_DROP=$2
AGENT_TYPE=${3:-itemarl_td3}

# check agent_type
ALLOWED_AGENTS=("tosrl_sac" "tosrl_td3" "tosrl_td3bc" \
                "itemarl_td3" "lstm_td3" "lstm_sac"  )

if [[ ! " ${ALLOWED_AGENTS[@]} " =~ " ${AGENT_TYPE} " ]]; then
  echo "ERROR: Unknown agent_type '${AGENT_TYPE}'"
  echo "Allowed: ${ALLOWED_AGENTS[*]}"
  exit 1
fi

# check checkpoint
export PYTHONPATH="$(cd "$(dirname "$0")"/.. && pwd)"
BASE_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
RESULT_DIR="${BASE_DIR}/results/${AGENT_TYPE}_L${SEQ_LEN}_P${P_DROP}"
CHECKPOINT="${RESULT_DIR}/checkpoints/latest.pt"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

echo "============================================"
echo " Evaluating ${AGENT_TYPE}, seq_len=${SEQ_LEN}, p_drop=${P_DROP}"
echo " Checkpoint: ${CHECKPOINT}"
echo "============================================"

python3 -u "${BASE_DIR}/trainers/eval.py" \
  --env_config       "${BASE_DIR}/configs/env.yaml" \
  --agent_config     "${BASE_DIR}/configs/agent.yaml" \
  --train_config     "${BASE_DIR}/configs/train.yaml" \
  --model_checkpoint "${CHECKPOINT}" \
  --episodes         200 \
  --seed             36 \
  --seq_len          "${SEQ_LEN}" \
  --p_drop           "${P_DROP}" \
  --agent_type       "${AGENT_TYPE}" \
  --num_obstacles    10 \
  --seeds            101,102,103,104,105
