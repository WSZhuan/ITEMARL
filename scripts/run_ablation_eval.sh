#!/bin/bash
# scripts/run_ablation_eval.sh

set -e

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --experiment NAME    evaluate single experiment"
    echo "  --all                evaluate all ablation experiments"
    echo "  --episodes NUM       number of episodes to evaluate per experiment (default: 200)"
    echo "  --seeds SEEDS        evaluation seeds (default: 101,102,103,104,105)"
    echo "  --gpu GPU_ID         specify GPU ID"
    echo "  --help               display this help message"
    echo "  --seq_len LEN        sequence length"
}

#EXPERIMENT can use any name in configs/ablation.yaml,
# for example: "transformer_original", "thread_0", "thread_1", "thread_2",
#              "thread_3", "thread_4", "thread_5" "thread_6", "thread_7", "thread_8",
#              "curriculum_static", "curriculum_none"

EXPERIMENT=""
RUN_ALL=false
EPISODES=200
SEEDS="101,102,103,104,105"
SEQ_LEN=12
GPU_ID=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# set GPU
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "using GPU: $GPU_ID"
fi

# base directory
BASE_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$BASE_DIR"

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# run ablation experiment evaluation
if [ -n "$EXPERIMENT" ]; then
    echo "evaluating single experiment: $EXPERIMENT"
    python3 -u trainers/ablation_eval.py \
        --ablation_config configs/ablation.yaml \
        --env_config configs/env.yaml \
        --agent_config configs/agent.yaml \
        --train_config configs/train.yaml \
        --results_dir results \
        --episodes $EPISODES \
        --seeds $SEEDS \
        --experiment $EXPERIMENT \
        --seq_len $SEQ_LEN
elif [ "$RUN_ALL" = true ]; then
    echo "evaluating all ablation experiments"
    python3 -u trainers/ablation_eval.py \
        --ablation_config configs/ablation.yaml \
        --env_config configs/env.yaml \
        --agent_config configs/agent.yaml \
        --train_config configs/train.yaml \
        --results_dir results \
        --episodes $EPISODES \
        --seeds $SEEDS \
        --seq_len $SEQ_LEN
else
    echo "please specify --experiment NAME or --all"
    usage
    exit 1
fi