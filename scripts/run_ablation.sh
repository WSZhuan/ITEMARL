#!/bin/bash
# scripts/run_ablation.sh

set -e

usage() {
    echo "usage: $0 [option]"
    echo "option:"
    echo "  --seq_len LENGTH     observation sequence length (default: 12)"
    echo "  --p_drop RATE        dropout rate (default: 0.2)"
    echo "  --experiment NAME    run a single experiment"
    echo "  --all                run all ablation experiments"
    echo "  --gpu GPU_ID         specify GPU ID"
    echo "  --help               display this help message"
}

#EXPERIMENT can use any name in configs/ablation.yaml,
# for example: "transformer_original", "thread_0", "thread_1", "thread_2",
#              "thread_3", "thread_4", "thread_5" "thread_6", "thread_7", "thread_8",
#              "curriculum_static", "curriculum_none"

SEQ_LEN=12
P_DROP=0.2
EXPERIMENT=""
RUN_ALL=false
GPU_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --seq_len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --p_drop)
            P_DROP="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
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

# run ablation experiments
if [ -n "$EXPERIMENT" ]; then
    echo "running single experiment: $EXPERIMENT"
    python3 -u trainers/ablation_runner.py \
        --config configs/ablation.yaml \
        --seq_len $SEQ_LEN \
        --p_drop $P_DROP \
        --experiment $EXPERIMENT
elif [ "$RUN_ALL" = true ]; then
    echo "running all ablation experiments"
    python3 -u trainers/ablation_runner.py \
        --config configs/ablation.yaml \
        --seq_len $SEQ_LEN \
        --p_drop $P_DROP
else
    echo "please specify --experiment NAME or --all"
    usage
    exit 1
fi