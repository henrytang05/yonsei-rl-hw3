#!/usr/bin/env bash

set -e

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Use venv python directly (safer)
PY=".venv/bin/python"

commands=(
    "$PY cas4160/scripts/run_hw3.py -cfg experiments/dqn/cartpole.yaml --seed 1"
    "$PY cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 1"
    "$PY cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 2"
    "$PY cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist.yaml --seed 3"
    "$PY cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 1"
    "$PY cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 2"
    "$PY cas4160/scripts/run_hw3.py -cfg experiments/dqn/bankheist_ddqn.yaml --seed 3"
)

max_jobs=3

for i in "${!commands[@]}"; do
    cmd="${commands[$i]}"
    log_file="$LOG_DIR/run_$i.log"

    echo "Starting job $i"
    bash -c "$cmd" >"$log_file" 2>&1 &
    echo "PID $!"

    # 🔥 HARD LIMIT: wait until running jobs < max_jobs
    while (($(jobs -rp | wc -l) >= max_jobs)); do
        sleep 1
    done
done

wait
echo "All jobs completed."
