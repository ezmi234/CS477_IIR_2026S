#!/usr/bin/env bash
set -euo pipefail

SCENES="${SCENES:-200}"
START_SEED="${START_SEED:-1}"
SPLIT="${SPLIT:-train}"
OUT_ROOT="${OUT_ROOT:-/home/ubuntu/cs477_ws/datasets/gazebo_yolo_five_objects}"
SETTLE_SEC="${SETTLE_SEC:-8}"

for ((i=0; i<SCENES; i++)); do
  seed=$((START_SEED + i))
  echo "[dataset-loop] launching Gazebo random scene seed=${seed}"
  ros2 launch manip_challenge ur5_setup_random_picking.launch.py random_seed:="${seed}" &
  sim_pid=$!

  cleanup() {
    kill "${sim_pid}" 2>/dev/null || true
    wait "${sim_pid}" 2>/dev/null || true
  }
  trap cleanup EXIT

  sleep "${SETTLE_SEC}"
  ros2 run team_1 collect_gazebo_yolo_dataset \
    --scenes 1 \
    --scene-start "$((i + 1))" \
    --split "${SPLIT}" \
    --out-root "${OUT_ROOT}" \
    --cameras top,wrist \
    --bbox-mode pointcloud_visible_box \
    --save-debug

  cleanup
  trap - EXIT
  sleep 2
done

ros2 run team_1 check_gazebo_yolo_dataset --dataset "${OUT_ROOT}" --min-instances-per-class 1
