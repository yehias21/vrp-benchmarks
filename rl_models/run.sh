#!/bin/bash
export CUDA_VISIBLE_DEVICES=5,6,7,1

# Define problem sizes in ascending order (including 200 & 500 to match batch‐size map)
sizes=(10 20 50 100 200 500)

# Training function (foreground)
train() {
  local num_loc=$1
  local algo=$2
  local batch_size

  # Map num_loc → batch_size
  case "$num_loc" in
    10)  batch_size=512  ;;
    20)  batch_size=256  ;;
    50)  batch_size=128  ;;
    100) batch_size=64   ;;
    200) batch_size=512  ;;
    500) batch_size=160  ;;
    *)
      echo "Error: no batch size defined for num_loc=$num_loc"
      exit 1
      ;;
  esac

  echo "[START] variant=twvrp algo=${algo} num_loc=${num_loc} batch_size=${batch_size}"
  python train.py \
    --variant    twvrp \
    --num_loc    "${num_loc}" \
    --algo       "${algo}" \
    --batch_size "${batch_size}"
}

# Uncomment to run Attention experiments:
# for size in "${sizes[@]}"; do
#   train "$size" "attention"
# done

# Run POMO experiments for TW-VRP:
for size in "${sizes[@]}"; do
  train "$size" "pomo"
done
