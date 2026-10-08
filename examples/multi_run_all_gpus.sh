#!/bin/bash
# Generic pyKT multi-model sweep launcher with explicit GPU list support.
# Force W&B mirror for restricted server environments.
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

# Usage:
#   sh multi_run_all_gpus.sh {dataset} "{models}" {task_name} {log_name} {gpu_ids} [batch_size]
# Example:
#   sh multi_run_all_gpus.sh xes3g5m "akt,simplekt,folibikt,stablekt" classic_xes xes_classic 0,1,2,3,4 128

dataset=$1
models=$2
task_name=$3
log_name=$4
gpu_ids=$5
batch_size=${6:-128}

echo "Input params is: "
echo dataset=$dataset
echo models=$models
echo task_name=$task_name
echo log_name=$log_name
echo gpu_ids=$gpu_ids
echo batch_size=$batch_size

echo "Start generate yamls"
python generate_wandb.py \
  --model_names "$models" \
  --dataset_names "$dataset" \
  --project_name "$task_name" \
  --src_dir seedwandb \
  --batch_size "$batch_size" \
  --all_dir "$task_name"

echo "Start launch sweeps"
sh all_start.sh > "${log_name}.log" 2>&1

echo "Start find agents' command"
IFS=','
for i in $models; do
  echo "# $i"
  sh run_all.sh "${log_name}.log" 0 5 "$dataset" "$i" "$gpu_ids" "$task_name-$dataset"
  echo ""
done
