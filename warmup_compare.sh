#!/bin/bash
#SBATCH --account=p53267
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=pcie
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
cd "$HOME/tensorrt_infer"
prefix="reports/warmup_${SLURM_JOB_ID}"
nvidia-smi --query-gpu=timestamp,uuid,name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv > "${prefix}_environment.csv"
nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv -l 1 > "${prefix}_telemetry.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
index=0
for ms in 0 1000 1000 0; do
    index=$((index + 1))
    ./build-gcc/vector_bench --sizes 1000003 --threads 128,256,512 \
      --mode both --repeat 200 --batch 100 --rounds 3 --warmup 50 \
      --warmup-ms "$ms" --out "${prefix}_${index}_${ms}" \
      > "${prefix}_${index}_${ms}.log"
done
