#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=qwen3-bfcl
#
#SBATCH --output=/nlpgpu/data/andrz/logs/%j.%x.log
#SBATCH --error=/nlpgpu/data/andrz/logs/%j.%x.log
#SBATCH --time=7-0
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH --gpus=8
#SBATCH --constraint=48GBgpu
#SBATCH --mail-user=andrz@seas.upenn.edu
#SBATCH --mail-type=END,FAIL

# TP 8, 8x concurrency
bfcl generate --model kani-qwen3-4b-instruct-FC --include-input-log --num-threads 16
