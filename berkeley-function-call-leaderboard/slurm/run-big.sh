#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=bfcl:big
#
#SBATCH --output=/nlpgpu/data/andrz/logs/%j.%x.log
#SBATCH --error=/nlpgpu/data/andrz/logs/%j.%x.log
#SBATCH --time=7-0
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --gpus=8
#SBATCH --constraint=48GBgpu
#SBATCH --mail-user=andrz@seas.upenn.edu
#SBATCH --mail-type=END,FAIL

bfcl generate \
  --model Qwen/Qwen3-32B-FC \
  --model kani:Qwen/Qwen3-32B-FC \
  --model openai/gpt-oss-120b-FC \
  --model kani:openai/gpt-oss-120b-FC \
  --test-category simple_python,simple_java,simple_javascript,parallel,multiple,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum \
  --num-threads 16
