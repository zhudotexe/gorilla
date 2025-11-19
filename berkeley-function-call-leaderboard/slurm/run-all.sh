#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=bfcl:Qwen/Qwen3-4B-FC-bfcl
#
#SBATCH --output=/nlpgpu/data/andrz/logs/%j.%x.log
#SBATCH --error=/nlpgpu/data/andrz/logs/%j.%x.log
#SBATCH --time=7-0
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --gpus=8
#SBATCH --constraint=48GBgpu
#SBATCH --mail-user=andrz@seas.upenn.edu
#SBATCH --mail-type=END,FAIL


#  --model meta-llama/Llama-3.2-1B-Instruct-FC \
#  --model meta-llama/Llama-3.2-3B-Instruct-FC \
#  --model meta-llama/Llama-3.1-8B-Instruct-FC \
#  --model kani:meta-llama/Llama-3.2-1B-Instruct-FC \
#  --model kani:meta-llama/Llama-3.2-3B-Instruct-FC \
#  --model kani:meta-llama/Llama-3.1-8B-Instruct-FC \

bfcl generate \
  --model bfcl:Qwen/Qwen3-4B-FC \
  --model Qwen/Qwen3-0.6B-FC \
  --model Qwen/Qwen3-1.7B-FC \
  --model Qwen/Qwen3-4B-FC \
  --model Qwen/Qwen3-8B-FC \
  --model Qwen/Qwen3-14B-FC \
  --model openai/gpt-oss-20b-FC \
  --model kani:Qwen/Qwen3-0.6B-FC \
  --model kani:Qwen/Qwen3-1.7B-FC \
  --model kani:Qwen/Qwen3-4B-FC \
  --model kani:Qwen/Qwen3-8B-FC \
  --model kani:Qwen/Qwen3-14B-FC \
  --model kani:openai/gpt-oss-20b-FC \
  --test-category simple_python,simple_java,simple_javascript,parallel,multiple,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum \
  --num-threads 64
