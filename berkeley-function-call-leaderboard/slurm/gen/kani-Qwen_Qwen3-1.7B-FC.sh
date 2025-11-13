#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=Qwen_Qwen3-1.7B-FC-bfcl
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

bfcl generate --model kani:Qwen/Qwen3-1.7B-FC --num-threads 32 --test-category simple_python,simple_java,simple_javascript,parallel,multiple,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum