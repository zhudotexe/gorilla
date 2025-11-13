from pathlib import Path

TEMPLATE = """
#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name={model_name}-bfcl
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

bfcl generate --model {model_name} --num-threads {num_threads} --test-category simple_python,\
simple_java,simple_javascript,parallel,multiple,parallel_multiple,irrelevance,live_simple,live_multiple,\
live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,\
multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum
""".strip()
# exclude format_sensitivity and web_search

# models:
# qwen3 thinking 4b -- vllm tool parser hermes
# qwen3 1.7B, 4B, 8B, 14B -- vllm tool parser hermes
# mistral-small-2506-FC -- vllm tool parser mistral
# meta-llama/Llama-3.1-8B-Instruct-FC, meta-llama/Llama-3.2-1B-Instruct-FC, meta-llama/Llama-3.2-3B-Instruct-FC -- vllm tool parser llama
# gptoss 20b?
# also baselines w/o kani
MODEL_KEYS = [
    ("Qwen/Qwen3-0.6B-FC", 32),
    ("Qwen/Qwen3-1.7B-FC", 32),
    ("Qwen/Qwen3-4B-FC", 32),
    ("Qwen/Qwen3-8B-FC", 16),
    ("Qwen/Qwen3-14B-FC", 8),
    ("meta-llama/Llama-3.2-1B-Instruct-FC", 32),
    ("meta-llama/Llama-3.2-3B-Instruct-FC", 32),
    ("meta-llama/Llama-3.1-8B-Instruct-FC", 16),
    ("openai/gpt-oss-20b", 8),
]

SLURM_ROOT = Path(__file__).parent

for model, concurrency in MODEL_KEYS:
    model_fp_name = model.replace("/", "_")
    fp = SLURM_ROOT / f"gen/{model_fp_name}.sh"
    fp.write_text(TEMPLATE.format(model_name=model, num_threads=concurrency))
    fp2 = SLURM_ROOT / f"gen/kani-{model_fp_name}.sh"
    fp2.write_text(TEMPLATE.format(model_name=f"kani:{model}", num_threads=concurrency))
