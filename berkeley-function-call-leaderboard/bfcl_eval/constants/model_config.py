from dataclasses import dataclass
from typing import Optional

from bfcl_eval.model_handler.custom.kani_handler import (
    KaniGPTOSSVLLMHandler,
    KaniGPTOSSVLLMNoRetryHandler,
    KaniLlama31VLLMHandler,
    KaniLlama31VLLMNoRetryHandler,
    KaniLlama32VLLMHandler,
    KaniLlama32VLLMNoRetryHandler,
    KaniQwen3VLLMHandler,
    KaniQwen3VLLMNoRetryHandler,
)
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler


# -----------------------------------------------------------------------------
# A mapping of model identifiers to their respective model configurations.
# Each key corresponds to the model id passed to the `--model` argument
# in both generation and evaluation commands.
# Make sure to update the `supported_models.py` file as well when updating this map.
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """
    Model configuration class for storing model metadata and settings.

    Attributes:
        model_name (str): Name of the model as used in the vendor API or on Hugging Face (may not be unique).
        display_name (str): Model name as it should appear on the leaderboard.
        url (str): Reference URL for the model or hosting service.
        org (str): Organization providing the model.
        license (str): License under which the model is released.
        model_handler (str): Handler name for invoking the model.
        input_price (Optional[float]): USD per million input tokens (None for open source models).
        output_price (Optional[float]): USD per million output tokens (None for open source models).
        is_fc_model (bool): True if this model is used in Function-Calling mode, otherwise False for Prompt-based mode.
        underscore_to_dot (bool): True if model does not support '.' in function names, in which case we will replace '.' with '_'. Currently this only matters for checker.  TODO: We should let the tool compilation step also take this into account.

    """

    model_name: str
    display_name: str
    url: str
    org: str
    license: str

    model_handler: str

    # Prices are in USD per million tokens; open source models have None
    input_price: Optional[float] = None
    output_price: Optional[float] = None

    # True if the model is in function-calling mode, False if in prompt mode
    is_fc_model: bool = True

    # True if this model does not allow '.' in function names
    underscore_to_dot: bool = False


# Inference through local hosting
local_inference_model_map = {
    # bfcl impl for reference
    "bfcl:Qwen/Qwen3-4B-FC": ModelConfig(
        model_name="Qwen/Qwen3-4B",
        display_name="Qwen3-4B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-4B",
        org="Qwen",
        license="apache-2.0",
        model_handler=QwenFCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    # kani, no retry
    "meta-llama/Llama-3.1-8B-Instruct-FC": ModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama-3.1-8B-Instruct (FC)",
        url="https://llama.meta.com/llama3",
        org="Meta",
        license="Meta Llama 3 Community",
        model_handler=KaniLlama31VLLMNoRetryHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "meta-llama/Llama-3.2-1B-Instruct-FC": ModelConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        display_name="Llama-3.2-1B-Instruct (FC)",
        url="https://llama.meta.com/llama3",
        org="Meta",
        license="Meta Llama 3 Community",
        model_handler=KaniLlama32VLLMNoRetryHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "meta-llama/Llama-3.2-3B-Instruct-FC": ModelConfig(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        display_name="Llama-3.2-3B-Instruct (FC)",
        url="https://llama.meta.com/llama3",
        org="Meta",
        license="Meta Llama 3 Community",
        model_handler=KaniLlama32VLLMNoRetryHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "Qwen/Qwen3-0.6B-FC": ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        display_name="Qwen3-0.6B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-0.6B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMNoRetryHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "Qwen/Qwen3-1.7B-FC": ModelConfig(
        model_name="Qwen/Qwen3-1.7B",
        display_name="Qwen3-1.7B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-1.7B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMNoRetryHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "Qwen/Qwen3-4B-FC": ModelConfig(
        model_name="Qwen/Qwen3-4B",
        display_name="Qwen3-4B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-4B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMNoRetryHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "Qwen/Qwen3-8B-FC": ModelConfig(
        model_name="Qwen/Qwen3-8B",
        display_name="Qwen3-8B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-8B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMNoRetryHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "Qwen/Qwen3-14B-FC": ModelConfig(
        model_name="Qwen/Qwen3-14B",
        display_name="Qwen3-14B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-14B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMNoRetryHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "Qwen/Qwen3-32B-FC": ModelConfig(
        model_name="Qwen/Qwen3-32B",
        display_name="Qwen3-32B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-32B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMNoRetryHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "openai/gpt-oss-20b-FC": ModelConfig(
        model_name="openai/gpt-oss-20b",
        display_name="openai/gpt-oss-20b (FC)",
        url="https://huggingface.co/openai/gpt-oss-20b",
        org="OpenAI",
        license="apache-2.0",
        model_handler=KaniGPTOSSVLLMNoRetryHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
}

third_party_inference_model_map = {
    # kani
    "kani:meta-llama/Llama-3.1-8B-Instruct-FC": ModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama-3.1-8B-Instruct (FC)",
        url="https://llama.meta.com/llama3",
        org="Meta",
        license="Meta Llama 3 Community",
        model_handler=KaniLlama31VLLMHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "kani:meta-llama/Llama-3.2-1B-Instruct-FC": ModelConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        display_name="Llama-3.2-1B-Instruct (FC)",
        url="https://llama.meta.com/llama3",
        org="Meta",
        license="Meta Llama 3 Community",
        model_handler=KaniLlama32VLLMHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "kani:meta-llama/Llama-3.2-3B-Instruct-FC": ModelConfig(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        display_name="Llama-3.2-3B-Instruct (FC)",
        url="https://llama.meta.com/llama3",
        org="Meta",
        license="Meta Llama 3 Community",
        model_handler=KaniLlama32VLLMHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "kani:Qwen/Qwen3-0.6B-FC": ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        display_name="Qwen3-0.6B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-0.6B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "kani:Qwen/Qwen3-1.7B-FC": ModelConfig(
        model_name="Qwen/Qwen3-1.7B",
        display_name="Qwen3-1.7B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-1.7B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "kani:Qwen/Qwen3-4B-FC": ModelConfig(
        model_name="Qwen/Qwen3-4B",
        display_name="Qwen3-4B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-4B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "kani:Qwen/Qwen3-8B-FC": ModelConfig(
        model_name="Qwen/Qwen3-8B",
        display_name="Qwen3-8B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-8B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "kani:Qwen/Qwen3-14B-FC": ModelConfig(
        model_name="Qwen/Qwen3-14B",
        display_name="Qwen3-14B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-14B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "kani:Qwen/Qwen3-32B-FC": ModelConfig(
        model_name="Qwen/Qwen3-32B",
        display_name="Qwen3-32B (FC)",
        url="https://huggingface.co/Qwen/Qwen3-32B",
        org="Qwen",
        license="apache-2.0",
        model_handler=KaniQwen3VLLMHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
    "kani:openai/gpt-oss-20b-FC": ModelConfig(
        model_name="openai/gpt-oss-20b",
        display_name="openai/gpt-oss-20b (FC)",
        url="https://huggingface.co/openai/gpt-oss-20b",
        org="OpenAI",
        license="apache-2.0",
        model_handler=KaniGPTOSSVLLMHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
}


MODEL_CONFIG_MAPPING = {
    **local_inference_model_map,
    **third_party_inference_model_map,
}

# Uncomment to get the supported_models.py file contents
# print(repr(list(MODEL_CONFIG_MAPPING.keys())))

# curl 127.0.0.1:8000/v1/completions \
# -H "Content-Type: application/json" \
# -H "Authorization: Bearer YOUR_API_KEY" \
# -d '{"model": "Qwen/Qwen3-4B-Thinking-2507", "prompt": "Say this is a test", "max_tokens": null}'
