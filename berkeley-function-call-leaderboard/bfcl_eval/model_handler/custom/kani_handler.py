import asyncio
import itertools
import threading
import time
from typing import Any

from kani import AIFunction, ChatMessage, ChatRole, ToolCall
from kani.ext.vllm import VLLMServerEngine
from kani.model_specific.qwen3 import Qwen3ThinkingParser
from kani.utils.cli import create_engine_from_cli_arg

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.custom.basekani import TokenCountingKani
from bfcl_eval.model_handler.custom.pydantic_generation import create_pydantic_model_from_json_schema
from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    convert_to_function_call,
    convert_to_tool,
    extract_system_prompt,
)
from bfcl_eval.utils import contain_multi_turn_interaction


class KaniBaseHandler(BaseHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, engine=None, **kwargs):
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        # compat
        self.model_style = ModelStyle.OPENAI_COMPLETIONS

        self.engine = engine
        self.thread_local = threading.local()

    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        # FC model
        if contain_multi_turn_interaction(test_entry["id"]):
            return self.inference_multi_turn_FC(test_entry, include_input_log, exclude_state_log)
        else:
            return self.inference_single_turn_FC(test_entry, include_input_log)

    def _query_FC(self, inference_data: dict):
        system_prompt = inference_data.get("system_prompt")
        messages = inference_data["messages"].copy()
        tools = inference_data["tools"].copy()
        # print(inference_data)

        # asyncio setup
        if not hasattr(self.thread_local, "loop"):
            self.thread_local.loop = asyncio.new_event_loop()

        ai = TokenCountingKani(self.engine, system_prompt=system_prompt, chat_history=messages, functions=tools)
        msgs = []

        async def _full_round():
            async for msg in ai.full_round(query=None):
                msgs.append(msg)

        start_time = time.monotonic()
        self.thread_local.loop.run_until_complete(_full_round())
        end_time = time.monotonic()

        return msgs, end_time - start_time

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["messages"] = []

        # extract system prompt to pin
        system_prompt = extract_system_prompt(test_entry["question"][0])
        if system_prompt is not None:
            inference_data["system_prompt"] = system_prompt

        # merge consecutive here
        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(test_entry["question"][round_idx])

        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        oai_tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        # convert openai-spec tools to AIFunctions
        # we want after=user so that we delegate the actual call to BFCL
        tools = []
        for oai_tool in oai_tools:
            oai_tool = oai_tool["function"]
            aif = AIFunction(
                lambda: None,
                after=ChatRole.USER,
                name=oai_tool["name"],
                desc=oai_tool["description"],
                json_schema=oai_tool["parameters"],
            )
            # hack: explicitly set aif.inner to a pydantic model's validate
            val_model = create_pydantic_model_from_json_schema(oai_tool["name"], oai_tool["parameters"])

            def _inner(**kwargs):
                val_model.model_validate(kwargs)
                return "[dummy response]"

            aif.inner = _inner
            tools.append(aif)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        # get all the valid tool calls
        valid_tc_ids: list[str] = [
            m.tool_call_id for m in api_response if m.role == ChatRole.FUNCTION and not m.is_tool_call_error
        ]
        all_tcs: list[ToolCall] = itertools.chain.from_iterable(
            m.tool_calls for m in api_response if m.role == ChatRole.ASSISTANT and m.tool_calls
        )
        valid_tcs = [tc for tc in all_tcs if tc.id in valid_tc_ids]

        model_responses = [{tc.function.name: tc.function.arguments} for tc in valid_tcs]
        tool_call_ids = [tc.id for tc in valid_tcs]

        # token counting
        prompt_tokens = sum(m.extra["prompt_tokens"] for m in api_response if "prompt_tokens" in m.extra)
        completion_tokens = sum(m.extra["completion_tokens"] for m in api_response if "completion_tokens" in m.extra)

        return {
            "model_responses": model_responses,
            "tool_call_ids": tool_call_ids,
            "input_token": prompt_tokens,
            "output_token": completion_tokens,
            "raw_msgs": api_response,
        }

    def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["messages"].extend(ChatMessage.model_validate(m) for m in first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["messages"].extend(ChatMessage.model_validate(m) for m in user_message)
        return inference_data

    def _add_assistant_message_FC(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["messages"].extend(model_response_data["raw_msgs"])
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:

        # edit the function messages if present, else append them
        for execution_result, tool_call_id in zip(execution_results, model_response_data["tool_call_ids"]):
            tool_message = ChatMessage(role=ChatRole.FUNCTION, content=execution_result, tool_call_id=tool_call_id)
            for idx, msg in enumerate(inference_data["messages"]):
                if msg.tool_call_id == tool_call_id:
                    inference_data["messages"][idx] = tool_message
                    break
            else:
                inference_data["messages"].append(tool_message)

        return inference_data

    def decode_execute(self, result, has_tool_call_tag):
        return convert_to_function_call(result)


class KaniHandler(KaniBaseHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        engine = create_engine_from_cli_arg(model_name)
        super().__init__(model_name, temperature, registry_name, is_fc_model, engine=engine, **kwargs)


# ===== model impls =====
class KaniQwen3VLLMHandler(KaniBaseHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        engine = VLLMServerEngine(
            model_id=model_name,
            vllm_args={
                "tensor_parallel_size": 8,
                "enable_chunked_prefill": True,
            },
            temperature=temperature,
        )
        engine.model = model_name
        engine = Qwen3ThinkingParser(engine)
        super().__init__(model_name, temperature, registry_name, is_fc_model, engine=engine, **kwargs)
