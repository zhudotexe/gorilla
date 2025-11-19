import asyncio
import functools
import itertools
import json
import signal
import threading
import time
from pathlib import Path
from typing import Any

from kani import AIFunction, ChatMessage, ChatRole, ToolCall
from kani.ext.vllm import VLLMOpenAIEngine, VLLMServerEngine
from kani.ext.vllm.vllm_server import VLLMServer
from kani.model_specific.gpt_oss import GPTOSSParser
from kani.model_specific.qwen3 import Qwen3Parser
from kani.utils.cli import print_width
from kani.utils.message_formatters import assistant_message_contents_thinking

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

REPO_ROOT = Path(__file__).parents[3]
DEBUG_PRINT = False


class KaniBaseHandler(BaseHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        temperature = max(temperature, 0.01) if temperature != 0 else 0  # silence a vllm warning
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        # compat
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.thread_local = threading.local()

    def _create_engine(self):
        raise NotImplementedError

    def _ensure_engine(self):
        if not hasattr(self.thread_local, "engine"):
            self.thread_local.engine = self._create_engine()

    @property
    def engine(self):
        # thread-local engine
        return self.thread_local.engine

    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        # asyncio setup
        # make a thread local event loop
        if not hasattr(self.thread_local, "loop"):
            self.thread_local.loop = asyncio.new_event_loop()
        self._ensure_engine()

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

        # main generation
        ai = TokenCountingKani(self.engine, system_prompt=system_prompt, chat_history=messages, functions=tools)
        msgs = []

        async def _full_round():
            async for msg in ai.full_round(query=None, max_function_rounds=inference_data.get("max_function_rounds")):
                if DEBUG_PRINT and (
                    text := assistant_message_contents_thinking(msg, show_args=True, show_reasoning=True)
                ):
                    print_width(text, prefix="AI: ")
                msgs.append(msg)

        start_time = time.monotonic()
        self.thread_local.loop.run_until_complete(_full_round())
        end_time = time.monotonic()

        # save for logging
        log_dir = REPO_ROOT / "result" / self.registry_dir_name / "_kani_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ai.save(log_dir / f"{inference_data['test_entry_id']}__round_{inference_data['round_num']}.json")
        inference_data["round_num"] += 1

        return msgs, end_time - start_time

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["test_entry_id"] = test_entry["id"]
        inference_data["round_num"] = 0
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

        def _validate_model(model, **kwargs):
            model.model_validate(kwargs)
            return "[dummy response]"

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
            aif.inner = functools.partial(_validate_model, val_model)
            tools.append(aif)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        # get all the tool calls that were not retried by kani
        invalid_tc_ids: list[str] = [
            m.tool_call_id for m in api_response if m.role == ChatRole.FUNCTION and m.is_tool_call_error
        ]
        all_tcs: list[ToolCall] = itertools.chain.from_iterable(
            m.tool_calls for m in api_response if m.role == ChatRole.ASSISTANT and m.tool_calls
        )
        valid_tcs = [tc for tc in all_tcs if tc.id not in invalid_tc_ids]

        model_responses = [{tc.function.name: tc.function.arguments} for tc in valid_tcs]
        tool_call_ids = [tc.id for tc in valid_tcs]

        # it expects the content of the last message if no tool calls
        if not model_responses:
            model_responses = [m for m in api_response if m.role == ChatRole.ASSISTANT][-1].text
            tool_call_ids = []

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
        for m in first_turn_message:
            msg = ChatMessage.model_validate(m)
            if DEBUG_PRINT:
                print_width(msg.text, prefix="USER: ")
            inference_data["messages"].append(msg)
        return inference_data

    def _add_next_turn_user_message_FC(self, inference_data: dict, user_message: list[dict]) -> dict:
        for m in user_message:
            msg = ChatMessage.model_validate(m)
            if DEBUG_PRINT:
                print_width(msg.text, prefix="USER: ")
            inference_data["messages"].append(msg)
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

    def decode_ast(self, result, language, has_tool_call_tag):
        decoded_output = []
        for invoked_function in result:
            name = list(invoked_function.keys())[0]
            params = json.loads(invoked_function[name])
            decoded_output.append({name: params})
        return decoded_output


class KaniNoRetryHandler(KaniBaseHandler):
    def _query_FC(self, inference_data: dict):
        inference_data["max_function_rounds"] = 0
        return super()._query_FC(inference_data)

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        oai_tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        # convert openai-spec tools to AIFunctions
        # we want after=user so that we delegate the actual call to BFCL
        tools = []
        for oai_tool in oai_tools:
            oai_tool = oai_tool["function"]
            aif = AIFunction(
                lambda *_, **__: None,
                after=ChatRole.USER,
                name=oai_tool["name"],
                desc=oai_tool["description"],
                json_schema=oai_tool["parameters"],
            )
            tools.append(aif)

        inference_data["tools"] = tools

        return inference_data


# ===== model impls =====
# ---- vllm ----
class KaniVLLMHandler(KaniBaseHandler):
    vllm_args: dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_process = None

    def start_managed_engine(self):
        self.vllm_process = VLLMServer(model_id=self.model_name, vllm_args=self.vllm_args)
        self.vllm_process.start()

    def stop_managed_engine(self):
        self.vllm_process.process.send_signal(signal.SIGINT)


class KaniLlama31VLLMHandler(KaniVLLMHandler):
    vllm_args = {
        "tensor_parallel_size": 8,
        "enable_chunked_prefill": True,
        "enable-auto-tool-choice": True,
        "tool-call-parser": "llama3_json",
    }

    def _create_engine(self):
        return VLLMOpenAIEngine(
            model_id=self.model_name,
            vllm_port=self.vllm_process.port,
            temperature=self.temperature,
            use_managed_server=False,
        )


class KaniLlama32VLLMHandler(KaniVLLMHandler):
    vllm_args = {
        "tensor_parallel_size": 8,
        "enable_chunked_prefill": True,
        "enable-auto-tool-choice": True,
        "tool-call-parser": "pythonic",
    }

    def _create_engine(self):
        return VLLMOpenAIEngine(
            model_id=self.model_name,
            vllm_port=self.vllm_process.port,
            temperature=self.temperature,
            use_managed_server=False,
        )


class KaniQwen3VLLMHandler(KaniVLLMHandler):
    vllm_args = {
        "tensor_parallel_size": 8,
        "enable_chunked_prefill": True,
    }

    def _create_engine(self):
        engine = VLLMServerEngine(
            model_id=self.model_name,
            vllm_port=self.vllm_process.port,
            temperature=self.temperature,
            use_managed_server=False,
        )
        engine.model = self.model_name
        return Qwen3Parser(engine)


class KaniGPTOSSVLLMHandler(KaniVLLMHandler):
    vllm_args = {
        "tensor_parallel_size": 8,
        "enable_chunked_prefill": True,
    }

    def _create_engine(self):
        engine = VLLMServerEngine(
            model_id=self.model_name,
            vllm_port=self.vllm_process.port,
            temperature=self.temperature,
            use_managed_server=False,
        )
        engine.model = self.model_name
        return GPTOSSParser(engine)


# no retry
class KaniLlama31VLLMNoRetryHandler(KaniLlama31VLLMHandler, KaniNoRetryHandler):
    pass


class KaniLlama32VLLMNoRetryHandler(KaniLlama32VLLMHandler, KaniNoRetryHandler):
    pass


class KaniQwen3VLLMNoRetryHandler(KaniQwen3VLLMHandler, KaniNoRetryHandler):
    pass


class KaniGPTOSSVLLMNoRetryHandler(KaniGPTOSSVLLMHandler, KaniNoRetryHandler):
    pass
