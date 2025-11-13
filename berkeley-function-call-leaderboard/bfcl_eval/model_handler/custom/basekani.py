from kani import Kani
from kani.engines.base import BaseCompletion


class TokenCountingKani(Kani):
    def add_completion_to_history(self, completion: BaseCompletion):
        completion.message.extra["prompt_tokens"] = completion.prompt_tokens
        completion.message.extra["completion_tokens"] = completion.completion_tokens
        return super().add_completion_to_history(completion)
