import json
from typing import Iterator

from .common import Task, Prompt


class ReplayTask(Task):
    def __init__(self, replay_file: str):
        self.replay_file = replay_file

    def get_prompts(self) -> Iterator[Prompt]:
        with open(self.replay_file, "r") as f:
            for line in f:
                resp = json.loads(line)
                choice = resp["choices"][0]
                n_prompt = resp["usage"]["prompt_tokens"]
                tokens = choice["logprobs"]["tokens"]
                yield Prompt(
                    text="".join(tokens[:n_prompt]),
                    forced_generation="".join(tokens[n_prompt:]),
                )

    @property
    def logprobs(self) -> int:
        return 4

    @property
    def echo(self) -> bool:
        return True
