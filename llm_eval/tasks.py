import os
from typing import Iterator, Dict, List, Type
from dataclasses import dataclass, field
import json
import tempfile
import pathlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from jinja2 import Template
import numpy
import datasets
import numpy as np
import tabulate

from .execution import check_correctness


@dataclass
class Prompt:
    text: str
    labels: Dict[str, str] = field(default_factory=dict)
    forced_generation: str | List[int] = field(default_factory=list)


def _get_template_dir(task_name: str) -> str:
    curr_dir = os.path.dirname(__file__)
    return os.path.join(curr_dir, "templates", task_name)


def _load_template(task_name: str, prompt_style: str | None = None):
    if not prompt_style:
        return Template("{{ text }}")

    template_file = os.path.join(_get_template_dir(task_name), prompt_style + ".jinja")
    if not os.path.isfile(template_file):
        raise ValueError(
            f"Unsupported prompt style {prompt_style} for dataset {task_name}"
        )
    with open(template_file) as f:
        template_str = f.read()
        return Template(template_str, keep_trailing_newline=True)


def _perplexity(resp):
    choice = resp["choices"][0]
    logprobs = choice["logprobs"]["token_logprobs"]
    return numpy.exp(-numpy.mean(logprobs))


def _get_dataset_dir(ds_name: str, ds_cache_dir: str) -> str | None:
    code_dir = os.path.join(os.path.dirname(__file__), f"datasets/{ds_name}")
    if os.path.exists(code_dir):
        return code_dir

    cache_dir = os.path.join(ds_cache_dir, ds_name)
    if os.path.exists(cache_dir):
        return cache_dir

    raise ValueError(f"No dataset folder found {code_dir} or {cache_dir}")


class Task:
    def get_prompts(self) -> Iterator[Prompt]:
        raise NotImplementedError()

    @property
    def max_tokens(self) -> int:
        return None

    @property
    def logprobs(self) -> int:
        return 0

    @property
    def echo(self) -> bool:
        return False

    @classmethod
    def analyze(cls, responses: Iterator, verbose: bool = False):
        pass


def _analyze_multichoice(responses: Iterator, verbose=False):
    table = [["Pos", "ID", "Perplexity", "Accuracy"]]

    total_perplex = 0
    total_acc = 0
    num_responses = 0
    for i, response in enumerate(responses):
        num_responses += 1
        if "choices" not in response:
            continue

        perplex = _perplexity(response)
        total_perplex += perplex

        choice = response["choices"][0]
        answer = "".join(choice["text"]).strip()
        splits = answer.split("\n")
        if len(splits) > 1:
            answer = splits[0]
        acc = response["labels"]["correct_answer"] == answer

        total_acc += acc

        table.append(
            [
                i,
                response["id"],
                perplex,
                acc,
            ]
        )

    if verbose:
        print(tabulate.tabulate(table))
    print(
        tabulate.tabulate(
            [
                ["Perplexity", total_perplex / num_responses],
                ["Accuracy", total_acc / num_responses],
            ],
            tablefmt="plain",
        )
    )


def _analyze_generic(responses: List, verbose: bool):
    table = [["Pos", "ID", "Perplexity", "Prompt Tokens", "Gen Tokens"]]

    total_perplex = 0
    num_responses = 0
    prompt_lens = []
    gen_lens = []
    for i, response in enumerate(responses):
        if "choices" not in response:
            continue
        perplex = _perplexity(response)
        total_perplex += perplex
        prompt_lens.append(response["usage"]["prompt_tokens"])
        gen_lens.append(response["usage"]["completion_tokens"])

        num_responses += 1

        table.append(
            [
                i,
                response["id"],
                perplex,
                prompt_lens[-1],
                gen_lens[-1],
            ]
        )

    if verbose:
        print(tabulate.tabulate(table))
    print(
        tabulate.tabulate(
            [
                [
                    "Perplexity",
                    total_perplex / num_responses,
                ],
                [
                    "Avg Prompt Tokens",
                    sum(prompt_lens) / len(prompt_lens),
                ],
                [
                    "Avg Completion Tokens",
                    sum(gen_lens) / len(gen_lens),
                ],
            ],
            tablefmt="plain",
        )
    )


def _pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k).
    See HumanEval paper: https://arxiv.org/pdf/2107.03374.pdf
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _analyze_code(
    ids: List[str],
    test_cases: List[str],
    logprobs: List[List[int]],
    num_workers=8,
    timeout=3.0,
    verbose=False,
):
    passed = [0]
    table = [["ID", "Passed"]]

    def _on_done(future, id, ppl):
        result = future.result()
        if result["passed"]:
            passed[0] += 1
        table.append([id, result["passed"], ppl])

    total_ppl = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for id, test_case, lp in zip(ids, test_cases, logprobs):
            ppl = numpy.exp(-numpy.mean(lp))
            total_ppl += ppl

            args = (test_case, timeout)
            future = executor.submit(check_correctness, *args)
            future.add_done_callback(partial(_on_done, id=id, ppl=ppl))

    if verbose:
        print(tabulate.tabulate(table))
    print(
        tabulate.tabulate(
            [
                ["Perplexity", total_ppl / len(ids)],
                ["Pass@1", _pass_at_k(len(ids), passed[0], 1)],
            ],
            tablefmt="plain",
        )
    )


class GenericTask(Task):
    def __init__(self, ds_name: str, ds_cache_dir: str, prompt_style: str = "base"):
        self.ds_name = ds_name
        self.ds_cache_dir = ds_cache_dir
        self.prompt_style = prompt_style

    def get_prompts(self) -> Iterator[Prompt]:
        subtask = os.path.join("generic", self.ds_name)
        if os.path.exists(_get_template_dir(subtask)):
            template = _load_template(subtask, self.prompt_style)
        else:
            template = _load_template("generic", self.prompt_style)

        # Format prompts.
        ds_dir = _get_dataset_dir(self.ds_name, self.ds_cache_dir)
        with open(os.path.join(ds_dir, "test.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                prompt = Prompt(template.render(**data))
                yield prompt

    @classmethod
    def analyze(cls, responses: Iterator, verbose=False):
        _analyze_generic(responses, verbose)

    @property
    def logprobs(self) -> int:
        return 4

    @property
    def echo(self) -> bool:
        return True


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


class MmluTask(Task):
    CATEGORIES = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]
    CHOICES = ["A", "B", "C", "D"]

    def __init__(self, ds_cache_dir: str, prompt_style: str = "base"):
        self.prompt_style = prompt_style
        self.ds_cache_dir = ds_cache_dir

    def _prepare(self) -> None:
        ds_folder = os.path.join(self.ds_cache_dir, "mmlu")
        if os.path.exists(ds_folder):
            return

        os.makedirs(self.ds_cache_dir, exist_ok=True)
        ds_tmp_folder = tempfile.mkdtemp(prefix="mmlu", dir=self.ds_cache_dir)

        def _process_split(split):
            for config in self.CATEGORIES:
                split_dir = os.path.join(ds_tmp_folder, split)
                os.makedirs(split_dir, exist_ok=True)
                with open(os.path.join(split_dir, config + ".jsonl"), "w+") as f:
                    ds = datasets.load_dataset(
                        "hails/mmlu_no_train", config, split=split
                    )
                    for v in ds:
                        f.write(json.dumps(v))
                        f.write("\n")

        _process_split("dev")
        _process_split("test")
        os.rename(ds_tmp_folder, ds_folder)

    def get_prompts(self) -> Iterator[Prompt]:
        self._prepare()
        ds_folder = _get_dataset_dir("mmlu", self.ds_cache_dir)
        template = _load_template("mmlu", self.prompt_style)
        for cat in sorted(self.CATEGORIES):
            shots = []
            data = {"shots": shots}

            with open(os.path.join(ds_folder, "dev", cat + ".jsonl")) as f:
                for line in f:
                    shots.append(json.loads(line))

            with open(os.path.join(ds_folder, "test", cat + ".jsonl")) as f:
                for line in f:
                    task = json.loads(line)
                    correct_answer = self.CHOICES[int(task["answer"])]
                    del task["answer"]
                    data["task"] = task
                    prompt = template.render(data)
                    yield Prompt(prompt, labels={"correct_answer": correct_answer})

    @classmethod
    def analyze(cls, responses: Iterator, verbose=False):
        _analyze_multichoice(responses, verbose=verbose)

    @property
    def max_tokens(self):
        return 1


class HumanevalTask(Task):
    def __init__(self, ds_cache_dir: str):
        self.ds_cache_dir = ds_cache_dir

    def _prepare(self) -> None:
        ds_folder = os.path.join(self.ds_cache_dir, "humaneval")
        if os.path.exists(ds_folder):
            return

        os.makedirs(self.ds_cache_dir, exist_ok=True)
        ds_tmp_folder = tempfile.mkdtemp(prefix="humaneval", dir=self.ds_cache_dir)
        with open(os.path.join(ds_tmp_folder, "test.jsonl"), "w+") as f:
            ds = datasets.load_dataset(
                "openai_humaneval", split="test"
            )
            for v in ds:
                f.write(json.dumps(v))
                f.write("\n")

        os.rename(ds_tmp_folder, ds_folder)

    def get_prompts(self) -> Iterator[Prompt]:
        self._prepare()
        ds_folder = _get_dataset_dir("humaneval", self.ds_cache_dir)    
        with open(os.path.join(ds_folder, "test.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                yield Prompt(
                    data["prompt"],
                    labels={
                        "task_id": data["task_id"],
                        "prompt": data["prompt"],
                        "test": data["test"],
                        "entry_point": data["entry_point"],
                    },
                )

    @classmethod
    def analyze(cls, responses: Iterator, verbose=False):
        ids = []
        test_cases = []
        log_probs = []
        for response in responses:
            ids.append(response["labels"]["task_id"])
            log_probs.append(response["choices"][0]["logprobs"]["token_logprobs"])
            test_cases.append(
                response["labels"]["prompt"]
                + "\n"
                + response["choices"][0]["text"]
                + "\n"
                + response["labels"]["test"]
                + "\n\ncheck("
                + response["labels"]["entry_point"]
                + ")\n"
            )
        _analyze_code(ids, test_cases, log_probs, verbose=verbose)

    @property
    def max_tokens(self):
        return 1024

    @property
    def stop_words(self) -> List[str]:
        return ["\nclass", "\ndef"]


def get_task(
    name: str,
    ds_cache_dir: str | None = None,
    prompt_style: str = "base",
    task_args: Dict = {},
) -> Task:
    if ds_cache_dir is None:
        ds_cache_dir = os.path.join(pathlib.Path.home(), ".cache/fireworks/datasets")

    cls = get_task_class(name)
    if cls == MmluTask:
        return MmluTask(ds_cache_dir, prompt_style)
    elif cls == HumanevalTask:
        return HumanevalTask(ds_cache_dir)
    elif cls == GenericTask:
        return GenericTask(task_args["generic_ds_name"], ds_cache_dir, prompt_style)
    elif cls == ReplayTask:
        return ReplayTask(task_args["replay_file"])
    else:
        raise ValueError(f"Unsupported task {name}")


def get_task_class(name: str) -> Type[Task]:
    if name == "mmlu":
        return MmluTask
    elif name == "humaneval":
        return HumanevalTask
    elif name == "generic":
        return GenericTask
    elif name == "replay":
        return ReplayTask
    else:
        raise ValueError(f"Unsupported task {name}")
