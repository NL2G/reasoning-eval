from argparse import ArgumentParser
import sys
from rich import print

import sienna
from fastllm import DiskCache, OpenAIProvider, RequestBatch, RequestManager

from reasoning_eval.summarization_evaluation.evaluator.prompter import (
    COHERENCE_TEMPLATE,
    CONSISTENCY_TEMPLATE,
    FLUENCY_TEMPLATE,
    GEvalPrompter,
    RELEVANCE_TEMPLATE,
    ORIGINAL_LAST_COHERENCE,
    ORIGINAL_LAST_FLUENCY,
    ORIGINAL_LAST_RELEVANCE,
    ORIGINAL_LAST_CONSISTENCY
)
from reasoning_eval.summarization_evaluation.utils import DATA_BASE_DIR

MODELNAME2ENDPOINT = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "http://qwen32.tensor.rocks/v1",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "http://llama8.tensor.rocks/v1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "http://qwen1-5.tensor.rocks/v1",
    "deepseek/deepseek-r1": "https://openrouter.ai/api/v1",
    "openai/gpt-4o-mini": "https://openrouter.ai/api/v1",
    "meta-llama/llama-3.1-8b-instruct": "https://openrouter.ai/api/v1",
    "Qwen/Qwen2.5-32B-Instruct": "https://api.studio.nebius.com/v1/",
}


def run():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )
    parser.add_argument("--api-base", type=str)
    parser.add_argument("--metric-type", type=str)
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--do-json", action="store_true")
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--reasoning-effort", type=str, default="none", choices=["none", "low", "medium", "high"])
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    model_name: str = args.model_name

    match args.metric_type:
        case "consistency":
            template = CONSISTENCY_TEMPLATE
            original_last = ORIGINAL_LAST_CONSISTENCY
        case "coherence":
            template = COHERENCE_TEMPLATE
            original_last = ORIGINAL_LAST_COHERENCE
        case "fluency":
            template = FLUENCY_TEMPLATE
            original_last = ORIGINAL_LAST_FLUENCY
        case "relevance":
            template = RELEVANCE_TEMPLATE
            original_last = ORIGINAL_LAST_RELEVANCE
        case _:
            raise ValueError()

    prompter = GEvalPrompter(template, do_json=args.do_json, original_last=original_last)
    data = sienna.load(DATA_BASE_DIR / "model_annotations.aligned.paired.jsonl")

    manager = RequestManager(
        provider=OpenAIProvider(
            api_base=args.api_base,
            api_key=args.api_key,
        ),
        caching_provider=DiskCache(directory="./.cache"),
        concurrency=args.concurrency,
        timeout=180,
        show_progress=True,
        return_dummy_on_error=True if "openai" in model_name else False
    )

    ids = []
    with RequestBatch() as batch:
        for x in data:
            prompt = prompter.generate_prompt(x["src"], x["decoded"])
            ids.append(
                batch.chat.completions.create(
                    include_reasoning=True if "openai" not in model_name else None,
                    temperature=0.6,
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort=args.reasoning_effort if args.reasoning_effort != "none" else None,
                )
            )
    responses = manager.process_batch(batch)

    outputs = []
    for x, res in zip(data, responses):
        usage = dict(res.response.usage)
        for k, v in usage.items():
            if v is not None:
                if type(v) != int:
                    usage[k] = dict(v)
        if hasattr(res.response.choices[0].message, "reasoning_content"):
            outputs.append(
                {
                    "id": x["id"],
                    "reasoning": res.response.choices[0].message.reasoning_content,
                    "content": res.response.choices[0].message.content,
                    "usage": usage,
                }
            )
        elif hasattr(res.response.choices[0].message, "reasoning"):
            outputs.append(
                {
                    "id": x["id"],
                    "reasoning": res.response.choices[0].message.reasoning,
                    "content": res.response.choices[0].message.content,
                    "usage": usage,
                }
            )
        else:
            outputs.append(
                {
                    "id": x["id"],
                    "reasoning": None,
                    "content": res.response.choices[0].message.content,
                    "usage": usage,
                }
            )

    model_name = model_name.replace("/", "_")
    if args.reasoning_effort != "none":
        model_name += f".{args.reasoning_effort}"
    sienna.save(
        outputs,
        f"{args.output_dir}/{model_name}.{args.metric_type}{'.json-format' if args.do_json else ''}.jsonl",
    )


if __name__ == "__main__":
    run()
