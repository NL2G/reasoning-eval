import sienna
from argparse import ArgumentParser
from tqdm import tqdm

from reasoning_eval.summarization_evaluation.evaluator.evaluator import Evaluator
from reasoning_eval.summarization_evaluation.evaluator.prompter import (
    COHERENCE_TEMPLATE,
    CONSISTENCY_TEMPLATE,
    GEvalPrompter,
)
from reasoning_eval.summarization_evaluation.utils import DATA_BASE_DIR

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument("--metric-type", type=str)
    args = parser.parse_args()

    model_name: str = args.model_name

    match args.metric_type:
        case "consistency":
            template = CONSISTENCY_TEMPLATE
        case "coherence":
            template = COHERENCE_TEMPLATE
        case _:
            raise ValueError()

    evaluator = Evaluator(model_name, GEvalPrompter(template))
    data = sienna.load(DATA_BASE_DIR / "model_annotations.aligned.paired.jsonl")

    outputs = []

    for x in tqdm(data):
        out = evaluator.run(x["src"], x["decoded"])
        outputs.append({"output": out})
        sienna.save(
            outputs, f"./{model_name.replace('/', '_')}.{args.metric_type}.jsonl"
        )
