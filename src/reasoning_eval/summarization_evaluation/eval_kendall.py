import re
from argparse import ArgumentParser
from pathlib import Path

import sienna
from scipy.stats import kendalltau, spearmanr

from reasoning_eval.summarization_evaluation.utils import DATA_BASE_DIR


def aggreage_expert_annotations(data_point: dict) -> dict:
    aspects = ["coherence", "consistency", "fluency", "relevance"]
    aspect_score = {aspect: -1 for aspect in aspects}
    for aspect in aspects:
        scores = []
        for annotation in data_point["expert_annotations"]:
            scores.append(annotation[aspect])
        aspect_score[aspect] = sum(scores) / len(scores)
    return aspect_score


def aggreage_turker_annotations(data_point: dict) -> dict:
    aspects = ["coherence", "consistency", "fluency", "relevance"]
    aspect_score = {aspect: -1 for aspect in aspects}
    for aspect in aspects:
        scores = []
        for annotation in data_point["turker_annotations"]:
            scores.append(annotation[aspect])
        aspect_score[aspect] = sum(scores) / len(scores)
    return aspect_score


def parse_output(output_data: dict[str, str]) -> int | float | None:
    content = output_data["content"]
    if "Consistency" in content:
        content = content[content.find("Consistency") :]

    score_match = re.search(r"\d", content, re.IGNORECASE)
    score = int(score_match.group(0)) if score_match else None
    return score


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--metric-type", type=str)
    parser.add_argument("--cut-off", type=int, required=False, default=None)
    args = parser.parse_args()

    output_file = Path(args.output_file)
    data = sienna.load(DATA_BASE_DIR / "model_annotations.aligned.paired.jsonl")
    prediction = sienna.load(output_file)

    if args.cut_off:
        data = data[: args.cut_off]
        prediction = prediction[: args.cut_off]

    assert len(data) == len(prediction)

    expert_scores = []
    model_scores = []

    for d, p in zip(data, prediction):
        score = parse_output(p)
        if (score is None) or (score > 5) or (score < 1):
            continue
        expert = aggreage_expert_annotations(d)
        expert_scores.append(expert[args.metric_type])
        model_scores.append(score)

    print(kendalltau(expert_scores, model_scores))
    print(spearmanr(expert_scores, model_scores))
