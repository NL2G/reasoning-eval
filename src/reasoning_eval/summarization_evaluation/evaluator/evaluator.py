import json
import os

from reasoning_eval.summarization_evaluation.evaluator.prompter import GEvalPrompter
import torch
from transformers import BitsAndBytesConfig, pipeline


class Evaluator:
    def __init__(self, model_name: str, prompter: GEvalPrompter):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model_kwargs = {
            "quantization_config": bnb_config,
        }

        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs=model_kwargs,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        )
        self.prompter = prompter

    def score(self, src: str, hypo: str) -> float | None:
        prompt = self.prompter.generate_prompt(src, hypo)
        out = self.pipe(
            [
                {
                    "role": "system",
                    "content": "You are a helpful and honest assistant. Please, respond concisely and truthfully.",
                },
                {"role": "user", "content": prompt},
            ],
            do_sample=False,
            max_new_tokens=5120,
            return_full_text=False,
            top_p=None,
            temperature=0.0,
        )
        try:
            out_str: str = out[0]["generated_text"]
            out_str = out_str.replace("\n", "")
            out_dict = json.loads(out_str)
            score = float(out_dict["score"])
        except:
            score = None

        return score

    def run(self, src: str, hypo: str) -> str:
        prompt = self.prompter.generate_prompt(src, hypo)
        out = self.pipe(
            [
                {
                    "role": "system",
                    "content": "You are a helpful and honest assistant. Please, respond concisely and truthfully.",
                },
                {"role": "user", "content": prompt},
            ],
            do_sample=False,
            max_new_tokens=5120,
            return_full_text=False,
            top_p=None,
            temperature=0.0,
        )

        return out[0]["generated_text"]
