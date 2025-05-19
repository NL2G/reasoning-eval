import json
import os
import re
import sys
from collections import defaultdict
import time
from turtle import st
import numpy as np
from openai import OpenAI
import pandas as pd
from termcolor import colored
import tqdm
from rich import print
from rich.progress import track
from transformers import AutoTokenizer

import os

#from fastllm.fastllm.cache import DiskCache
from fastllm.cache import DiskCache
#from fastllm.fastllm.core import RequestBatch, RequestManager
from fastllm.core import RequestBatch, RequestManager
#from fastllm.fastllm.providers.openai import OpenAIProvider
from fastllm.providers.openai import OpenAIProvider

from gemba_utils.gemba_utils import TEMPLATE_GEMBA_MQM, apply_template, parse_mqm_answer

class DeepSeekR1Score:
    def __init__(self, models = None, arguments: dict = {}):
        #if 'o3-mini' in model:
        #    self.model = "-".join(model.split('-')[:-1])
        #    self.reasoning_effort = model.split('-')[-1]
        #else:
        #    self.model = model
        #    self.reasoning_effort = None
        #
        self.manager = RequestManager(provider=OpenAIProvider(
                api_base=os.environ.get("API_BASE"),
                api_key=os.environ.get("API_KEY")),
            caching_provider=DiskCache(
                directory="./cache", 
                expire=None, 
                size_limit=int(10e10), 
                cull_limit=0, 
                eviction_policy='none'
            ), concurrency=1000, show_progress=True, timeout=180)
        
        if models:
            self.models = models
        else:
            self.models = ["deepseek/deepseek-r1"] # nebius has a different model name

        # Mapping from API model names to HuggingFace model names
        self.model_to_hf_mapping = {
            "deepseek/deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "deepseek/deepseek-r1": "deepseek-ai/DeepSeek-R1",
            "deepseek/deepseek-r1-distill-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        }
        
        # Set default arguments if not provided
        self.arguments = arguments.copy()
        
        # Ensure include_reasoning is set if not already provided
        if 'include_reasoning' not in self.arguments:
            self.arguments['include_reasoning'] = True

    
    def __call__(
        self, 
        src: list[str],
        ref: list[str],
        hyps: dict[str, list[list[str]]],
        outfile = None, 
        source_lang = None,
        target_lang = None,
    ) -> dict[str, list[float]]:
        """
        Generate metric scores.

        Args:
        level: Level for which to produce scores, 'sys' or 'seg'.
        lp: Language pair, e.g. 'en-de'.
        domains: Map from domain name to [[beg, end+1], ...] segment position lists.
        docs: Map from doc name to [beg, end+1] segment positions.
        src: List of source segments.
        ref: List of reference segments.
        hyps: Map from MT system name to output segments for that system.

        Returns:
        Map from system name to scores, a list of segment-level scores if level is
        'seg', or a list containing a single score if level is 'sys'.
        """
        
        if not source_lang or not target_lang:
            raise ValueError("source_lang and target_lang must be provided.")
        
        src_segs = []
        hyp_segs = []
        hyp_ids = []
        for k, v in hyps.items():
            src_segs.extend(src)
            hyp_segs.extend(v)
            hyp_ids.extend([k] * len(v))
            
        df = pd.DataFrame({'source_seg': src_segs, 'target_seg': hyp_segs, 'system': hyp_ids})
        df['source_lang'] = source_lang
        df['target_lang'] = target_lang

        #cache = dc.Cache(f'cache/deepseek_mqm', expire=None, size_limit=int(10e10), cull_limit=0, eviction_policy='none')

        df["prompt"] = df.apply(lambda x: apply_template(TEMPLATE_GEMBA_MQM, x), axis=1)
        parse_answer = lambda x: parse_mqm_answer(x, count_limit=True)
        answers = self.bulk_request(df, parse_answer, max_tokens=None) # 500
        
            
        # Return answer by system as a dict of lists
        scores_seg = {}
        full_answers = {}
        full_response = {}
        for i, sys in enumerate(hyp_ids):
            if not sys in scores_seg:
                scores_seg[sys] = []
                full_answers[sys] = []
            #print(answers[i][0]["answer"])
            #raise ValueError
            if answers[i][0]["answer"] is None:
                scores_seg[sys].append(-25)
            else:
                scores_seg[sys].append(answers[i][0]["answer"][-1])
            full_answers[sys].append(answers[i])
        for k, v in scores_seg.items():
            assert len(v) == len(src), v
            assert all([-25<=s<=0 for s in v])
        scores_sys = {m: [np.mean(v)] for m, v in scores_seg.items()}
        
        if outfile:
            scores = {'seg': scores_seg, 'sys': scores_sys, 'full_answers': full_answers}
            with open(outfile, 'w', encoding='utf-8') as f:
                json.dump(scores, f)
        
            r = defaultdict(list)
            for system in scores_seg.keys():
                for idx, (score, answer) in enumerate(zip(scores_seg[system], full_answers[system])):
                    r['system'].append(system)
                    r['src_segment'].append(src[idx])
                    r['hyp_segment'].append(hyps[system][idx])
                    r['llm_mqm_score'].append(score)
                    
                    # Extract and save reasoning tokens
                    reasoning_tokens = answer[0].get('reasoning_tokens', None)
                    r['reasoning_tokens'].append(reasoning_tokens)
                    
                    # Extract and save reasoning content
                    reasoning_content = answer[0].get('reasoning_content', None)
                    r['reasoning_content'].append(reasoning_content)
                    
                    # Debug log to verify reasoning content is present
                    if reasoning_content:
                        print(f"Found reasoning content for system {system}, idx {idx}: {reasoning_content[:50]}...")
                    
                    if answer[0]['answer'] is None:
                        r['response'].append(None)
                        r['errors'].append(None)
                    else:
                        r['response'].append(answer[0]['answer'][0])
                        r['errors'].append(answer[0]['answer'][1])
            
            r = pd.DataFrame(r)
            # Ensure columns are in the right order to make reasoning_content more visible
            column_order = ['system', 'src_segment', 'hyp_segment', 'llm_mqm_score', 
                            'reasoning_tokens', 'reasoning_content', 'response', 'errors']
            column_order = [col for col in column_order if col in r.columns]
            r = r[column_order]
            r.to_csv(outfile.replace('json', 'csv'), index=False)
        return scores_seg, scores_sys, full_answers
    
    
    def request(self, prompts, parse_response, temperature=0, answer_id=-1, cache=None, max_tokens=None):
        ids = []
        with RequestBatch() as batch:
            for i in range(len(prompts)):
                # Prepare API call parameters
                api_params = {
                    "model": self.models[0],
                    "messages": prompts[i],
                    "max_completion_tokens": max_tokens,
                    "temperature": temperature/10,
                    "top_p": 1,
                    "n": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stop": None,
                }
                
                # Add return_reasoning_content parameter if include_reasoning is True
                if 'include_reasoning' in self.arguments and self.arguments['include_reasoning']:
                    api_params["return_reasoning_content"] = True
                
                # Merge with any additional arguments
                api_params.update(self.arguments)
                
                ids.append(batch.chat.completions.create(**api_params))
                
        answer_list = self.request_api(batch, temperature=temperature)
        

        outputs = []
        while len(outputs) == 0 or "INVALID" in outputs:
            if "INVALID" in outputs: # implementing the retry without recursion to make it somewhat parallel
                temperature += 1
                invalid_idx = [i for i, o in enumerate(outputs) if o == "INVALID"]
                invalid_prompts = [prompts[i] for i in invalid_idx]
                with RequestBatch() as invalid_batch:
                    for i in range(len(invalid_prompts)):
                        # Prepare API call parameters for retry
                        retry_params = {
                            "model": self.models[0],
                            "messages": invalid_prompts[i],
                            "max_completion_tokens": max_tokens,
                            "temperature": temperature/10,
                            "top_p": 1,
                            "n": 1,
                            "frequency_penalty": 0,
                            "presence_penalty": 0,
                            "stop": None,
                        }
                        
                        # Add return_reasoning_content if requested in arguments
                        if 'include_reasoning' in self.arguments and self.arguments['include_reasoning']:
                            retry_params["return_reasoning_content"] = True
                        
                        # Merge with any additional arguments
                        retry_params.update(self.arguments)
                        
                        ids.append(invalid_batch.chat.completions.create(**retry_params))
                
                answer_list_invalid = self.request_api(invalid_batch)
                answer_list = [answer_list[i] if o != "INVALID" else answer_list_invalid.pop(0) for i, o in enumerate(outputs)]
                outputs = []
                
            for answers, prompt in zip(answer_list, prompts):
                if len(answers) == 0: # if temp > 10, an empty list is returned
                    outputs.append([{
                            "temperature": temperature,
                            "answer_id": answer_id,
                            "answer": None,
                            "prompt": prompt,
                            "finish_reason": None,
                            "model": self.models[0],
                            "reasoning_tokens": None,
                            "reasoning_content": None,
                            }])
                    
                else:
                    parsed_answers = []
                    for full_answer in answers:
                        finish_reason = full_answer["finish_reason"]
                        reasoning_tokens = full_answer.get("reasoning_tokens", None)
                        reasoning_content = full_answer.get("reasoning_content", None)
                        full_answer = full_answer["answer"]

                        answer_id += 1
                        answer = parse_response(full_answer)
                        if temperature > 0:
                            print(f"Answer (t={temperature}): " + colored(answer, "yellow") + " (" + colored(full_answer, "blue") + ")", file=sys.stderr)
                        if answer is None:
                            continue
                        parsed_answers.append(
                            {
                                "temperature": temperature,
                                "answer_id": answer_id,
                                "answer": answer,
                                "prompt": prompt,
                                "finish_reason": finish_reason,
                                "model": self.models[0],
                                "reasoning_tokens": reasoning_tokens,
                                "reasoning_content": reasoning_content,
                            }
                        )

                    # there was no valid answer, increase temperature and try again
                    if len(parsed_answers) == 0:
                        outputs.append("INVALID")
                    else:
                        outputs.append(parsed_answers)
        
        return outputs

    
    def request_api(self, batch, temperature):
        if temperature > 10:
            return []
        
        try:
            i_list = [i for i in self.manager.process_batch(batch)]
            response_list = [i.response for i in i_list]
        except Exception as e:
            print(f"Error processing batch: {e.errors()}")
            raise e
        
        #print("i_list", response_list)

        # Pre-load tokenizer once if we're using deepseek models
        tokenizer = None
        if any("deepseek" in model for model in self.models):
            try:
                # Get the corresponding HuggingFace model name
                hf_model_name = None
                for api_model, hf_model in self.model_to_hf_mapping.items():
                    if api_model in self.models:
                        hf_model_name = hf_model
                        break
                
                # If we have a direct mapping, load the tokenizer
                if hf_model_name:
                    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            except Exception as e:
                print(f"Error loading tokenizer: {e}", file=sys.stderr)

        answer_list = []
        for response in track(response_list, description="Processing responses"):
            # Log response structure to help debug how to access reasoning_content
            if len(answer_list) == 0:  # Only log for the first response to avoid spam
                print("Response structure:")
                response_attrs = dir(response)
                print(f"Response attributes: {[attr for attr in response_attrs if not attr.startswith('_')]}")
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    choice_attrs = dir(response.choices[0])
                    print(f"Choice attributes: {[attr for attr in choice_attrs if not attr.startswith('_')]}")
                    if hasattr(response.choices[0], 'message'):
                        message_attrs = dir(response.choices[0].message)
                        print(f"Message attributes: {[attr for attr in message_attrs if not attr.startswith('_')]}")
            
            has_reasoning_tokens = hasattr(response.usage, "completion_tokens_details") and hasattr(response.usage.completion_tokens_details, "reasoning_tokens")
            if has_reasoning_tokens:
                reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
            else:
                # For deepseek models, calculate reasoning tokens by subtracting content tokens from completion tokens
                if any("deepseek" in model for model in self.models) and tokenizer is not None:
                    try:
                        # Get the answers for token counting
                        answers = []
                        for choice in response.choices:
                            if hasattr(choice, "message"):
                                if choice.message.content is None:
                                    answers.append("")
                                else:
                                    answers.append(choice.message.content.strip())
                            else:
                                answers.append(choice.text.strip())
                        
                        # Calculate tokens for all answers
                        content_tokens = 0
                        for answer in answers:
                            if answer:
                                content_tokens += len(tokenizer.encode(answer))
                        
                        # Subtract content tokens from completion tokens to get reasoning tokens
                        reasoning_tokens = response.usage.completion_tokens - content_tokens
                    except Exception as e:
                        print(f"Error calculating reasoning tokens: {e}", file=sys.stderr)
                        reasoning_tokens = response.usage.completion_tokens
                else:
                    reasoning_tokens = response.usage.completion_tokens
            
            answers = []
            for choice in response.choices:
                # Extract reasoning content at various possible locations in the response structure
                reasoning_content = None
                
                # Try to get reasoning_content from choice directly
                if hasattr(choice, "reasoning_content"):
                    reasoning_content = choice.reasoning_content
                    print("Found reasoning_content in choice:", reasoning_content[:50] if reasoning_content else None)
                # Try to get reasoning_content from message if present
                elif hasattr(choice, "message") and hasattr(choice.message, "reasoning_content"):
                    reasoning_content = choice.message.reasoning_content
                    print("Found reasoning_content in choice.message:", reasoning_content[:50] if reasoning_content else None)
                # Try to get from response object directly
                elif hasattr(response, "reasoning_content"):
                    reasoning_content = response.reasoning_content
                    print("Found reasoning_content in response:", reasoning_content[:50] if reasoning_content else None)
                # Check for a potential nested structure
                elif hasattr(response, "extra") and hasattr(response.extra, "reasoning_content"):
                    reasoning_content = response.extra.reasoning_content
                    print("Found reasoning_content in response.extra:", reasoning_content[:50] if reasoning_content else None)
                # Check if it might be in usage details
                elif hasattr(response, "usage") and hasattr(response.usage, "reasoning_content"):
                    reasoning_content = response.usage.reasoning_content
                    print("Found reasoning_content in response.usage:", reasoning_content[:50] if reasoning_content else None)
                elif hasattr(response, "usage") and hasattr(response.usage, "completion_tokens_details") and hasattr(response.usage.completion_tokens_details, "reasoning_content"):
                    reasoning_content = response.usage.completion_tokens_details.reasoning_content
                    print("Found reasoning_content in response.usage.completion_tokens_details:", reasoning_content[:50] if reasoning_content else None)
                
                # If still not found, try to parse it from any content field that might contain it
                if reasoning_content is None and hasattr(choice, "message") and hasattr(choice.message, "content") and choice.message.content:
                    content = choice.message.content
                    
                    # Try to extract reasoning content from various common formats
                    # Format 1: <reasoning>...</reasoning> tags
                    if "<reasoning>" in content and "</reasoning>" in content:
                        reasoning_start = content.find("<reasoning>") + len("<reasoning>")
                        reasoning_end = content.find("</reasoning>")
                        if reasoning_start < reasoning_end:
                            reasoning_content = content[reasoning_start:reasoning_end].strip()
                            print("Extracted reasoning_content from message content using <reasoning> tags")
                    # Format 2: ```reasoning\n...\n``` markdown blocks
                    elif "```reasoning" in content and "```" in content[content.find("```reasoning")+12:]:
                        reasoning_start = content.find("```reasoning") + len("```reasoning")
                        reasoning_end = content.find("```", reasoning_start)
                        if reasoning_start < reasoning_end:
                            reasoning_content = content[reasoning_start:reasoning_end].strip()
                            print("Extracted reasoning_content from markdown code block")
                    # Format 3: Check for JSON {"reasoning": "..."} pattern
                    elif '"reasoning":' in content.lower() or "'reasoning':" in content.lower():
                        try:
                            import json
                            # Try to extract JSON-like content
                            json_start = max(content.find('{'), 0)
                            json_end = content.rfind('}') + 1
                            if json_start < json_end:
                                json_content = content[json_start:json_end]
                                data = json.loads(json_content)
                                if "reasoning" in data:
                                    reasoning_content = data["reasoning"]
                                    print("Extracted reasoning_content from JSON structure")
                        except:
                            pass
                    # Format 4: If model is deepseek and no format detected, use the entire content as reasoning
                    # when include_reasoning is False (since this means it's outputting reasoning in content)
                    elif "deepseek" in self.models[0] and not self.arguments.get('include_reasoning', True):
                        # For DeepSeek when include_reasoning=False, the entire content might be the reasoning
                        reasoning_content = content
                        print("Using entire message content as reasoning for DeepSeek with include_reasoning=False")
                
                if hasattr(choice, "message"):
                    if choice.message.content is None:
                        answer = "INVALID"
                    else:
                        answer = choice.message.content.strip()
                else:
                    answer = choice.text.strip()

                answers.append({
                    "answer": answer,
                    "finish_reason": choice.finish_reason,
                    "reasoning_tokens": reasoning_tokens,
                    "reasoning_content": reasoning_content,
                })

            if len(answers) > 1:
                # remove duplicate answers
                answers = [dict(t) for t in {tuple(d.items()) for d in answers}]

            answer_list.append(answers)
        return answer_list
    
    def bulk_request(self, df, parse_mqm_answer, max_tokens=None):
        return self.request(df["prompt"].tolist(), parse_mqm_answer, max_tokens=max_tokens)
    
    #def call_api(self, prompt, model, temperature, max_tokens):
    #    parameters = {
    #        "temperature": temperature/10,
    #        "top_p": 1,
    #        "n": 1,
    #        "frequency_penalty": 0,
    #        "presence_penalty": 0,
    #        "stop": None,
    #        "model": model,
    #        "reasoning_effort": self.reasoning_effort,
    #    }

    #    if self.model in ["deepseek/deepseek-r1"]:
    #        if max_tokens is not None:
    #            parameters["max_tokens"] = max_tokens

    #    assert all(isinstance(p, dict) for p in prompt), "Prompts must be a list of dictionaries."
    #    assert all("role" in p and "content" in p for p in prompt), "Prompts must be a list of dictionaries with role and content."

    #    parameters["messages"] = prompt


    #    return self.client.chat.completions.create(**parameters)

if __name__ == '__main__':
    dsr = DeepSeekR1Score()
    
    src = ['Dies ist ein Test.', 'Dies ist ein Test']
    hyps = {'hyp1': ['This i a test.', 'This is a test.'], 
            'hyp2': ['This aint a test.', 'This is a tost.']}
    
    seg, sys, full = dsr(src, "", hyps, 'dsr_scores_test.json', source_lang="German", target_lang="English")
    
    for k, v in seg.items():
        for i, s in enumerate(v):
            print(f"{k} seg {i}: {s}, src: {src[i]}, hyp: {hyps[k][i]}")