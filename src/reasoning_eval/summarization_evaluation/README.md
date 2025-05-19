


Note: The module paths are all messsed up. I will fix them at some point. Now, this codebase can only serve as a reference.

- `./openrouter/batch.py`: Request APIs to generate predictions for SummEval inputs and save results in jsonl format.
- `./eval_kendall.py`: It first parses the outputs from LLM APIs, then compute kendall's tau against expert annotations labels in SummEval dataset.
- `build_summeval.py`: SummEval official repo only provides labels not inputs. This code merges label data to input text.
