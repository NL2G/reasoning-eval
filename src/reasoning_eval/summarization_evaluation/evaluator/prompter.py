from dataclasses import dataclass

ORIGINAL_LAST_CONSISTENCY = """Evaluation Form (scores ONLY):

- Consistency:
"""

ORIGINAL_LAST_COHERENCE = """Evaluation Form (scores ONLY):

- Coherence:
"""

ORIGINAL_LAST_FLUENCY = """Evaluation Form (scores ONLY):

- Fluency:
"""

ORIGINAL_LAST_RELEVANCE = """Evaluation Form (scores ONLY):

- Relevance:
"""

JSON_LAST = (
    "Generate ONLY the json as the output which contains one key which is `score`."
)

CONSISTENCY_TEMPLATE = """
You will be given a news article. You will then be given one summary written for this article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts. 

Evaluation Steps:

1. Read the news article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.

Source Text: 

{source_text}

Output: 

{hypothesis}

"""

COHERENCE_TEMPLATE = """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Evaluation Steps:

1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.


Source Text:

{source_text}

Summary:

{hypothesis}

"""

FLUENCY_TEMPLATE = """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

Evaluation Steps:

1. Read the summary carefully and check for any grammatical errors, punctuation mistakes, or awkward sentence structures.
2. Check if the summary is well-written and easy to read.
3. Assign a score for fluency on a scale of 1 to 3, where 1 is the lowest and 3 is the highest based on the Evaluation Criteria.

Summary:

{hypothesis}

"""

RELEVANCE_TEMPLATE = """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Evaluation Steps:

1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.

Source Text:

{source_text}

Summary:

{hypothesis}

"""

@dataclass(frozen=True)
class GEvalPrompter:
    template: str
    do_json: bool
    original_last: str

    def generate_prompt(self, doc: str, hypo: str) -> str:
        return self.template.format(
            source_text=doc,
            hypothesis=hypo,
        ) + (JSON_LAST if self.do_json else self.original_last)
