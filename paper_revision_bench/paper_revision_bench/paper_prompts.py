"""
Paper-specific evaluation prompts from XtraGPT paper.
These prompts are designed for evaluating academic paper revisions.
"""

from typing import Dict

# Section-specific evaluation criteria from the XtraGPT paper
PAPER_SECTION_PROMPTS: Dict[str, str] = {
    "title": """Criteria 1. Consistency and Alignment of Title with Paper's Content Explanation 1. Evaluate the degree to which the title accurately captures its principal topics, arguments, or findings. Does the title reflect the scope and focus of the paper, and is it consistent with the main concepts and keywords presented in the abstract and introduction? Identify any discrepancies or misalignment between the title and the content.
Criteria 2. Conciseness and Clarity of Title Explanation 2. Evaluate the paper's title for redundancy. Are there repeated words or concepts that could be removed without changing the core meaning? Does the final title remain succinct, clear, and accurately convey the paper's main focus or contribution?""",

    "abstract": """Criteria 1. Clarity and Impact of the Conclusion Explanation 1. Evaluate the clarity and impact of the conclusion in the abstract. Does it clearly summarize the research steps, highlight key outcomes, and explain the significance of these outcomes for the field of computer science? Are the primary technical advancements and their contributions presented in a concise and unambiguous manner?
Criteria 2. Motivation and Purpose in the Abstract Explanation 2. Evaluate how well the abstract communicates the research's motivation. Does it clearly articulate the broader issue, concept, or problem in Computer Science that the work addresses? Does it explicitly state the specific research problem being solved and why it is important?
Criteria 3. Explanation of Existing Solutions and Research Gap Explanation 3. Assess how well the abstract explains the shortcomings of current solutions and highlights the corresponding research gap. Does it clearly articulate why existing methods are insufficient and how the proposed approach addresses these limitations? Is the explanation comprehensible to a wide audience, from domain experts to non-specialists?
Criteria 4. Clarity and Adequacy of Proposed Solutions Explanation 4. Assess how effectively the abstract communicates the proposed solutions. Does it clearly identify the research gap or problem being addressed, and explain how the proposed solution tackles this gap? Does it highlight the novelty or contribution of the solution, demonstrating its relevance or improvement over existing work? Rate the clarity, completeness, and significance of the explanation provided in the abstract.""",

    "introduction": """Criteria 1. Clarity and Depth of Problem Statement Explanation 1. Evaluate how clearly and thoroughly the introduction defines the research problem. Does it provide sufficient context for readers to understand the significance of the problem? Is the problem statement specific enough to guide the research direction?
Criteria 2. Motivation and Significance Explanation 2. Assess how well the introduction motivates the research. Does it explain why this problem matters? Does it connect the research to broader impacts in the field?
Criteria 3. Research Gap Identification Explanation 3. Evaluate how effectively the introduction identifies gaps in existing work. Does it clearly explain what is missing from current approaches and why a new solution is needed?
Criteria 4. Contribution Statement Explanation 4. Assess the clarity and completeness of the contribution statement. Are the main contributions clearly listed and explained? Do they align with the identified research gap?""",

    "background": """Criteria 1. Coverage and Relevance Explanation 1. Evaluate the comprehensiveness of the related work coverage. Does it include all relevant prior work? Is each cited work clearly relevant to the research problem?
Criteria 2. Organization and Structure Explanation 2. Assess how well the background section is organized. Are related works grouped logically? Is there a clear narrative flow?
Criteria 3. Critical Analysis Explanation 3. Evaluate the depth of analysis of prior work. Does it go beyond mere description to critically assess strengths and limitations of existing approaches?
Criteria 4. Positioning Explanation 4. Assess how well the paper positions itself relative to prior work. Is it clear how this work differs from and improves upon existing approaches?""",

    "evaluation": """Criteria 1. Experimental Design Explanation 1. Evaluate the rigor of the experimental design. Are the experiments well-designed to test the claims? Are baselines appropriate and sufficient?
Criteria 2. Clarity of Presentation Explanation 2. Assess how clearly the results are presented. Are tables and figures easy to understand? Is the narrative around results clear?
Criteria 3. Analysis Depth Explanation 3. Evaluate the depth of analysis. Does it go beyond reporting numbers to provide insights? Are failure cases and limitations discussed?
Criteria 4. Reproducibility Explanation 4. Assess whether sufficient details are provided for reproducibility. Are hyperparameters, datasets, and implementation details clearly specified?""",

    "conclusion": """Criteria 1. Summary Completeness Explanation 1. Evaluate how well the conclusion summarizes the main contributions and findings. Does it capture the key points without unnecessary repetition?
Criteria 2. Impact Discussion Explanation 2. Assess how well the conclusion discusses the broader impact and significance of the work.
Criteria 3. Limitations Acknowledgment Explanation 3. Evaluate whether limitations are honestly acknowledged and discussed.
Criteria 4. Future Work Explanation 4. Assess the quality of future work discussion. Are the suggested directions concrete and well-motivated?""",
}


# AlpacaEval-style prompt template for paper revision evaluation
ALPACA_EVAL_TEMPLATE = """<|im_start|>system
You are a highly efficient assistant, who evaluates and rank large language models (LLMs) based on the quality of their responses to given prompts. This process will create a leaderboard reflecting the most accurate and human-preferred answers.
<|im_end|>
<|im_start|>user
I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding responses. Your task is to assess these responses, ranking the models in order of preference from a human perspective. Once ranked, please output the results in a structured JSON format for the make_partial_leaderboard function.

## Prompt

{{
    "instruction": \"\"\"{instruction}\"\"\",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model": "m",
        "output": \"\"\"{output_1}\"\"\"
    }},
    {{
        "model": "M",
        "output": \"\"\"{output_2}\"\"\"
    }}
}}

## Task

Evaluate and rank the models based on the quality and relevance of their outputs. The ranking should be such that the model with the highest quality output is ranked first.
{section_criteria}
<|im_end|>"""


def get_paper_eval_prompt(section: str) -> str:
    """
    Get the AlpacaEval-style prompt template for a specific paper section.

    Args:
        section: Paper section (title, abstract, introduction, background, evaluation, conclusion)

    Returns:
        Prompt template string with placeholders for {instruction}, {output_1}, {output_2}
    """
    section = section.lower()
    if section not in PAPER_SECTION_PROMPTS:
        raise ValueError(f"Unknown section: {section}. Valid sections: {list(PAPER_SECTION_PROMPTS.keys())}")

    criteria = PAPER_SECTION_PROMPTS[section]
    return ALPACA_EVAL_TEMPLATE.replace("{section_criteria}", criteria)


def list_paper_sections() -> list:
    """Return list of supported paper sections."""
    return list(PAPER_SECTION_PROMPTS.keys())
