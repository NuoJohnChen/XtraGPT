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

    "introduction": """Criteria 1. Strength and Clarity of Motivation in the Introduction Explanation 1. Evaluate whether the motivation in the Introduction is specific and convincing. Does the paper avoid over-generalization and clearly articulate the significance of the issue? Are concrete examples, statistics, or contextual details used to establish why the problem matters?
Criteria 2. Review of Existing Approaches in Introduction Explanation 2. Assess the thoroughness and clarity of the literature review within the introduction. Does the paper cite and critique relevant prior works, highlighting both their methodologies and limitations? Does the introduction establish how the proposed work builds upon or differentiates itself from existing research, and is there sufficient context provided to demonstrate the significance of the current study? Are any quantitative or qualitative comparisons made where appropriate?
Criteria 3. Audience Alignment and Appropriateness Explanation 3. Evaluate whether the introduction is effectively tailored to its target audience. Is the complexity, depth, and choice of terminology suitable for the presumed background knowledge of the readership? Does the introduction provide sufficient context without oversimplifying or overwhelming the intended audience?
Criteria 4. Clarity and Visibility of Contributions Explanation 4. Assess the clarity and visibility of the paper's contributions. Are the core contributions explicitly stated in a dedicated paragraph or section toward the end of the introduction? Are they understandable to a broad scientific audience, presented succinctly, and positioned logically following the problem statement and background information?
Criteria 5. Clarity and Specificity of Problem Definition Explanation 5. Evaluate the paper's problem definition in terms of four key elements: current situation, ideal situation, the gap between them, and how the research aims to address this gap. Are these components clearly stated, distinct, and directly tied to the research objectives? Does the definition provide sufficient clarity and focus for the research?
Criteria 6. Integration of State-of-the-Art in Problem Framing Explanation 6. Evaluate how effectively the introduction incorporates the State-of-the-Art (SOTA) to frame the research problem. Does it include explicit references to key works, methodologies, or findings that highlight relevant gaps or limitations in the field? Is there a clear logical link between the SOTA discussion and the stated research objectives, demonstrating how the proposed work builds upon or extends existing research?""",

    "background": """Criteria 1. Contextual Relevance and Clarity of Background Explanation 1. Assess how effectively the background section establishes context for the research. Does it provide a clear overview of the broader field in computer science, then narrow down to the specific problem? Does the paper avoid making unwarranted assumptions about the reader's prior knowledge? Finally, does it clarify why addressing the problem is important to the field?
Criteria 2. Coverage of Key Preliminary Concepts Explanation 2. Evaluate the thoroughness and clarity of the paper's background or preliminary section. Does it introduce and define all the critical concepts, algorithms, or theorems necessary to understand the technical contributions? Are these concepts clearly explained, logically organized, and accessible to readers who are not experts in the field? Does the paper use consistent terminology and adequately explain symbols, abbreviations, or specialized terms before their first usage?
Criteria 3. Clarity and Consistency of Terminology Explanation 3. Assess the clarity and consistency of the key terms introduced in the background section. Are all critical terminologies defined at their first occurrence and used consistently throughout the paper? Does the paper avoid undefined shifts or redefinitions of terms, and does it align terminology with standard conventions in the field?""",

    "evaluation": """Criteria 1. Experimental Setup Clarity and Reproducibility Explanation 1. Evaluate how clearly and thoroughly the experimental setup is described. Does the paper provide all necessary information on hardware/software configurations, parameter settings, data preprocessing steps, and any contingency procedures, such that others could replicate the experiments with the same resources?
Criteria 2. Depth and Clarity of Figures and Tables Analysis Explanation 2. Evaluate the thoroughness and clarity of the paper's analysis of figures and tables. Are the data clearly explained and linked to the research objectives or hypotheses? Do the authors discuss trends, patterns, or anomalies, and interpret quantitative metrics in a way that highlights their significance? Is there a clear comparison to baselines or related work, demonstrating how the results fit into or advance the field? Do the authors emphasize key takeaways and practical or theoretical implications arising from the findings?
Criteria 3. Experimental Support for Main Innovations Explanation 3. Evaluate how thoroughly the paper's main innovations or contributions are backed by experimental evidence. Does the paper provide direct tests or comparisons to validate each innovation? Are quantitative or qualitative results clearly linked to the claims made, with appropriate metrics and comparisons against baselines or existing methods? Are ablation studies or sensitivity analyses included to demonstrate the significance of each component? If certain claims are not experimentally supported, have the authors either provided additional experiments or adjusted their claims accordingly?""",

    "conclusion": """Criteria 1. Broader Impact and Future Directions Explanation 1. Assess the thoroughness of the paper's conclusion or discussion sections in addressing the broader impact of the research, including societal, technological, regulatory, or academic implications. Does the paper provide specific and clear avenues for future work that build upon the current findings? Are these points presented cohesively and with enough detail to demonstrate the study's relevance beyond its immediate scope?
Criteria 2. Clarity and Impact of Key Innovations and Findings Explanation 2. Evaluate whether the conclusion effectively highlights and summarizes the paper's key innovations and major findings. Does it clearly restate the novel aspects of the work (e.g., methodologies, frameworks, or algorithms) and present concrete results or evidence of improvements over existing approaches? Does the conclusion provide sufficient detail on how these innovations contribute to the field, including any quantitative or qualitative outcomes, and compare them to previous work to demonstrate significance?""",
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
