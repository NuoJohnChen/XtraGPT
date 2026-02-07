"""
Evaluation criteria and prompts for different paper sections.
"""

from typing import Dict, List, Optional


# Paper sections
SECTIONS = [
    "title",
    "abstract",
    "introduction",
    "background",
    "evaluation",
    "conclusion",
]

# Criteria for each section
SECTION_CRITERIA: Dict[str, List[str]] = {
    "title": [
        "overall",
        "conciseness",
        "clarity",
        "impact",
        "specificity",
        "accuracy",
    ],
    "abstract": [
        "overall",
        "conciseness",
        "clarity",
        "completeness",
        "impact",
        "coherence",
        "motivation",
        "contribution",
    ],
    "introduction": [
        "overall",
        "clarity",
        "motivation",
        "coherence",
        "flow",
        "context",
        "contribution",
        "organization",
    ],
    "background": [
        "overall",
        "clarity",
        "completeness",
        "relevance",
        "organization",
        "coherence",
        "accuracy",
    ],
    "evaluation": [
        "overall",
        "clarity",
        "completeness",
        "rigor",
        "presentation",
        "analysis",
        "reproducibility",
    ],
    "conclusion": [
        "overall",
        "conciseness",
        "clarity",
        "completeness",
        "impact",
        "future_work",
    ],
}

# Base prompt template - using $$ for JSON braces to avoid format() conflicts
BASE_PROMPT_RAW = """You are an expert academic writing evaluator. Your task is to compare two versions of a text from a research paper and determine which one is better.

## Task
Compare the ORIGINAL text with the REVISED text and decide which is better based on the evaluation criterion.

## Evaluation Criterion: {criterion_name}
{criterion_description}

{context_section}

## ORIGINAL TEXT
{original}

## REVISED TEXT
{revised}

## Instructions
1. Carefully analyze both texts based on the evaluation criterion
2. Consider academic writing standards and best practices
3. Provide your judgment and explanation

## Output Format
You must respond with a JSON object in the following format:
{"winner": "revised" or "original" or "tie", "score": <float 0-1>, "explanation": "<brief explanation>"}

Respond ONLY with the JSON object, no other text."""

CONTEXT_TEMPLATE = """## PAPER CONTEXT
The following is the full context of the paper for reference:
{context}
"""

# Criterion descriptions
CRITERION_DESCRIPTIONS: Dict[str, str] = {
    "overall": "Evaluate the overall quality of the text, considering clarity, conciseness, coherence, and academic writing standards.",
    "conciseness": "Evaluate whether the text is concise and avoids unnecessary words, redundancy, or verbosity while maintaining completeness.",
    "clarity": "Evaluate whether the text is clear, easy to understand, and free from ambiguity or confusing language.",
    "impact": "Evaluate whether the text is impactful, engaging, and effectively communicates the significance of the work.",
    "completeness": "Evaluate whether the text is complete and covers all necessary information without omitting important details.",
    "coherence": "Evaluate whether the text flows logically, with smooth transitions and well-connected ideas.",
    "motivation": "Evaluate whether the text effectively motivates the research problem and explains why it matters.",
    "contribution": "Evaluate whether the text clearly articulates the contributions and novelty of the work.",
    "flow": "Evaluate whether the text has good flow and readability, with well-structured sentences and paragraphs.",
    "context": "Evaluate whether the text provides appropriate context and background for the reader.",
    "organization": "Evaluate whether the text is well-organized with a logical structure.",
    "relevance": "Evaluate whether the content is relevant and focused on the topic at hand.",
    "accuracy": "Evaluate whether the text is accurate and free from factual errors or misleading statements.",
    "rigor": "Evaluate whether the text demonstrates scientific rigor and methodological soundness.",
    "presentation": "Evaluate whether the results and findings are presented clearly and effectively.",
    "analysis": "Evaluate whether the analysis is thorough, insightful, and well-explained.",
    "reproducibility": "Evaluate whether the text provides sufficient detail for reproducibility.",
    "future_work": "Evaluate whether future directions are clearly and appropriately discussed.",
    "specificity": "Evaluate whether the text is specific enough to accurately describe the work.",
}

# Section-specific prompt additions
SECTION_PROMPTS: Dict[str, str] = {
    "title": "You are evaluating a paper TITLE. A good title should be concise, specific, informative, and capture the essence of the paper.",
    "abstract": "You are evaluating a paper ABSTRACT. A good abstract should concisely summarize the motivation, methods, results, and conclusions.",
    "introduction": "You are evaluating a paper INTRODUCTION. A good introduction should motivate the problem, provide context, and clearly state contributions.",
    "background": "You are evaluating a BACKGROUND/RELATED WORK section. It should comprehensively cover relevant prior work and position the paper appropriately.",
    "evaluation": "You are evaluating an EVALUATION/EXPERIMENTS section. It should clearly present methodology, results, and analysis with appropriate rigor.",
    "conclusion": "You are evaluating a CONCLUSION section. It should summarize key findings, discuss implications, and suggest future directions.",
}


def list_sections() -> List[str]:
    """Return list of available paper sections."""
    return SECTIONS.copy()


def list_criteria(section: Optional[str] = None) -> List[str]:
    """
    Return list of available criteria.

    Args:
        section: If provided, return criteria for that section only.
                 If None, return all unique criteria.
    """
    if section:
        validate_section(section)
        return SECTION_CRITERIA[section].copy()

    # Return all unique criteria
    all_criteria = set()
    for criteria in SECTION_CRITERIA.values():
        all_criteria.update(criteria)
    return sorted(list(all_criteria))


def validate_section(section: str) -> None:
    """Validate that section is valid."""
    if section not in SECTIONS:
        raise ValueError(
            f"Invalid section: '{section}'. "
            f"Valid sections are: {SECTIONS}"
        )


def validate_criterion(section: str, criterion: str) -> None:
    """Validate that criterion is valid for the given section."""
    validate_section(section)
    valid_criteria = SECTION_CRITERIA[section]
    if criterion not in valid_criteria:
        raise ValueError(
            f"Invalid criterion '{criterion}' for section '{section}'. "
            f"Valid criteria are: {valid_criteria}"
        )


def get_criterion_prompt(section: str, criterion: str) -> str:
    """
    Get the full prompt template for a section and criterion.

    Args:
        section: Paper section (title, abstract, etc.)
        criterion: Evaluation criterion (conciseness, clarity, etc.)

    Returns:
        Prompt template string with placeholders for {original}, {revised}, {context}
    """
    validate_section(section)
    validate_criterion(section, criterion)

    section_intro = SECTION_PROMPTS.get(section, "")
    criterion_desc = CRITERION_DESCRIPTIONS.get(criterion, CRITERION_DESCRIPTIONS["overall"])

    # Build the full prompt
    prompt = f"{section_intro}\n\n{BASE_PROMPT_RAW}"

    return prompt.replace(
        "{criterion_name}", criterion.replace("_", " ").title()
    ).replace(
        "{criterion_description}", criterion_desc
    )


def format_prompt(
    template: str,
    original: str,
    revised: str,
    context: Optional[str] = None,
) -> str:
    """
    Format a prompt template with actual values.

    Args:
        template: Prompt template from get_criterion_prompt()
        original: Original text
        revised: Revised text
        context: Optional paper context

    Returns:
        Formatted prompt ready for the judge model
    """
    context_section = ""
    if context:
        context_section = CONTEXT_TEMPLATE.replace("{context}", context)

    return template.replace(
        "{context_section}", context_section
    ).replace(
        "{original}", original
    ).replace(
        "{revised}", revised
    )
