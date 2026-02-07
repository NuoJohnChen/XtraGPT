"""
Example: Basic usage of paper-revision-bench
"""

from paper_revision_bench import evaluate, list_sections, list_criteria

# Example 1: Simple evaluation
print("=" * 60)
print("Example 1: Simple Evaluation")
print("=" * 60)

original_texts = [
    "The method is very good and achieves good results.",
    "We propose a new approach that works well on the task.",
]

revised_texts = [
    "The method achieves state-of-the-art performance, outperforming baselines by 15%.",
    "We propose a novel attention-based approach that improves accuracy by 12% on standard benchmarks.",
]

# Note: You need to set OPENAI_API_KEY environment variable or pass api_key parameter
# results = evaluate(
#     original_texts=original_texts,
#     revised_texts=revised_texts,
#     section="abstract",
#     criterion="impact",
#     judge_model="gpt-4-turbo",
# )
# print(results.summary())

# Example 2: List available sections and criteria
print("\n" + "=" * 60)
print("Example 2: Available Sections and Criteria")
print("=" * 60)

print("\nAvailable sections:")
for section in list_sections():
    print(f"  - {section}")

print("\nCriteria for 'abstract' section:")
for criterion in list_criteria("abstract"):
    print(f"  - {criterion}")

print("\nCriteria for 'title' section:")
for criterion in list_criteria("title"):
    print(f"  - {criterion}")

# Example 3: With context
print("\n" + "=" * 60)
print("Example 3: Evaluation with Paper Context")
print("=" * 60)

paper_context = """
We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence
and convolutions entirely. Experiments on two machine translation
tasks show these models to be superior in quality while being more
parallelizable and requiring significantly less time to train.
"""

original = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."
revised = "Sequence transduction has traditionally relied on RNNs and CNNs with attention mechanisms."

print(f"Original: {original}")
print(f"Revised: {revised}")
print(f"Context: {paper_context[:100]}...")

# Uncomment to run (requires API key):
# results = evaluate(
#     original_texts=[original],
#     revised_texts=[revised],
#     contexts=[paper_context],
#     section="introduction",
#     criterion="conciseness",
#     judge_model="gpt-4-turbo",
# )
# print(results.summary())

print("\n" + "=" * 60)
print("To run actual evaluation, set OPENAI_API_KEY and uncomment the evaluate() calls")
print("=" * 60)
