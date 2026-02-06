"""
XtraGPT Inference Example

This script demonstrates how to use XtraGPT for paper revision.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = "Xtra-Computing/XtraGPT-7B"):
    """Load XtraGPT model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer


def revise_paper_section(
    model,
    tokenizer,
    paper_content: str,
    selected_content: str,
    instruction: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.1
) -> str:
    """
    Revise a section of a paper based on the given instruction.

    Args:
        model: The loaded XtraGPT model
        tokenizer: The tokenizer
        paper_content: Full paper content for context
        selected_content: The specific text to revise
        instruction: Revision instruction (e.g., "Make this more concise")
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Revised text
    """
    prompt = f"""Act as an expert model for improving articles **PAPER_CONTENT**.
The output needs to answer the **QUESTION** on **SELECTED_CONTENT** in the input. Avoid adding unnecessary length, unrelated details, overclaims, or vague statements.
Focus on clear, concise, and evidence-based improvements that align with the overall context of the paper.
<PAPER_CONTENT>
{paper_content}
</PAPER_CONTENT>
<SELECTED_CONTENT>
{selected_content}
</SELECTED_CONTENT>
<QUESTION>
{instruction}
</QUESTION>"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0
    )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response


def main():
    # Example usage
    model, tokenizer = load_model("Xtra-Computing/XtraGPT-7B")

    # Example paper content (from "Attention Is All You Need")
    paper_content = """
    The dominant sequence transduction models are based on complex recurrent or
    convolutional neural networks in an encoder-decoder configuration. The best
    performing models also connect the encoder and decoder through an attention
    mechanism. We propose a new simple network architecture, the Transformer,
    based solely on attention mechanisms, dispensing with recurrence and
    convolutions entirely. Experiments on two machine translation tasks show
    these models to be superior in quality while being more parallelizable and
    requiring significantly less time to train.
    """

    selected_content = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration."

    instruction = "Make this sentence more concise and impactful."

    print("=" * 60)
    print("XtraGPT Paper Revision Example")
    print("=" * 60)
    print(f"\nOriginal text:\n{selected_content}")
    print(f"\nInstruction: {instruction}")
    print("\nGenerating revision...")

    revised = revise_paper_section(
        model,
        tokenizer,
        paper_content,
        selected_content,
        instruction
    )

    print(f"\nRevised text:\n{revised}")
    print("=" * 60)


if __name__ == "__main__":
    main()
