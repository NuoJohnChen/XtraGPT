"""
Example: Using different judge models (local and cloud)
"""

from paper_revision_bench import evaluate

# Sample data
original = "The method achieves good performance on the dataset."
revised = "Our method achieves 95.2% accuracy on ImageNet, surpassing the previous state-of-the-art by 2.3%."


def example_openai():
    """Using OpenAI models."""
    print("=" * 60)
    print("OpenAI Judge (gpt-4-turbo)")
    print("=" * 60)

    results = evaluate(
        original_texts=[original],
        revised_texts=[revised],
        judge_model="gpt-4-turbo",
        # api_key="sk-xxx",  # or set OPENAI_API_KEY env var
    )
    print(results.summary())


def example_anthropic():
    """Using Anthropic Claude models."""
    print("=" * 60)
    print("Anthropic Judge (claude-3-opus)")
    print("=" * 60)

    results = evaluate(
        original_texts=[original],
        revised_texts=[revised],
        judge_model="claude-3-opus-20240229",
        # api_key="sk-ant-xxx",  # or set ANTHROPIC_API_KEY env var
    )
    print(results.summary())


def example_ollama():
    """Using local Ollama models."""
    print("=" * 60)
    print("Ollama Judge (llama3:70b)")
    print("=" * 60)

    # First, make sure Ollama is running:
    # ollama serve
    # ollama pull llama3:70b

    results = evaluate(
        original_texts=[original],
        revised_texts=[revised],
        judge_model="ollama/llama3:70b",
        api_base="http://localhost:11434",
    )
    print(results.summary())


def example_vllm():
    """Using vLLM-served models."""
    print("=" * 60)
    print("vLLM Judge (Llama-3-70b)")
    print("=" * 60)

    # First, start vLLM server:
    # python -m vllm.entrypoints.openai.api_server \
    #     --model meta-llama/Meta-Llama-3-70B-Instruct \
    #     --port 8000

    results = evaluate(
        original_texts=[original],
        revised_texts=[revised],
        judge_model="vllm/meta-llama/Meta-Llama-3-70B-Instruct",
        api_base="http://localhost:8000",
    )
    print(results.summary())


def example_openai_compatible():
    """Using any OpenAI-compatible API."""
    print("=" * 60)
    print("OpenAI-Compatible API")
    print("=" * 60)

    # Works with any OpenAI-compatible endpoint:
    # - Together AI
    # - Anyscale
    # - Fireworks AI
    # - Local LLM servers

    results = evaluate(
        original_texts=[original],
        revised_texts=[revised],
        judge_model="meta-llama/Llama-3-70b-chat-hf",
        api_key="your-api-key",
        api_base="https://api.together.xyz/v1",
    )
    print(results.summary())


if __name__ == "__main__":
    print("Different Judge Models Example")
    print("Uncomment the example you want to run\n")

    # Uncomment one of these to run:
    # example_openai()
    # example_anthropic()
    # example_ollama()
    # example_vllm()
    # example_openai_compatible()

    print("Set appropriate API keys and uncomment an example to run")
