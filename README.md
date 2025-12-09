# XtraGPT

Refiner of [Friend Project: PaperDebugger](https://github.com/PaperDebugger/paperdebugger)

## Overview

**XtraGPT** is a family of open-source Large Language Models (LLMs) designed specifically for **human-AI collaborative academic paper revision**. Unlike general-purpose models that often perform surface-level polishing, XtraGPT is fine-tuned to **understand the full context** of a research paper and execute specific, **criteria-guided** revision instructions.

The models were trained on a dataset of 140,000 high-quality instruction-revision pairs derived from top-tier conference papers (ICLR).

**Key Features:**

* **Context-Aware:** Processes the full paper context to ensure revisions maintain consistency with the global narrative.
* **Controllable:** Follows specific user instructions aligned with 20 academic writing criteria across 6 sections (Abstract, Introduction, etc.).
* **Iterative Workflow:** Designed to support the "Human-AI Collaborative" (HAC) lifecycle where authors retain creative control.

---

## Inference with Transformers

To use XtraGPT with the standard Hugging Face `transformers` library, ensure you format your input using the specific tags `<PAPER_CONTENT>`, `<SELECTED_CONTENT>`, and `<QUESTION>`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Select the model size: "XtraGPT-1.5B", "XtraGPT-3B", "XtraGPT-7B", or "XtraGPT-14B"
model_name = "Xtra-Computing/XtraGPT-7B" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define the Prompt Template tailored for XtraGPT
prompt_template = """Act as an expert model for improving articles **PAPER_CONTENT**.
The output needs to answer the **QUESTION** on **SELECTED_CONTENT** in the input. Avoid adding unnecessary length, unrelated details, overclaims, or vague statements.
Focus on clear, concise, and evidence-based improvements that align with the overall context of the paper.
<PAPER_CONTENT>
{paper_content}
</PAPER_CONTENT>
<SELECTED_CONTENT>
{selected_content}
</SELECTED_CONTENT>
<QUESTION>
{user_question}
</QUESTION>"""

# Example Data (from the "Attention Is All You Need" paper)
paper_content = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train."
selected_content = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration."
user_question = "help me make it more concise."

# Format the input
formatted_prompt = prompt_template.format(
    paper_content=paper_content,
    selected_content=selected_content,
    user_question=user_question
)

messages = [
    {"role": "user", "content": formatted_prompt}
]

# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384,
    temperature=0.1
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

---

## Repo Components

This repo comprises those components:

### 1. Data Preprocessing

See [here](https://github.com/NuoJohnChen/paperLLM/tree/f4dfa784cc5bf973d379a57efebc0afd9f9106d5).

See processed ReviseQA data [here](https://huggingface.co/datasets/Xtra-Computing/ReviseQA).

### 2. Fine-tuning and Prediction

We utilize **LLaMA-Factory** for controllable post-training and running predictions on LLMs. The toolkit enables efficient fine-tuning using popular techniques such as LoRA, QLoRA, and supports a wide range of base models.

See [here](./train/README.md).

See XtraGPT Model [here](https://huggingface.co/Xtra-Computing/XtraGPT-7B).

### 3. Component-wise Evaluation

To evaluate individual sections of a paper, including **Title**, **Abstract**, **Introduction**, **Background**, **Evaluation**, and **Conclusion**, we use a modified version of **AlpacaEval**. This allows us to benchmark each component independently using LLM-based comparative evaluation.

See [here](./6_component_evaluation/README.md).

### 4. Full Paper Evaluation

For holistic, end-to-end evaluation of entire research papers, we leverage the **AI Scientist** framework. This toolchain is designed for assessing scientific content quality using an LLM-centric pipeline.

See [here](./full_paper_evaluation/README.md).

Please refer to the paper for methodology, dataset details, and results.

See [here](https://arxiv.org/abs/2505.11336).

## Model License

This model is released under the **ModelGo Zero License 2.0 (MG0-2.0)**.

MG0-2.0 is a highly permissive open model license designed to facilitate the widest possible adoption and collaboration. It allows for **unrestricted use**, reproduction, distribution, and the creation of derivative works including for commercial purposes, without requiring attribution or imposing copyleft restrictions.

For more details on the license terms, please visit [ModelGo.li](https://www.modelgo.li/) or refer to the `LICENSE` file in the repository.

## 🙌 Acknowledgements

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 🔗
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) 🔗
- [AI Scientist](https://github.com/SakanaAI/AI-Scientist) 🔗

## Citation

```
@misc{nuo2025xtragpt,
      title={XtraGPT: LLMs for Human-AI Collaboration on Controllable Academic Paper Revision}, 
      author={Nuo Chen, Andre Lin HuiKai, Jiaying Wu, Junyi Hou, Zining Zhang, Qian Wang, Xidong Wang, Bingsheng He},
      year={2025},
      eprint={2505.11336},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.11336}, 
}
```
