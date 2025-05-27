# XtraGPT

This repo comprises three main components:

---

## 1. Fine-tuning and Prediction: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

We utilize **LLaMA-Factory** for instruction-tuning and running predictions on LLMs. This toolkit enables efficient fine-tuning using popular techniques such as LoRA, QLoRA, and supports a wide range of base models.

Repository: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

See [here](./prediction/README.md).

---

## 2. Component-wise Evaluation: [Modified AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)

To evaluate individual sections of a paper—**Title**, **Abstract**, **Introduction**, **Background**, **Evaluation**, and **Conclusion**—we use a modified version of **AlpacaEval**. This allows us to benchmark each component independently using LLM-based comparative evaluation.

Repository: [https://github.com/tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)

See [here](./6_component_evaluation/README.md).
---

## 3. Full Paper Evaluation: [AI Scientist](https://github.com/SakanaAI/AI-Scientist)

For holistic, end-to-end evaluation of entire research papers, we leverage the **AI Scientist** framework. This toolchain is designed for assessing scientific content quality using an LLM-centric pipeline.

Repository: [https://github.com/SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)

---

Please refer to the paper for methodology, dataset details, and results.

See [here](./full_paper_evaluation/README.md).
