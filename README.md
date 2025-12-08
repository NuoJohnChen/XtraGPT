# XtraGPT

Friend Project: 
[PaperDebugger](https://github.com/PaperDebugger/paperdebugger)

This repo comprises those components:

---

## 1. Data Preprocessing

See [here](https://github.com/NuoJohnChen/paperLLM/tree/f4dfa784cc5bf973d379a57efebc0afd9f9106d5).

See processed ReviseQA data [here](https://huggingface.co/datasets/Xtra-Computing/ReviseQA).

## 2. Fine-tuning and Prediction: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

We utilize **LLaMA-Factory** for instruction-tuning and running predictions on LLMs. This toolkit enables efficient fine-tuning using popular techniques such as LoRA, QLoRA, and supports a wide range of base models.

Repository: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

See [here](./prediction/README.md).

See XtraGPT Model [here](https://huggingface.co/Xtra-Computing/XtraGPT-7B).

---

## 3. Component-wise Evaluation: [Modified AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)

To evaluate individual sections of a paper—**Title**, **Abstract**, **Introduction**, **Background**, **Evaluation**, and **Conclusion**—we use a modified version of **AlpacaEval**. This allows us to benchmark each component independently using LLM-based comparative evaluation.

Repository: [https://github.com/tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)

See [here](./6_component_evaluation/README.md).

---

## 4. Full Paper Evaluation: [AI Scientist](https://github.com/SakanaAI/AI-Scientist)

For holistic, end-to-end evaluation of entire research papers, we leverage the **AI Scientist** framework. This toolchain is designed for assessing scientific content quality using an LLM-centric pipeline.

Repository: [https://github.com/SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)

See [here](./full_paper_evaluation/README.md).

---

Please refer to the paper for methodology, dataset details, and results.

See [here](https://arxiv.org/abs/2505.11336).

---

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
