## LLaMA Factory

We use llama-factory for finetuning and running prediction of our models.

## Download
First, download the open-source code: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

```
llamafactory-cli train examples/paper_sft.yaml
```

## Run
```
./run_prediction.sh
```

## Format Conversion
After obtaining the prediction files, we need to ensure the format is suitable for [alpaca_eval](../6_component_evaluation/README.md) to process.

```
./run_converter.sh
```
