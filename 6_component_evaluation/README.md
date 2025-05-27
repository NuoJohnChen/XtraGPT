# Run Alpaca Eval

Alpaca Eval tries to simulate human judgement using a preference model (their ML model is trained on human-labelled data).

Here, we seek to use the pairwise evaluation functionality to determine if the modifed output is indeed better than the original.

## Create Environment
> conda create --name alpaca_eval python=3.10
> conda activate alpaca_eval

## Install alpaca-eval
First, download [alpaca-eval](https://github.com/tatsu-lab/alpaca_eval) from the open-source repository. We need to download the source code because we will be making some modifications.

Then, install required packages with:
```
pip install -r requirements.txt
```

## Modified Version
Replace `alpaca_eval/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4_turbo_fn` with `./alpaca_eval_gpt4_turbo_fn`. This modifies the judgging process.

## Prepare Dataset of JSON Format

After obtaining the prediction json files, you should have run [converter](../prediction/run_converter.sh) to convert the prediction files into a format suitable for alpaca-eval.

## Ready API Key
```
export OPENAI_API_KEY=<your_api_key>
```

## Run
```
./run_scorer.sh
```
