# AI-SCIENTIST 

We use the highly acclaimed ai-scientist for full paper evaluation.

## Download
First, download [ai-scientist](https://github.com/SakanaAI/AI-Scientist)

## Validation
We validated ai-scientist reliablity as a full paper evaluator by analyzing how closely its predicted decision (accept/reject) of a given conference paper aligns with the actual. The code is in [validate_ai_scientist.py](./validate_ai_scientist.py)

## Evaluation
We evaluated XtraGPT modified papers with [ai_sicentist.py](./ai_scientist.py). The results are stored in [paper_results](./paper_results/)

## Analysis
We analyze improvements in acceptance rate in [analyze_improvement.py](./analyze_improvement.py) and plots used in the diagram detailing improvements to different metrics such as `soundness`, `presentation`, `contribution` are found in [xtragpt_plots](./xtragpt_plots/). The code for the plots can be found in [plot.py](./plot.py)
