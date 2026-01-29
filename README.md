# Sanity Checks for Self-Preference in LLM Evaluators



## Installation

```bash
# Clone the repository
git clone https://github.com/username/self-preference-llm.git
cd self-preference-llm

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install matplotlib openai seaborn numpy scipy
```

## Project results
 
To see the results for verifiable datasets go to judge_swap_null_verif_smoke2 and go to analysis folder in each dataset

to see the results for Quality go to judge_swap_null_author_obfuscation/quality/analysis

To see the results for the chain of thought experiments go to judge_swap_null_verif_cot and go to the analysis folder in each dataset

to see the results for the DBG score datasets go to judge_swap_null_dbg_results and go to the analysis folder in each dataset

to see the results for CNN and XSUM go to CNN_and_XSUM results/cnn_results/cnn/analysis for cnn and CNN_and_XSUM results/xsum_result/xsum/analysis for xsum.

for entropy results, see per_reference_entropy.json

For the python files, check which arguments are required.
to run 



to make the proxy_robustness figure from the main figure run analyze_proxy_robustness.py, proxy_robustness_plot1.py, and proxy_robustness_plot2.py

to make the scatter plots for harmful self-preference vs task accuracy for before and after the evaluator quality baseline, run analyze_judge_self_preference_scatter.py (for cnn and xsum run analyze_judge_self_preference_scatter_cnn_xsum.py)






