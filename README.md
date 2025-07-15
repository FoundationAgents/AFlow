# AFlow: Automating Agentic Workflow Generation

[![Arxiv](https://img.shields.io/badge/arXiv-AFlow-b31b1b)](https://arxiv.org/abs/2410.10762)
[![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/FoundationAgents/AFlow/pulls)

> If you encounter any difficulties in using or reproducing the code, please contact me directly (Email: didi4goooogle@gmail.com, Wechat: 18831933368). Some Operators may have bugs during the migration from MetaGPT to this repository.


AFlow is a framework for automatically generating and optimizing Agentic Workflows. It uses Monte Carlo tree search in a code-represented workflow space to find effective workflows, replacing manual development with machine effort. Our approach shows potential to outperform handcrafted workflows on various tasks. 

We're building it to support more benchmarks and open-ended tasks! If you have any questions, please open an issue or email us!

<p align="center">
<a href=""><img src="assets/AFLOW-performance.jpg" alt="Performance Of AFlow" title="Performance of AFlow<sub>1</sub>" width="80%"></a>
</p>

## Framework Components

- **Node**: Basic unit of LLM invocation. See `metagpt_core/action_nodes/action_node.py` for a flexible interface to control LLM, temperature, format, and prompt.
- **Operator**: Predefined combinations of Nodes to enhance search efficiency. Encapsulates common operations like Generate, Format, Review, Revise, Ensemble, Test, and Programmer. See `operator.py` for details. You can customize your own Operator by referencing the implementations in this code.
- **Workflow**: A sequence of LLM-invoking nodes connected by edges. Can be represented as graphs, neural networks, or code to express various execution structures. See `workflow.py` for our implementation.
- **Optimizer**: Uses LLMs within a Monte Carlo Tree Search variant to explore and refine workflows. Iteratively selects, expands, evaluates, and updates workflows based on performance. See `optimizer.py` for details.
- **Evaluator**: Assesses workflow performance on given tasks. Provides feedback to guide the optimization process towards more effective workflows. See `evaluator.py` for details.

<p align="center">
<a href=""><img src="assets/AFLOW-method.jpg" alt="Framework of AFlow" title="Framework of AFlow <sub>1</sub>" width="80%"></a>
</p>

## Datasets

### Experimental Datasets
We conducted experiments on six datasets (HumanEval, MBPP, GSM8K, MATH, HotpotQA, DROP) and provide their evaluation code. The data can be found in this [datasets](https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e) link, or you can download them using `metagpt/ext/aflow/data/download_data.py`

<p align="center">
<a href=""><img src="assets/AFLOW-experiment.jpg" alt="Performance Of AFlow" title="Performance Of AFlow <sub>1</sub>" width="80%"></a>
</p>

### Custom Datasets
For custom tasks, you can reference the code in the `benchmark` folder. Inherit the `BaseBenchmark` class and implement `evaluate_problem`, `calculate_score`, and `get_result_columns` to add your custom dataset benchmark. Then, add your benchmark name in `evaluator.py` and `optimizer.py` to find effective workflows for your custom dataset.

## Quick Start

1. Set up the Python environment:
   ```bash
   # Create and activate a Python 3.9 virtual environment
   conda create -n <your_env_name> python=3.9

   # Install dependencies
   pip install -r requirements.txt
   ```

2. Configure optimization parameters:
   - Use command line arguments or modify default parameters in `run.py`:
     ```python
     --dataset              # (Required) Dataset type (HumanEval/MBPP/GSM8K/MATH/HotpotQA/DROP)
     --sample 4             # Sample count - number of workflows to be resampled
     --optimized_path PATH  # Optimized result save path
     --initial_round 1      # Initial round
     --max_rounds 20        # Max iteration rounds for AFLOW
     --check_convergence    # Whether to enable early stop
     --validation_rounds 5  # Validation rounds for AFLOW
     --if_first_optimize    # Set True for first optimization, False afterwards
     ```

3. Configure LLM parameters in `config/config2.yaml` (see `config/config2.example.yaml` for reference)

4. Set up operators in `run.py` and in `operator.py`, `optimized_path/template/operator.json`. You can reference our implementation to add operators for specific datasets

5. For first-time use, download datasets and initial rounds by setting `download(["datasets"])` in `run.py`

6. (Optional) Add your custom dataset and corresponding evaluation function following the [Custom Datasets](#custom-datasets) section

7. (Optional) If you want to use a portion of the validation data, you can set `va_list` in `evaluator.py`

8. Run the optimization:
   ```bash
   # Using default parameters
   python run.py --dataset MATH
   
   # Or with custom parameters
   python run.py --dataset MATH --sample n --optimized_path xxx ...
   ```

## Reproduce the Results in the Paper
1. We provide the raw data obtained from our experiments in this [link](https://drive.google.com/uc?export=download&id=1Sr5wjgKf3bN8OC7G6cO3ynzJqD4w6_Dv), including the workflows and prompts generated in each iteration, as well as their trajectories on the validation dataset. We also provide the optimal workflow for each dataset and the corresponding data on the test dataset. You can download these data using `data/download_data.py`. 
2. You can directly reproduce our experimental results by use different `ExperimentConfig` of `run.py`.

## Roadmap

- Support multiple search algorithms
- Support multi model search in workflow
- Support LeaderBoard
- Support more benchmarks
- Support multimodality tasks

## Citation

If you use AFlow in your research, please cite our paper:

```
@inproceedings{
   zhang2025aflow,
   title={{AF}low: Automating Agentic Workflow Generation},
   author={Jiayi Zhang and Jinyu Xiang and Zhaoyang Yu and Fengwei Teng and Xiong-Hui Chen and Jiaqi Chen and Mingchen Zhuge and Xin Cheng and Sirui Hong and Jinlin Wang and Bingnan Zheng and Bang Liu and Yuyu Luo and Chenglin Wu},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=z5uVAKwmjf}
}
```
