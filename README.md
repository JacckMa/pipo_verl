# PIPO: Policy Improvement Policy Optimization

## Overview

PIPO (Policy Improvement Policy Optimization) is a plug-and-play framework for closed-loop RL post-training. It augments a base policy optimization algorithm with policy-improvement feedback computed from subsequent batches, so that local learning signals can be reinforced or suppressed according to measured inter-iteration progress.

This codebase is built on **verl v0.6.1** and contains the implementation for:

1. **PPO mathematical reasoning** on MATH-style RL tasks.
2. **Code RL** on TACO, LiveCodeBench v6, HumanEval, and MBPP.
3. **Tool-use RL** on ToolRL/RLLA, BFCL, and APIBank.

The paper uses three code versions for different baseline families:

| Code version | Base codebase | Experiments |
| --- | --- | --- |
| [`main`](https://github.com/JacckMa/pipo_verl/tree/main) | verl v0.6.1 | PPO math, code RL, tool-use RL |
| [`verl-0.5-math`](https://github.com/JacckMa/pipo_verl/tree/verl-0.5-math) | verl v0.5 development stack | GRPO, GSPO, DAPO math |
| [`sdpo`](https://github.com/JacckMa/pipo_verl/tree/sdpo) | official SDPO codebase | SDPO and SDPO+PIPO on SciKnowEval |

SDPO experiments are implemented on top of the official SDPO codebase to match its original training setup. The verl v0.5 math branch contains the group-relative math experiments developed in that stack.

## Installation

Our installation mirrors the standard [verl](https://github.com/volcengine/verl) setup.

```bash
conda create -n pipo-verl python=3.12
conda activate pipo-verl
bash scripts/install_vllm_sglang_mcore.sh
pip install -e .
```

## Dataset

Download and preprocess data, then change the paths in the scripts according to your machine. The main datasets used in this version are:

- **Math**: MATH training data, MATH500, AIME 2025, AMC 2023, Minerva, and OlympiadBench.
- **Code**: TACO, LiveCodeBench v6, HumanEval, and MBPP.
- **Tool-use**: ToolRL/RLLA, BFCL, and APIBank.

Useful preprocessing helpers are provided under `scripts/code/` and `scripts/tooluse/`.

## Training

Set model, data, and output paths in the corresponding scripts before launching. Scripts with `pipo` in the name are PIPO runs.
Math runs use `scripts/math/pipoverl_math_reward_route.py` by default to keep the evaluation route consistent with the verl v0.5 math experiments.

The main launchers are organized by task:

- `scripts/math/`: PPO math experiments.
- `scripts/code/`: code RL experiments.
- `scripts/tooluse/`: tool-use RL experiments.

The GRPO, GSPO, and DAPO math experiments in the paper are provided in the [`verl-0.5-math`](https://github.com/JacckMa/pipo_verl/tree/verl-0.5-math) branch.

## Main Implementation

- `verl/trainer/ppo/layback_utils.py`: policy-improvement feedback and historical anchor.
- `verl/trainer/ppo/core_algos.py`: PIPO-modulated policy objectives.
- `verl/trainer/ppo/ray_trainer.py`: training-loop integration.
- `verl/trainer/config/algorithm.py`: PIPO configuration.
- `verl/utils/reward_score/`: reward functions for math, code, and tool-use tasks.

## Acknowledgements

This implementation is built on top of [verl](https://github.com/volcengine/verl). We thank the verl team and community for the open-source RL post-training infrastructure.

## Citation

```bibtex
@article{wang2026pirl,
  title={Policy Improvement Reinforcement Learning},
  author={Wang, Huaiyang and Li, Xiaojie and Wang, Xiaohan and Zhang, Zhixia and Lu, Xiaodong and Huang, Zixuan and Chai, Jiajun and Yin, Guojun and Wang, Deqing and Zhou, Haoyi and Yang, Yaodong and Li, Jianxin and Ban, Yikun},
  journal={arXiv preprint arXiv:2604.00860},
  year={2026}
}
```
