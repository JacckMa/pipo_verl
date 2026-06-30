# PIPO verl v0.5 Math

## Overview

This codebase contains the verl v0.5 development-stack implementation used for the group-relative mathematical reasoning experiments in the PIPO paper.

It supports:

1. **GRPO + PIPO** for mathematical reasoning.
2. **GSPO + PIPO** for mathematical reasoning.
3. **DAPO + PIPO** for mathematical reasoning.

Other code versions are provided in separate branches:

| Branch | Base codebase | Experiments |
| --- | --- | --- |
| [`main`](https://github.com/JacckMa/pipo_verl/tree/main) | verl v0.6.1 | PPO math, code RL, tool-use RL |
| [`verl-0.5-math`](https://github.com/JacckMa/pipo_verl/tree/verl-0.5-math) | verl v0.5 development stack | GRPO, GSPO, DAPO math |
| [`sdpo`](https://github.com/JacckMa/pipo_verl/tree/sdpo) | official SDPO codebase | SDPO and SDPO+PIPO on SciKnowEval |

PPO math experiments and code/tool-use experiments are provided in the [`main`](https://github.com/JacckMa/pipo_verl/tree/main) branch, which is based on verl v0.6.1.

## Installation

Our installation mirrors the standard [verl](https://github.com/volcengine/verl) setup.

```bash
conda create -n pipo-verl05 python=3.10
conda activate pipo-verl05
pip install -r requirements.txt
pip install -e .
```

## Dataset

Download and preprocess data, then change the paths in the experiment configurations according to your machine. The math experiments use MATH training data and evaluate on MATH500, AIME 2025, AMC 2023, Minerva, and OlympiadBench.

## Training

Launchers and experiment configurations are organized under `scripts/` and `verl/trainer/config/experiments/math/`. Configs with `pipo` correspond to PIPO runs.

## Main Implementation

- `verl/trainer/ppo/layback_utils.py`: policy-improvement feedback and historical anchor.
- `verl/trainer/ppo/core_algos.py`: GRPO/GSPO PIPO-modulated policy objectives.
- `verl/trainer/ppo/ray_trainer.py`: training-loop integration.
- `recipe/dapo/`: DAPO training implementation.
- `scripts/run_experiment.sh`: experiment launcher.

## Acknowledgements

This implementation is built on top of [verl](https://github.com/volcengine/verl). We thank the verl team and community for the open-source RL post-training infrastructure.
