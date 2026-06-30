# PIPO SDPO

## Overview

This branch contains the official SDPO-codebase implementation used for the self-distillation experiments in the PIPO paper.

It supports:

1. **SDPO** on SciKnowEval.
2. **SDPO + PIPO** on SciKnowEval.

We implement PIPO directly on top of the official SDPO repository to preserve the original SDPO training configuration.

Other code versions are provided in separate branches:

| Branch | Base codebase | Experiments |
| --- | --- | --- |
| [`main`](https://github.com/JacckMa/pipo_verl/tree/main) | verl v0.6.1 | PPO math, code RL, tool-use RL |
| [`verl-0.5-math`](https://github.com/JacckMa/pipo_verl/tree/verl-0.5-math) | verl v0.5 development stack | GRPO, GSPO, DAPO math |
| [`sdpo`](https://github.com/JacckMa/pipo_verl/tree/sdpo) | official SDPO codebase | SDPO and SDPO+PIPO on SciKnowEval |

## Installation

Follow the SDPO environment setup.

```bash
conda create -n pipo-sdpo python=3.12
conda activate pipo-sdpo
bash scripts/install_vllm_sglang_mcore.sh
pip install -e .
```

## Dataset

The processed SciKnowEval splits used in our SDPO experiments are kept under `datasets/sciknoweval/`. Change the paths in the scripts according to your machine if you rebuild or relocate the data.

## Training

Set model, data, and output paths in the corresponding scripts before launching. SDPO and SDPO+PIPO launchers are organized under `scripts/sdpo_tasks/`.

## Main Implementation

- `verl/trainer/ppo/layback_utils.py`: policy-improvement feedback and historical anchor.
- `verl/trainer/ppo/core_algos.py`: SDPO/PIPO objective integration.
- `verl/trainer/ppo/ray_trainer.py`: training-loop integration.
- `verl/trainer/config/sdpo.yaml`: SDPO and PIPO configuration.
- `data/format/sciknoweval.py`: SciKnowEval formatting.
- `scripts/sdpo_tasks/`: experiment launchers.

## Acknowledgements

This implementation is built on top of the official [SDPO](https://github.com/lasgroup/SDPO) codebase and the underlying verl infrastructure. We thank the SDPO and verl authors for their open-source implementations.
