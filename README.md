<div align="center">

# Policy Improvement Reinforcement Learning

</div>

This is the official implementation of our paper "**Policy Improvement Reinforcement Learning**" by Huaiyang Wang*, Xiaojie Li*, Deqing Wang, Haoyi Zhou, Zixuan Huang, Yaodong Yang, Jianxin Li, and Yikun Ban&dagger;.

<div align="center">
    <a href="https://arxiv.org/abs/2604.00860"><img src="https://img.shields.io/badge/Paper-%23FF2442?style=for-the-badge"></a>
    <a href="https://jacckma.github.io/pirl/"><img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge"></a>
    <a href="https://github.com/JacckMa/pirl_code"><img src="https://img.shields.io/badge/Code-%2300B4D8?style=for-the-badge"></a>
</div>

**Contact:** For any codebase related questions, please reach out to [Huaiyang Wang](mailto:huaiyangwang@buaa.edu.cn) or [Yikun Ban](mailto:yikunb@buaa.edu.cn).


## Installation

This repository is built upon the [Verl](https://github.com/verl-project/verl) codebase. We provide two methods to set up the environment.

### Method 1: Docker

1. Pull the base Verl image:
```bash
docker pull verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2
```

1. Launch container and install PIRL:
```bash
docker run --runtime=nvidia --net=host --net=host --shm-size="10g" \
  -v "$PWD":workspace/pirl_code --name pirl_container -it \
  verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 bash

# Inside the container
cd /workspace/pirl_code
pip install --no-deps -e .
```

### Method 2: Conda

```bash
conda create --name pirl python=3.12
conda activate pirL
pip install -r requirements.txt
pip install -e .
```


## Reproducing our experiments

### 1. Data & Model

**Datasets**: The MATH and SciKnowEval datasets are included in the `dataset/` folder.

**Model**: This codebase supports Qwen3-4B-Base and Qwen3-8B-Base. You can download the model weights using the provided script:

```bash
cd scripts/
bash down_model.sh
```

> Note:  Please make sure to update the model and checkpoint paths in the configuration files under `verl/trainer/config/experiments/` to match your local setup.

### 2. Running Experiments

*(Optional) We recommend running these experiments on at least 8 NVIDIA A100 (80GB) GPUs.*

#### MATH Dataset

To reproduce the results on the MATH dataset:

**GRPO (Baseline):**
```bash
./scripts/run_experiment.sh experiments/math/grpo
```

**GRPO + PIPO (Ours):**
```bash
./scripts/run_experiment.sh experiments/math/grpo_pipo
```

#### SciKnowEval Dataset

Replace `{domain}` with `biology`, `chemistry`, `physics`, or `material`:

**GRPO (Baseline):**
```bash
./scripts/run_experiment.sh experiments/sciknoweval/grpo_{domain}_baseline
```

**GRPO + PIPO (Ours):**
```bash
./scripts/run_experiment.sh experiments/sciknoweval/grpo_{domain}_pipo
```

## PIPO Hyperparameters

The following parameters are critical for reproducing the results reported in the paper:

| Parameter | Value | Description |
| --- | --- | --- |
| `history_window_size` (K) | **8** | Sliding window size for historical baseline |
| `progress_scale_min` | **0** | Lower bound for progress scale (ReLU truncation) |
| `progress_scale_max` | **0.5** | Upper bound for progress scale |
| `rollout.n` (G) | **8** | Number of responses sampled per prompt |

These parameters are configured in the YAML files under `verl/trainer/config/experiments/`.

## Repository Structure

```
.
├── verl/
│   ├── trainer/
│   │   ├── config/experiments/    # Experiment configurations
│   │   │   ├── math/              
│   │   │   └── sciknoweval/       
│   │   └── ppo/                   # Core training logic
│   └── workers/                   # Actor/rollout workers
├── dataset/                       
├── scripts/                       
└── requirements.txt               
```

## Acknowledgements

The codebase for the algorithm is built on top of [Verl](https://github.com/verl-project/verl), and we express our gratitude to the authors of Verl for providing us with an easy-to-work-with codebase.

## Citation

If you find this repository useful for your research, please consider citing our paper:

```
@article{wang2026policyimprovement,
    title={Policy Improvement Reinforcement Learning},
    author={Wang, Huaiyang and Li, Xiaojie and Wang, Deqing and Zhou, Haoyi and Huang, Zixuan and Yang, Yaodong and Li, Jianxin and Ban, Yikun},
    journal={arXiv preprint arXiv:2604.00860},
    year={2026},
}
```
