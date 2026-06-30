# SciKnowEval Scripts

These scripts run PPO, GRPO, GSPO, DAPO, and their PIPO variants on SciKnowEval.

Prepare the data:

```bash
python scripts/sciknoweval/prepare_sciknoweval_data.py
```

Run one task:

```bash
bash scripts/sciknoweval/run_grpo.sh chemistry
bash scripts/sciknoweval/run_grpo_pipo.sh chemistry
```

Run all configured subjects:

```bash
ALGOS="ppo grpo gspo dapo" bash scripts/sciknoweval/run_all.sh
ALGOS="ppo_pipo grpo_pipo gspo_pipo dapo_pipo" bash scripts/sciknoweval/run_all.sh
```

SDPO experiments are implemented in the SDPO-based code version.
