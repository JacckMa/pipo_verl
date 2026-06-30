def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Match the reward routing used by the verl0.5 math experiments.
    data_source = str(data_source)
    if data_source == "openai/gsm8k":
        from verl.utils.reward_score import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from verl.utils.reward_score import math_reward
        res = math_reward.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from verl.utils.reward_score import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source.startswith("knoveleng/Minerva"):
        from verl.utils.reward_score import latex_math
        res = latex_math.compute_score(solution_str, ground_truth)
    elif "OlympiadBench" in data_source:
        from verl.utils.reward_score import latex_math
        res = latex_math.compute_score(solution_str, ground_truth)
    else:
        from verl.utils.reward_score import math_reward
        res = math_reward.compute_score(solution_str, ground_truth)
    if isinstance(res, dict):
        return res
    return float(res)
