import re


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def is_correct_format(text: str) -> bool:
    pattern = r"<answer>\s*(A|B|C|D)\s*</answer>$"
    return re.search(pattern, text.strip()) is not None


def compute_score(solution_str: str, ground_truth: str, *args, **kwargs) -> dict:
    pred = extract_xml_answer(solution_str)
    score = float(pred == ground_truth)
    format_ok = is_correct_format(solution_str)
    return {
        "score": score,
        "acc": score,
        "pred": pred,
        "incorrect_format": 0.0 if format_ok else 1.0,
    }
