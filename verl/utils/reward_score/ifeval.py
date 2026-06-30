
import json
import re
import string


def _text(x):
    return "" if x is None else str(x)


def _lower(x):
    return _text(x).lower()


def _compare(value, relation, target):
    relation = _lower(relation)
    if relation in {"at least", "at_least", ">=", "more than or equal to"}:
        return value >= target
    if relation in {"less than", "less_than", "<"}:
        return value < target
    if relation in {"at most", "at_most", "<=", "no more than"}:
        return value <= target
    if relation in {"exactly", "equal to", "equals", "=="}:
        return value == target
    if relation in {"more than", ">"}:
        return value > target
    return value >= target


def _words(s):
    return re.findall(r"\b[\w'-]+\b", _text(s), flags=re.UNICODE)


def _sentences(s):
    parts = [p.strip() for p in re.split(r"[.!?]+(?:\s+|$)", _text(s)) if p.strip()]
    if parts:
        return parts
    return [p.strip() for p in _text(s).splitlines() if p.strip()]


def _paragraphs(s):
    s = _text(s).strip()
    if not s:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n|\*\*\*", s) if p.strip()]
    return paras if paras else [s]


def _strip_wrapping(s):
    s = _text(s).strip()
    if s.startswith("<think>") and "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    if "<response>" in s and "</response>" in s:
        s = s.split("<response>", 1)[1].split("</response>", 1)[0].strip()
    return s


def _count_keyword(s, keyword):
    return len(re.findall(r"\b" + re.escape(_text(keyword).lower()) + r"\b", _lower(s)))


def _bullet_count(s):
    return len(re.findall(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\S", _text(s)))


def _highlight_count(s):
    # Count simple markdown emphasis spans, avoiding bullet markers.
    return len(re.findall(r"(?<!\*)\*[^*\n][^*\n]*[^*\n]\*(?!\*)", _text(s)))


def _placeholder_count(s):
    return len(re.findall(r"\[[^\[\]\n]{1,80}\]", _text(s)))


def _has_title(s):
    return bool(re.search(r"<<[^<>\n]+>>", _text(s)))


def _is_json(s):
    try:
        json.loads(_text(s).strip())
        return True
    except Exception:
        return False


def _has_postscript(s, marker):
    marker = _text(marker or "P.S.")
    return marker.lower() in _lower(s)


def _mostly_lowercase_english(s):
    letters = [c for c in _text(s) if c.isalpha()]
    if not letters:
        return False
    uppers = sum(c.isupper() for c in letters)
    return uppers == 0


def _mostly_uppercase_english(s):
    letters = [c for c in _text(s) if c.isalpha()]
    if not letters:
        return False
    uppers = sum(c.isupper() for c in letters)
    return uppers / len(letters) > 0.8


def _capital_word_count(s):
    return sum(1 for w in re.findall(r"\b[A-Z]{2,}\b", _text(s)))


def _script_ratio(s, ranges):
    letters = [c for c in _text(s) if c.isalpha()]
    if not letters:
        return 0.0
    hit = 0
    for c in letters:
        o = ord(c)
        if any(lo <= o <= hi for lo, hi in ranges):
            hit += 1
    return hit / len(letters)


def _language_ok(s, lang):
    s = _text(s)
    ranges = {
        "hi": [(0x0900, 0x097F)],
        "mr": [(0x0900, 0x097F)],
        "ne": [(0x0900, 0x097F)],
        "kn": [(0x0C80, 0x0CFF)],
        "pa": [(0x0A00, 0x0A7F)],
        "gu": [(0x0A80, 0x0AFF)],
        "te": [(0x0C00, 0x0C7F)],
        "ta": [(0x0B80, 0x0BFF)],
        "ko": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
        "ru": [(0x0400, 0x04FF)],
        "bg": [(0x0400, 0x04FF)],
        "fa": [(0x0600, 0x06FF)],
        "ar": [(0x0600, 0x06FF)],
        "ur": [(0x0600, 0x06FF)],
        "th": [(0x0E00, 0x0E7F)],
        "bn": [(0x0980, 0x09FF)],
    }
    if lang in ranges:
        return _script_ratio(s, ranges[lang]) > 0.25
    low = _lower(s)
    latin_letters = [c for c in low if c.isalpha()]
    if not latin_letters:
        return False
    if lang == "vi":
        return any(ch in low for ch in "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
    if lang == "de":
        return any(w in low.split() for w in ["der", "die", "das", "und", "nicht", "ein", "eine"]) or any(ch in low for ch in "äöüß")
    if lang == "pt":
        return any(w in low.split() for w in ["de", "que", "não", "para", "com", "uma"]) or any(ch in low for ch in "ãõçáéíóúâêô")
    if lang == "it":
        return any(w in low.split() for w in ["che", "non", "per", "con", "una", "gli", "della"])
    if lang == "fi":
        return any(w in low.split() for w in ["ja", "on", "ei", "että", "kun"]) or any(ch in low for ch in "äö")
    if lang == "sw":
        return any(w in low.split() for w in ["na", "kwa", "ni", "ya", "katika", "hii"])
    return True


def _check(iid, kwargs, response):
    response = _strip_wrapping(response)
    kwargs = kwargs or {}
    if iid == "punctuation:no_comma":
        return "," not in response
    if iid == "keywords:existence":
        return all(_text(k).lower() in _lower(response) for k in kwargs.get("keywords", []))
    if iid == "keywords:forbidden_words":
        return all(_text(k).lower() not in _lower(response) for k in kwargs.get("forbidden_words", []))
    if iid == "keywords:frequency":
        return _compare(_count_keyword(response, kwargs.get("keyword", "")), kwargs.get("relation", "at least"), int(kwargs.get("frequency", 1)))
    if iid == "keywords:letter_frequency":
        return _compare(_lower(response).count(_lower(kwargs.get("letter", ""))), kwargs.get("let_relation", "at least"), int(kwargs.get("let_frequency", 1)))
    if iid == "length_constraints:number_words":
        return _compare(len(_words(response)), kwargs.get("relation", "at least"), int(kwargs.get("num_words", 0)))
    if iid == "length_constraints:number_sentences":
        return _compare(len(_sentences(response)), kwargs.get("relation", "at least"), int(kwargs.get("num_sentences", 0)))
    if iid == "length_constraints:number_paragraphs":
        return len(_paragraphs(response)) == int(kwargs.get("num_paragraphs", 0))
    if iid == "length_constraints:nth_paragraph_first_word":
        paras = _paragraphs(response)
        n = int(kwargs.get("nth_paragraph", 1)) - 1
        if len(paras) != int(kwargs.get("num_paragraphs", len(paras))) or n < 0 or n >= len(paras):
            return False
        words = _words(paras[n])
        return bool(words) and words[0].lower().strip(string.punctuation) == _lower(kwargs.get("first_word", ""))
    if iid == "detectable_format:number_highlighted_sections":
        return _highlight_count(response) >= int(kwargs.get("num_highlights", 1))
    if iid == "detectable_format:number_bullet_lists":
        return _bullet_count(response) == int(kwargs.get("num_bullets", 0))
    if iid == "detectable_format:title":
        return _has_title(response)
    if iid == "detectable_format:json_format":
        return _is_json(response)
    if iid == "detectable_format:multiple_sections":
        splitter = _text(kwargs.get("section_spliter", "SECTION")).upper()
        num = int(kwargs.get("num_sections", 1))
        if splitter == "PARAGRAPH":
            return len(_paragraphs(response)) >= num
        return len(re.findall(re.escape(splitter), response, flags=re.I)) >= num - 1
    if iid == "detectable_format:constrained_response":
        return len(_words(response)) <= 10
    if iid == "detectable_content:number_placeholders":
        return _placeholder_count(response) >= int(kwargs.get("num_placeholders", 1))
    if iid == "detectable_content:postscript":
        return _has_postscript(response, kwargs.get("postscript_marker", "P.S."))
    if iid == "combination:repeat_prompt":
        return response.lstrip().startswith(_text(kwargs.get("prompt_to_repeat", "")).strip())
    if iid == "combination:two_responses":
        return "******" in response and len([p for p in response.split("******") if p.strip()]) == 2
    if iid == "startend:end_checker":
        return response.rstrip().endswith(_text(kwargs.get("end_phrase", "")).strip())
    if iid == "startend:quotation":
        r = response.strip()
        return len(r) >= 2 and ((r[0] == r[-1] == '"') or (r[0] == "“" and r[-1] == "”"))
    if iid == "change_case:english_lowercase":
        return _mostly_lowercase_english(response)
    if iid == "change_case:english_capital":
        return _mostly_uppercase_english(response)
    if iid == "change_case:capital_word_frequency":
        return _compare(_capital_word_count(response), kwargs.get("capital_relation", "at least"), int(kwargs.get("capital_frequency", 1)))
    if iid == "language:response_language":
        return _language_ok(response, kwargs.get("language", ""))
    return False


def compute_score(solution_str, ground_truth):
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except Exception:
            return 0.0
    ids = ground_truth.get("instruction_id_list", [])
    kwargs_list = ground_truth.get("kwargs", [{} for _ in ids])
    if not ids:
        return 0.0
    results = [_check(iid, kw, solution_str) for iid, kw in zip(ids, kwargs_list)]
    return 1.0 if all(results) else 0.0
