import os, json, random, re, datetime, uuid
from typing import Optional, Tuple
from wordfreq import top_n_list
from analogies.utils import generate_inference

# Random + word source
RNG = random.Random()
WORDS = top_n_list("en", 50000)

# ---------- token + parsing ----------
def _is_simple_token(w: str) -> bool:
    return w.isascii() and w.isalpha()

def sample_word() -> str:
    while True:
        w = RNG.choice(WORDS).lower()
        if _is_simple_token(w):
            return w

def clean_answer(resp: str) -> str:
    resp = (resp or "").strip()
    m = re.search(r"ANSWER:\s*([A-Za-z\-']+)", resp, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"\b([A-Za-z\-']+)\b", resp)
    return m2.group(1).lower() if m2 else ""

# ---------- concept checking ----------
def is_concept(word: str, model: str) -> Tuple[bool, Optional[str], str]:
    prompt = (
        "Determine if the following English WORD is a concept "
        "(noun/verb/adjective/adverb), not a function word "
        "(conjunction/preposition/article/pronoun), nor a proper noun.\n"
        "Respond ONLY in this format:\n"
        "ANSWER: Yes or No\nCONCEPT: <canonical form>\n\n"
        f"{word}\n"
    )
    resp = generate_inference(prompt, model)
    ans = re.search(r'ANSWER:\s*(Yes|No)', resp, re.IGNORECASE)
    con = re.search(r'CONCEPT:\s*(.*)', resp, re.IGNORECASE)
    is_yes = bool(ans and ans.group(1).lower() == "yes")
    norm = con.group(1).strip().lower() if con else None
    return is_yes, norm, resp

def sample_concept(model: str, *, verbose: bool = True) -> str:
    # Retry until we get one the LLM says "relies on concept" (Yes)
    tries = 0
    while True:
        tries += 1
        w = sample_word()
        ok, norm, raw = is_concept(w, model)
        if verbose:
            print(f"[is_concept] try#{tries} word={w} ok={ok} norm={norm} raw={raw.strip()[:120]}")
        if ok:
            return norm or w

# ---------- run dirs + logging ----------
def make_run_dir(base: str = "responses") -> str:
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    d = os.path.join(base, run_id)
    os.makedirs(d, exist_ok=True)
    return d

def append_jsonl(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_summary(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
