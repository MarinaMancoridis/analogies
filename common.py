import os, json, random, re, datetime, uuid
from typing import Optional, Tuple, Callable
from wordfreq import top_n_list
from analogies.utils import generate_inference
import random
from typing import List
import nltk # used for identity — part of speech
import stanza # used for identity — part of speech
from functools import lru_cache


RNG = random.Random()
ALL: List[str] = top_n_list("en", 50_000)  # master list, most→least popular
WORDS = top_n_list("en", 50000)

# ---------- POS tagging ----------
_UNIVERSAL_POS = {"NOUN", "VERB", "ADJ", "ADV", "ADP"} # five popular POS tags- noun, verb, adjective, adverb, adposition
_POS_NORMALIZE = {
    "noun": "NOUN", "n": "NOUN",
    "verb": "VERB", "v": "VERB",
    "adjective": "ADJ", "adj": "ADJ",
    "adverb": "ADV", "adv": "ADV",
}
_STANZA_PIPELINE = None
def _ensure_stanza():
    global _STANZA_PIPELINE
    if _STANZA_PIPELINE is not None:
        return
    try:
        # Fast, minimal pipeline for tokenized input (we pass a single token)
        _STANZA_PIPELINE = stanza.Pipeline(
            lang="en",
            processors="tokenize,pos",
            tokenize_pretokenized=True,
            verbose=False
        )
    except Exception:
        # If models are missing, attempt to download once, then retry
        try:
            stanza.download("en", verbose=False)
            _STANZA_PIPELINE = stanza.Pipeline(
                lang="en",
                processors="tokenize,pos",
                tokenize_pretokenized=True,
                verbose=False
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize Stanza. Try running:\n"
                "    pip install stanza\n"
                "    python -c \"import stanza; stanza.download('en')\"\n"
                f"Inner error: {e}"
            )
# The averaged_perceptron_tagger_eng is a pre-trained Part-of-Speech (POS) tagger model for the English language within the Natural Language Toolkit (NLTK) library.
def _ensure_nltk_tagger():
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            # Try downloading the newer first; if it fails, fall back.
            try:
                nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            except Exception:
                pass
            try:
                nltk.download("averaged_perceptron_tagger", quiet=True)
            except Exception:
                pass
def _ensure_nltk_resources():
    for res in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"taggers/{res}")
            break
        except LookupError:
            try:
                nltk.download(res, quiet=True)
                break
            except Exception:
                pass
    # Universal tagset mapping
    try:
        nltk.data.find("taggers/universal_tagset")
    except LookupError:
        try:
            nltk.download("universal_tagset", quiet=True)
        except Exception:
            # if download fails (e.g., offline), we'll use a PTB->coarse fallback
            pass

def _stanza_pos(word: str) -> Tuple[str, str]:
    """
    Returns (xpos, upos) for a single token using Stanza.
    - xpos: Penn Treebank–like fine tag (e.g., 'NN', 'JJ')
    - upos: Universal POS tag (e.g., 'NOUN', 'ADJ')
    """
    _ensure_stanza()
    doc = _STANZA_PIPELINE([[word]])  # pretokenized: a single sentence with one token
    tok = doc.sentences[0].words[0]
    return tok.xpos, tok.upos

@lru_cache(maxsize=100_000)
def _tag_universal(word: str) -> str:
    # _ensure_nltk_tagger()
    # _ensure_nltk_resources()
    # # Tag a single token; universal tagset gives coarse tags.
    # tag = nltk.pos_tag([word], tagset="universal")[0][1]
    # return tag
    _, upos = _stanza_pos(word)
    return upos

def _normalize_pos_name(pos: str) -> str:
    key = (pos or "").strip().lower()
    if key not in _POS_NORMALIZE:
        raise ValueError(
            f"Unknown POS '{pos}'. Choose one of: noun, verb, adjective, adverb"
        )
    return _POS_NORMALIZE[key]


#---------- word sampling ----------
def _pick(words: List[str]) -> str:
    while True:
        w = RNG.choice(words).lower()
        if _is_simple_token(w):
            return w

def sample_random_word() -> str:
    """Uniform over the whole list (ignores popularity)."""
    return _pick(ALL)

def sample_popular_word() -> str:
    """Uniform from the top x most popular words."""
    x = 1000
    if x <= 0: raise ValueError("x must be positive")
    return _pick(ALL[:min(x, len(ALL))])

def sample_rare_word() -> str:
    """Uniform from the bottom x (rarest) words."""
    x = 1000
    if x <= 0: raise ValueError("x must be positive")
    x = min(x, len(ALL))
    return _pick(ALL[-x:])

# helper to choose a frequency-based picker
def _get_picker(mode: str):
    if mode == "popular":
        return sample_popular_word
    elif mode == "rare":
        return sample_rare_word
    elif mode.startswith("pos:"):
        _, pos = mode.split(":", 1)
        return lambda: sample_word_by_pos(pos)  # uses NLTK tagging under the hood
    else:
        raise ValueError(f"unknown mode: {mode}")

def sample_word_by_pos(
    pos: str,
    *,
    pool: List[str] = ALL,
    max_tries: int = 20000
) -> str:
    """
    Sample a word from `pool` whose Universal POS tag (Stanza) matches `pos`.
    Supported: noun, verb, adjective, adverb.
    """
    target = _normalize_pos_name(pos)
    tries = 0
    while tries < max_tries:
        print(f"Sampling word by POS: {pos}")
        print(f"Tries: {tries}")
        tries += 1

        w = _pick(pool)  # respects _is_simple_token
        print(f"Picked word: {w}")

        xpos, upos = _stanza_pos(w)
        print(f"Tag (XPOS): {xpos}  Tag (UPOS): {upos}")

        if upos == target:
            print(f"Found word with POS={target}")
            return w

    raise RuntimeError(f"Could not find a word with POS={target} after {max_tries} tries.")

def sample_noun() -> str: return sample_word_by_pos("noun")
def sample_verb() -> str: return sample_word_by_pos("verb")
def sample_adjective() -> str: return sample_word_by_pos("adjective")
def sample_adverb() -> str: return sample_word_by_pos("adverb")


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

def sample_concept(model: str, *, picker: Callable[[], str], verbose: bool = True) -> str:
    # Retry until we get one the LLM says "relies on concept" (Yes)
    tries = 0
    while True:
        tries += 1
        w = picker()  # <- was: sample_word()
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
