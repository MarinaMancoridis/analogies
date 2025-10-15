import os, json, random, re, datetime, uuid
from typing import Optional, Tuple, Callable
from wordfreq import top_n_list
from analogies.utils import generate_inference
import random
from typing import List
import nltk # used for identity — part of speech
import stanza # used for identity — part of speech
from functools import lru_cache
import csv


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

# helper to choose a picker for how to sample words
def _get_picker(mode: str):
    if mode == "popular":
        return sample_popular_word
    elif mode == "rare":
        return sample_rare_word
    elif mode.startswith("pos:"):
        _, pos = mode.split(":", 1)
        return lambda: sample_word_by_pos(pos)  # uses NLTK tagging under the hood
    elif mode == "cooccurring":
        # JOINT picker: returns a (A, C) pair
        def _joint():
            return sample_cooccurring_words()
        _joint._joint = "cooccurring"   # tag so run_trial knows this is joint
        return _joint
    elif mode == "noncooccurring":
        # JOINT picker: returns a (A, C) pair
        def _joint():
            return sample_noncooccurring_words()
        _joint._joint = "noncooccurring"
        return _joint
    elif mode == "ac:abstract":
        return lambda: sample_abstract_brys(picker=sample_word)
    elif mode == "ac:concrete":
        return lambda: sample_concrete_brys(picker=sample_word)
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

# ---- concreteness sampling ----
_BRYS_NORMS = None  # word(lower) -> float (1..5)

def load_brysbaert_norms(path: str) -> None:
    """Load once; supports TSV or CSV with columns like 'Word'/'word' and 'Conc.M'/'concreteness'."""
    global _BRYS_NORMS
    if _BRYS_NORMS is not None:
        return
    norms = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        # Try to auto-detect delimiter; fall back to tab.
        raw = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(raw, delimiters=",\t;")
            delim = dialect.delimiter
        except Exception:
            delim = "\t"
        reader = csv.DictReader(f, delimiter=delim)
        # Heuristic column names used in various releases
        word_keys = ["Word", "word", "lemma", "Item", "item"]
        score_keys = ["Conc.M", "Conc.Mean", "Concreteness", "concreteness", "conc", "Mean"]
        for row in reader:
            w = None
            for k in word_keys:
                if k in row and row[k]:
                    w = row[k].strip().lower()
                    break
            s = None
            for k in score_keys:
                if k in row and row[k]:
                    try:
                        s = float(row[k])
                    except ValueError:
                        s = None
                    break
            if w and s is not None:
                norms[w] = s
    _BRYS_NORMS = norms

def _brys_score(word: str) -> Optional[float]:
    if _BRYS_NORMS is None:
        raise RuntimeError("Brysbaert norms not loaded. Call load_brysbaert_norms(path) once.")
    return _BRYS_NORMS.get(word.lower())

def ac_label_brys(word: str, *, low: float = 1, high: float = 5) -> str:
    """
    Map Brysbaert concreteness (1..5) to a label with a gap:
      v <= low   -> 'abstract'
      v >= high  -> 'concrete'
      else       -> 'neither'   (keeps borderline items out)
    """
    v = _brys_score(word)
    if v is None:
        return "neither"
    if v <= low:
        return "abstract"
    if v >= high:
        return "concrete"
    return "neither"

def sample_by_concreteness_brys(
    kind: str,
    *,
    picker: Optional[Callable[[], str]] = None,
    max_tries: int = 2000,
    low: float = 2.6,
    high: float = 3.4,
) -> str:
    """
    Sample until the Brysbaert label (with gap) matches.
    Increase the gap (e.g., low=2.0, high=4.0) to make the divide even larger.
    """
    if picker is None:
        picker = sample_word
    kind = kind.strip().lower()
    if kind not in ("abstract", "concrete"):
        raise ValueError("kind must be 'abstract' or 'concrete'")
    for _ in range(max_tries):
        w = picker()
        if not _is_simple_token(w):
            continue
        if ac_label_brys(w, low=low, high=high) == kind:
            return w
    raise RuntimeError(f"Could not sample a {kind} word via Brysbaert in {max_tries} tries.")

def sample_abstract_brys(*, picker: Optional[Callable[[], str]] = None, **kw) -> str:
    return sample_by_concreteness_brys("abstract", picker=picker, **kw)

def sample_concrete_brys(*, picker: Optional[Callable[[], str]] = None, **kw) -> str:
    return sample_by_concreteness_brys("concrete", picker=picker, **kw)



# ---- co-occurrence sampling ----
def _ensure_cooccurrence_index():
    """
    Build once: a list of strong co-occurring bigrams and a set of all observed bigrams
    from the NLTK Brown corpus.
    """
    global _COOC_READY, _COOC_LIST, _OBS_BIGRAMS, _COOC_VOCAB
    try:
        _COOC_READY
    except NameError:
        _COOC_READY = False

    if _COOC_READY:
        return

    import nltk
    from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)

    from nltk.corpus import brown

    tokens = [w.lower() for w in brown.words() if _is_simple_token(w)]

    measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)  # adjacent bigrams
    finder.apply_freq_filter(3)  # drop ultra-rare bigrams

    # Top N strongest collocations by PMI
    _COOC_LIST = finder.nbest(measures.pmi, 5000)

    # All seen bigrams (for fast negative checks)
    _OBS_BIGRAMS = set(finder.ngram_fd.keys())

    # Negative-sampling vocab: overlap with your ALL to keep words familiar
    tokset = set(tokens)
    _COOC_VOCAB = [w for w in ALL if w in tokset][:20_000] or list(tokset)

    _COOC_READY = True


def sample_cooccurring_words() -> Tuple[str, str]:
    """Return a frequently co-occurring word pair (adjacent bigram in Brown)."""
    _ensure_cooccurrence_index()
    return RNG.choice(_COOC_LIST)


def sample_noncooccurring_words(max_tries: int = 10_000) -> Tuple[str, str]:
    """
    Return two words that do NOT appear as an adjacent bigram in the Brown corpus
    (in either order).
    """
    _ensure_cooccurrence_index()
    for _ in range(max_tries):
        w1 = _pick(_COOC_VOCAB)
        w2 = _pick(_COOC_VOCAB)
        if w1 == w2:
            continue
        if (w1, w2) not in _OBS_BIGRAMS and (w2, w1) not in _OBS_BIGRAMS:
            return (w1, w2)
    return (RNG.choice(_COOC_VOCAB), RNG.choice(_COOC_VOCAB))


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


# ------------- TESTING CO-OCCURRENCE / NON-CO-OCCURRENCE word pairs -------------
# if __name__ == "__main__":
#     print("=== Co-occurring word pairs ===")
#     for i in range(10):
#         a, c = sample_cooccurring_words()
#         print(f"{i+1:2d}. {a} — {c}")

#     print("\n=== Non-co-occurring word pairs ===")
#     for i in range(10):
#         a, c = sample_noncooccurring_words()
#         print(f"{i+1:2d}. {a} — {c}")


# ------------- TESTING CONCRETE / ABSTRACT word pairs -------------
# if __name__ == "__main__":
#     import os, sys

#     # ---- Set your path to the Brysbaert file here ----
#     BRYS_PATH = "./analogies/concreteness.txt"
#     load_brysbaert_norms(BRYS_PATH)

#     def _unique_samples(fn, n=10, max_tries=5000):
#         out, seen = [], set()
#         tries = 0
#         while len(out) < n and tries < max_tries:
#             w = fn()
#             tries += 1
#             if w and w not in seen:
#                 out.append(w)
#                 seen.add(w)
#         return out

#     # Sample using your existing picker; swap picker=sample_popular_word if you prefer
#     abs_words = _unique_samples(lambda: sample_abstract_brys(picker=sample_word), n=10)
#     conc_words = _unique_samples(lambda: sample_concrete_brys(picker=sample_word), n=10)

#     print("=== Abstract (10) ===")
#     for i, w in enumerate(abs_words, 1):
#         print(f"{i:2d}. {w}")

#     print("\n=== Concrete (10) ===")
#     for i, w in enumerate(conc_words, 1):
#         print(f"{i:2d}. {w}")