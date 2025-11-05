"""
robust_word_sampler.py

Samples two independent words (A and B) from the top 50k English words and, for each,
returns (word, popularity_score, upos_tag, domain, abstraction_norm). Also computes:

- Pair-level co-occurrence (C4): same-sentence NPMI mapped to [0,1] on the
  AllenAI C4 corpus (streamed).

Datasets used:
- Word list & rank: wordfreq top_n_list("en", 50_000)
- POS tag: NLTK (averaged_perceptron* + Universal tagset)
- Semantic domain: Princeton WordNet (via NLTK)
- Abstraction: Brysbaert, Warriner & Kuperman (2014) concreteness (1–5)
- Co-occurrence (C4): same-sentence NPMI on allenai/c4 (streamed)
"""

import csv
import math
import random
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from wordfreq import top_n_list
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn

# C4 streaming (requires: pip install 'datasets==3.*' xxhash)
# We keep the import optional and handle failures gracefully.
try:
    from datasets import load_dataset  # type: ignore
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

# --- RNG + pool setup ---
RNG = random.Random()
ALL: List[str] = top_n_list("en", 50_000)  # most→least popular

def _is_simple_token(w: str) -> bool:
    """True if token is ASCII alphabetic (simple words only)."""
    return w.isascii() and w.isalpha()

# --- NLTK setup (POS + WordNet) ---
def _ensure_nltk():
    for res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
        try:
            nltk.data.find(f"taggers/{res}")
            break
        except LookupError:
            nltk.download(res, quiet=True)
    for res in ("universal_tagset", "wordnet"):
        try:
            path = f"taggers/{res}" if res == "universal_tagset" else f"corpora/{res}"
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)

_UPOS_TO_WNPOS = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ":  wn.ADJ,
    "ADV":  wn.ADV,
}

def _upos_with_nltk(word: str) -> Optional[str]:
    _ensure_nltk()
    try:
        tag = pos_tag([word], tagset="universal")[0][1]
        return tag or None
    except Exception:
        return None

def _wn_lexnames_for(word: str, upos: str) -> Set[str]:
    _ensure_nltk()
    pos = _UPOS_TO_WNPOS.get(upos)
    synsets = wn.synsets(word, pos=pos) if pos else wn.synsets(word)
    return {syn.lexname() for syn in synsets}

# --- Brysbaert et al. (2014) concreteness norms ---
_BRYS_NORMS: Optional[Dict[str, float]] = None  # word(lower) -> concreteness (1..5)

def load_brysbaert_norms(path: str) -> None:
    """
    Load Brysbaert, Warriner & Kuperman (2014) concreteness norms from TSV/CSV.

    Column names (first match used):
      word:  'Word', 'word', 'lemma', 'Item', 'item'
      score: 'Conc.M', 'Conc.Mean', 'Concreteness', 'concreteness', 'conc', 'Mean'
    """
    global _BRYS_NORMS
    if _BRYS_NORMS is not None:
        return
    norms: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        raw = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(raw, delimiters=",\t;")
            delim = dialect.delimiter
        except Exception:
            delim = "\t"
        reader = csv.DictReader(f, delimiter=delim)
        word_keys  = ["Word", "word", "lemma", "Item", "item"]
        score_keys = ["Conc.M", "Conc.Mean", "Concreteness", "concreteness", "conc", "Mean"]
        for row in reader:
            w = next((row[k].strip().lower() for k in word_keys if k in row and row[k]), None)
            s = None
            for k in score_keys:
                if k in row and row[k]:
                    try:
                        s = float(row[k])
                    except ValueError:
                        s = None
                    break
            if w and s is not None and 1.0 <= s <= 5.0:
                norms[w] = s
    _BRYS_NORMS = norms

def _brys_concreteness(word: str) -> Optional[float]:
    if _BRYS_NORMS is None:
        return None
    return _BRYS_NORMS.get(word.lower())

def _normalized_abstraction_from_concreteness(conc: float) -> float:
    """Concreteness (1=abstract … 5=concrete) → abstraction in [0,1]."""
    c = min(5.0, max(1.0, conc))
    return (c - 1.0) / 4.0

# --- single-word sampler (POS + WordNet + Brysbaert abstraction) ---
def robust_word_sampler(
    *,
    rng: Optional[random.Random] = None,
    pool: Optional[List[str]] = None,
    accept_determiners: bool = False,
    accept_pronouns: bool = False,
    allowed_domains: Optional[Iterable[str]] = None,
    brys_path: Optional[str] = None,
) -> Tuple[str, float, str, str, float]:
    """
    Returns (word, popularity, upos, domain, abstraction).
    Keeps sampling until POS, WordNet domain, and Brysbaert score are all available.

    Process: uniformly sample from the cleaned 50k list (ASCII alphabetic only),
    tag with NLTK Universal POS, require a WordNet lexname consistent with POS,
    require a Brysbaert concreteness score, then map rank to popularity [0,1]
    and concreteness (1–5) to abstraction [0,1] via (conc-1)/4.
    """
    global _BRYS_NORMS
    if _BRYS_NORMS is None:
        if brys_path:
            load_brysbaert_norms(brys_path)
        else:
            raise RuntimeError("Brysbaert norms not loaded.")
    r = rng or RNG
    lst = pool or ALL
    N = len(lst)

    disallow: Set[str] = set()
    if not accept_determiners:
        disallow.add("DET")
    if not accept_pronouns:
        disallow.add("PRON")
    allow_set = set(allowed_domains) if allowed_domains else None

    while True:
        w = r.choice(lst).lower()
        if not _is_simple_token(w):
            continue
        upos = _upos_with_nltk(w)
        if upos is None or upos in disallow:
            continue
        domains = _wn_lexnames_for(w, upos)
        if not domains:
            continue
        if allow_set:
            dom_matches = domains & allow_set
            if not dom_matches:
                continue
            chosen_domain = r.choice(sorted(dom_matches))
        else:
            chosen_domain = r.choice(sorted(domains))
        conc = _brys_concreteness(w)
        if conc is None:
            continue
        idx = lst.index(w)
        popularity = 1.0 - (idx / (N - 1))
        abstraction = _normalized_abstraction_from_concreteness(conc)
        return w, float(popularity), upos, chosen_domain, float(abstraction)

# --- two-word sampler ---
def robust_word_sampler_pair(*, rng: Optional[random.Random] = None, **kw):
    r = rng or RNG
    a = robust_word_sampler(rng=r, **kw)
    b = robust_word_sampler(rng=r, **kw)
    return a, b

# --- C4 sentence-level co-occurrence (NPMI -> [0,1]) ---
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE    = re.compile(r"[A-Za-z]+")

def _sent_tokenize(text: str) -> List[str]:
    """Lightweight sentence splitter; replace with nltk.sent_tokenize if desired."""
    return _SENT_SPLIT.split(text)

def _lower_ascii_words(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(s)]

def _npmi_to_unit_interval(p_ab: float, p_a: float, p_b: float) -> float:
    """Map NPMI in [-1,1] to [0,1] via (npmi+1)/2 with guards."""
    if p_ab <= 0.0 or p_a <= 0.0 or p_b <= 0.0:
        return 0.0
    denom = -math.log(p_ab)
    if denom <= 0.0:
        return 0.0
    npmi = math.log(p_ab / (p_a * p_b)) / denom
    score = (npmi + 1.0) / 2.0
    return max(0.0, min(1.0, score))

def c4_sentence_cooccurrence_score(
    word_a: str,
    word_b: str,
    *,
    lang: str = "en",
    sample_prob: float = 0.01,
    max_docs: Optional[int] = None,
) -> float:
    """
    Continuous 0–1 co-occurrence via SAME-SENTENCE events on AllenAI C4:
      p_ab = fraction of sentences containing both terms
      p_a, p_b = fractions with each term
      score = NPMI mapped to [0,1] via (NPMI+1)/2

    Notes:
    - Uses streaming with trust_remote_code=True to avoid interactive prompts.
    - If `datasets` is missing or any loader/runtime error occurs (including local torch issues),
      this returns 0.0 so the rest of the script can proceed.
    """
    if not _HAS_DATASETS:
        return 0.0

    # C4 configs are language-specific; English is "en".
    try:
        ds = load_dataset("allenai/c4", lang, split="train", streaming=True, trust_remote_code=True)  # type: ignore
    except Exception:
        return 0.0

    rng = random.Random(0)
    a = word_a.lower()
    b = word_b.lower()

    sent_total = 0
    sent_with_a = 0
    sent_with_b = 0
    sent_with_ab = 0
    seen_docs = 0

    try:
        for ex in ds:
            if max_docs is not None and seen_docs >= max_docs:
                break
            if rng.random() > sample_prob:
                continue
            text = (ex.get("text") or "") if isinstance(ex, dict) else ""
            for sent in _sent_tokenize(text):
                toks = _lower_ascii_words(sent)
                if not toks:
                    continue
                has_a = a in toks
                has_b = b in toks
                sent_total += 1
                if has_a:
                    sent_with_a += 1
                if has_b:
                    sent_with_b += 1
                if has_a and has_b:
                    sent_with_ab += 1
            seen_docs += 1
    except Exception:
        return 0.0

    if sent_total == 0:
        return 0.0
    p_ab = sent_with_ab / sent_total
    p_a  = sent_with_a  / sent_total
    p_b  = sent_with_b  / sent_total
    return _npmi_to_unit_interval(p_ab, p_a, p_b)

# --- clean table output ---
def _print_table(title: str, rows, col_widths=(13, 28, 28, 22)):
    """Print rows as aligned columns for one item, with a custom title line."""
    print(title)
    L, V, D, N = col_widths
    def fmt(a, w): return (a if len(a) <= w else a[: w - 1] + '…').ljust(w)
    header = fmt("Feature", L) + fmt("Value", V) + fmt("Dataset", D) + fmt("Note", N)
    sep = "-" * len(header)
    print(header)
    print(sep)
    for lab, val, data, note in rows:
        print(fmt(lab, L) + fmt(val, V) + fmt(data, D) + fmt(note, N))
    print()

# --- example run ---
if __name__ == "__main__":
    BRYS_PATH = "./concreteness.txt"  # Brysbaert file in same directory
    load_brysbaert_norms(BRYS_PATH)

    # Two independent samples A and B
    (a_word, a_pop, a_upos, a_dom, a_abstr), (b_word, b_pop, b_upos, b_dom, b_abstr) = robust_word_sampler_pair(
        accept_determiners=False,
        accept_pronouns=False,
        allowed_domains=None
    )

    # Per-word tables with requested headers
    rows_A = [
        ("Word",        a_word,           "wordfreq top_n_list", "simple ASCII token"),
        ("Popularity",  f"{a_pop:.4f}",   "wordfreq ranks",       "top one; bottom zero"),
        ("UPOS",        a_upos,           "NLTK Universal",       "grammar category tag"),
        ("Domain",      a_dom,            "Princeton WordNet",    "semantic domain lexname"),
        ("Abstraction", f"{a_abstr:.4f}", "Brysbaert 2014",       "0 abstract; 1 concrete"),
    ]
    rows_B = [
        ("Word",        b_word,           "wordfreq top_n_list", "simple ASCII token"),
        ("Popularity",  f"{b_pop:.4f}",   "wordfreq ranks",       "top one; bottom zero"),
        ("UPOS",        b_upos,           "NLTK Universal",       "grammar category tag"),
        ("Domain",      b_dom,            "Princeton WordNet",    "semantic domain lexname"),
        ("Abstraction", f"{b_abstr:.4f}", "Brysbaert 2014",       "0 abstract; 1 concrete"),
    ]

    _print_table(f"A Word: {a_word}", rows_A)
    _print_table(f"B Word: {b_word}", rows_B)

    # Pair-level co-occurrence (C4: same-sentence NPMI → [0,1])
    c4_cooc = c4_sentence_cooccurrence_score(
        a_word, b_word,
        lang="en",
        sample_prob=0.02,    # increase for more stable estimates
        max_docs=10000       # optional cap
    )
    print("=== Pair A–B ===")
    print(f"C4 Co-occurrence : {c4_cooc:.4f}   Dataset: allenai/c4 (same-sentence NPMI)   Note: 0 far-below-chance; 1 strong co-sentence")
