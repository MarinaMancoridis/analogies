"""
robust_word_sampler.py

Chooses a random word from the top 50k English words and returns:
(word, popularity_score, upos_tag, domain, abstraction_norm).

Process: uniformly sample from a cleaned 50k list (ASCII alphabetic only), then
(1) tag POS (NLTK Universal), (2) require a WordNet semantic domain (lexname)
consistent with that POS, and (3) require a Brysbaert (2014) concreteness score
to compute an abstraction score.

Datasets used:
- Word list & rank: wordfreq top_n_list("en", 50_000)
- POS tag: NLTK (averaged_perceptron* + Universal tagset)
- Semantic domain: Princeton WordNet (via NLTK)
- Abstraction: Brysbaert, Warriner & Kuperman (2014) concreteness (1–5)
"""

import csv
import random
from typing import Dict, Iterable, List, Optional, Set, Tuple

from wordfreq import top_n_list

# --- RNG + pool setup ---
RNG = random.Random()
ALL: List[str] = top_n_list("en", 50_000)  # most→least popular

def _is_simple_token(w: str) -> bool:
    """Return True if token is ASCII alphabetic (simple words only)."""
    return w.isascii() and w.isalpha()

# --- NLTK (POS + WordNet) ---
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn

def _ensure_nltk():
    # POS taggers (newer name first), universal tagset, and wordnet
    for res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
        try:
            nltk.data.find(f"taggers/{res}")
            break
        except LookupError:
            try:
                nltk.download(res, quiet=True)
                break
            except Exception:
                pass
    try:
        nltk.data.find("taggers/universal_tagset")
    except LookupError:
        try:
            nltk.download("universal_tagset", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
        except Exception:
            pass

_UPOS_TO_WNPOS = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
}

def _upos_with_nltk(word: str) -> Optional[str]:
    _ensure_nltk()
    try:
        # Universal tagset → e.g., NOUN, VERB, ADJ, ADV, ADP, DET, PRON, NUM, PRT, CONJ
        tag = pos_tag([word], tagset="universal")[0][1]
        return tag or None
    except Exception:
        return None

def _wn_lexnames_for(word: str, upos: str) -> Set[str]:
    """Return WordNet lexnames for `word`, filtered by the mapped WordNet POS when possible."""
    _ensure_nltk()
    pos = _UPOS_TO_WNPOS.get(upos)
    synsets = wn.synsets(word, pos=pos) if pos is not None else wn.synsets(word)
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
        word_keys = ["Word", "word", "lemma", "Item", "item"]
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
    """
    Convert Brysbaert concreteness (1=abstract … 5=concrete) to abstraction in [0,1]:
      abstraction = (conc - 1) / 4
      0.0 = most abstract, 1.0 = most concrete
    """
    c = min(5.0, max(1.0, conc))
    return (c - 1.0) / 4.0

# --- robust word sampler with POS + WordNet + Brysbaert ---
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
    Keep sampling until we can provide ALL features:
      - Universal POS tag (NLTK)
      - WordNet semantic domain (lexname) consistent with POS
      - Brysbaert-based abstraction score (derived from 1..5 concreteness)

    Returns:
      (word, popularity_score, upos_tag, domain, abstraction_norm)

    Popularity: rank-normalized to [0,1] (1 = most frequent, 0 = least).
    Abstraction: (concreteness-1)/4 → [0,1] (0 = most abstract, 1 = most concrete).
    """
    # Ensure Brysbaert norms available
    global _BRYS_NORMS
    if _BRYS_NORMS is None:
        if brys_path:
            load_brysbaert_norms(brys_path)
        else:
            raise RuntimeError(
                "Brysbaert norms not loaded. Call load_brysbaert_norms(path) once, "
                "or pass brys_path=... to robust_word_sampler."
            )

    r = rng or RNG
    lst = pool or ALL
    N = len(lst)
    if N == 0:
        raise RuntimeError("Empty pool: no words to sample from.")

    disallow = set()
    if not accept_determiners:
        disallow.add("DET")
    if not accept_pronouns:
        disallow.add("PRON")

    allow_set: Optional[Set[str]] = set(allowed_domains) if allowed_domains is not None else None

    while True:
        w = r.choice(lst).lower()
        if not _is_simple_token(w):
            continue

        # 1) POS
        upos = _upos_with_nltk(w)
        if upos is None or upos in disallow:
            continue

        # 2) WordNet domain(s) consistent with POS
        domains = _wn_lexnames_for(w, upos)
        if not domains:
            continue
        if allow_set is not None:
            dom_matches = domains & allow_set
            if not dom_matches:
                continue
            chosen_domain = r.choice(sorted(dom_matches))
        else:
            chosen_domain = r.choice(sorted(domains))

        # 3) Brysbaert concreteness (to compute abstraction)
        conc = _brys_concreteness(w)
        if conc is None:
            continue

        # Popularity score by rank: 1.0 at top, 0.0 at bottom
        try:
            idx = lst.index(w)
        except ValueError:
            idx = ALL.index(w)
            lst = ALL
            N = len(lst)

        popularity = 1.0 if N == 1 else 1.0 - (idx / (N - 1))
        abstraction = _normalized_abstraction_from_concreteness(conc)

        return w, float(popularity), upos, chosen_domain, float(abstraction)

# --- pretty printing ---
def _print_table(rows, col_widths=(13, 28, 28, 22)):
    """
    Print rows as aligned columns.
    rows: list of tuples -> (label, value, dataset, note)
    col_widths: widths for Label, Value, Dataset, Note
    """
    L, V, D, N = col_widths
    def fmt(a, w): return (a if len(a) <= w else a[: max(0, w - 1)] + "…").ljust(w)
    header = (
        fmt("Feature", L) + fmt("Value", V) + fmt("Dataset", D) + fmt("Note (3 words)", N)
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for lab, val, data, note in rows:
        print(fmt(lab, L) + fmt(val, V) + fmt(data, D) + fmt(note, N))

# --- example run (uses your local path) ---
if __name__ == "__main__":
    # Your note: Brysbaert file is beside the script
    BRYS_PATH = "./concreteness.txt"
    load_brysbaert_norms(BRYS_PATH)

    word, pop, upos, dom, abstr = robust_word_sampler(
        accept_determiners=False,
        accept_pronouns=False,
        allowed_domains=None
    )

    rows = [
        ("Word",        word,            "wordfreq top_n_list",        "simple ASCII token"),
        ("Popularity",  f"{pop:.4f}",    "wordfreq ranks",              "top one; bottom zero"),
        ("UPOS",        upos,            "NLTK Universal",              "grammar category tag"),
        ("Domain",      dom,             "Princeton WordNet",          "semantic domain lexname"),
        ("Abstraction", f"{abstr:.4f}",  "Brysbaert 2014",              "0 abstract; 1 concrete"),
    ]
    _print_table(rows)
