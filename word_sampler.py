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
import os
from typing import Dict, Iterable, List, Optional, Set, Tuple

from wordfreq import top_n_list
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn


from analogies.utils import generate_inference
from analogies.common import clean_answer
# from analogies.analogy_types.identity import _prompt_english as identity_prompt
from analogies.analogy_types.identity import _prompt_one_shot as identity_prompt
# from analogies.analogy_types.identity import _prompt_few_shot as identity_prompt



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

# ---- polysemy-based WordNet lexnames ----
# --- Polysemy (WordNet-based) ---

def _polysemy_count_wordnet(word: str) -> int:
    """
    Return the number of WordNet senses (synsets) for this word,
    across all parts of speech.
    """
    _ensure_nltk()
    return len(wn.synsets(word.lower()))


def _normalized_polysemy_from_count(n_senses: int) -> Optional[float]:
    """
    Map integer number of senses to a score in [0,1] with 0.1 steps:

        1 sense -> 0.0
        2 senses -> 0.1
        3 senses -> 0.2
        ...
        11+ senses -> 1.0

    Anything with no senses (n_senses <= 0) returns None.
    """
    if n_senses <= 0:
        return None

    score = 0.1 * (n_senses - 1)   # 1 -> 0.0, 2 -> 0.1, ...
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return float(score)


def _polysemy_score(word: str) -> Optional[float]:
    """
    Compute number of WordNet senses for `word` and convert to [0,1] score
    where 0 = 1 sense, 0.1 = 2 senses, ..., 1.0 = 11+ senses.
    """
    n = _polysemy_count_wordnet(word)
    return _normalized_polysemy_from_count(n)

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
) -> Tuple[str, float, str, str, float, float]:
    """
    Returns (word, popularity, upos, domain, abstraction, polysemy).

    Keeps sampling until POS, WordNet domain, Brysbaert score, and
    WordNet polysemy score are all available.

    Process: uniformly sample from the cleaned 50k list (ASCII alphabetic only),
    tag with NLTK Universal POS, require a WordNet lexname consistent with POS,
    require a Brysbaert concreteness score, then map:
      - rank to popularity [0,1]
      - concreteness (1–5) to abstraction [0,1] via (conc-1)/4
      - WordNet num_senses to polysemy [0,1] via linear scaling
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

        poly = _polysemy_score(w)
        if poly is None:
            continue

        idx = lst.index(w)
        popularity = 1.0 - (idx / (N - 1))
        abstraction = _normalized_abstraction_from_concreteness(conc)

        # Now include poly in the return tuple
        return w, float(popularity), upos, chosen_domain, float(abstraction), float(poly)

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
# python -m analogies.word_sampler
if __name__ == "__main__":
    # Path relative to this file's directory
    here = os.path.dirname(__file__)
    BRYS_PATH = os.path.join(here, "concreteness.txt")
    load_brysbaert_norms(BRYS_PATH)

    MODEL = "gpt-5"      # or whatever model name you want
    N_TRIALS = 100

    # CSV output path
    # csv_path = os.path.join(here, "robust_identity_runs_english_prompt.csv")
    csv_path = os.path.join(here, "robust_identity_runs_one_shot_prompt.csv")
    # csv_path = os.path.join(here, "robust_identity_runs_few_shot_prompt.csv")

    # Open in append mode and write header only if file is new/empty
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writerow([
                "trial_id",
                "model",
                # words
                "A_word",
                "B_word",
                # popularity
                "A_popularity",
                "B_popularity",
                # POS
                "A_upos",
                "B_upos",
                # domains
                "A_domain",
                "B_domain",
                # abstraction
                "A_abstraction",
                "B_abstraction",
                # polysemy
                "A_polysemy",
                "B_polysemy",
                # identity eval
                "identity_success",
                "prompt",
                "raw_response",
                "parsed_answer",
            ])

        hits = 0

        for trial_id in range(1, N_TRIALS + 1):
            # Sample two independent words with all semantic features
            (a_word, a_pop, a_upos, a_dom, a_abstr, a_poly), \
            (b_word, b_pop, b_upos, b_dom, b_abstr, b_poly) = robust_word_sampler_pair(
                accept_determiners=False,
                accept_pronouns=False,
                allowed_domains=None,
            )

            a_len = len(a_word)
            b_len = len(b_word)

            # Identity prompt: same as your identity test
            #   "Complete the analogy. Reply ONLY as: ANSWER: <word>\n\nA : A :: C : ____"
            prompt = identity_prompt(a_word, b_word)
            raw_response = generate_inference(prompt, MODEL)
            answer = clean_answer(raw_response)
            identity_success = (answer == b_word.lower())
            hits += int(identity_success)

            # ---- Print the tables for this pair ----
            rows_A = [
                ("Word",        a_word,           "wordfreq top_n_list", "simple ASCII token"),
                ("Popularity",  f"{a_pop:.4f}",   "wordfreq ranks",       "top one; bottom zero"),
                ("UPOS",        a_upos,           "NLTK Universal",       "grammar category tag"),
                ("Domain",      a_dom,            "Princeton WordNet",    "semantic domain lexname"),
                ("Abstraction", f"{a_abstr:.4f}", "Brysbaert 2014",       "0 abstract; 1 concrete"),
                ("Polysemy",    f"{a_poly:.4f}",  "WordNet synsets",      "0 = 1 sense; 1 = many"),
                ("IdentityHit", str(int(identity_success)),
                                "LLM identity",    "1 = success; 0 = failure"),
            ]
            rows_B = [
                ("Word",        b_word,           "wordfreq top_n_list", "simple ASCII token"),
                ("Popularity",  f"{b_pop:.4f}",   "wordfreq ranks",       "top one; bottom zero"),
                ("UPOS",        b_upos,           "NLTK Universal",       "grammar category tag"),
                ("Domain",      b_dom,            "Princeton WordNet",    "semantic domain lexname"),
                ("Abstraction", f"{b_abstr:.4f}", "Brysbaert 2014",       "0 abstract; 1 concrete"),
                ("Polysemy",    f"{b_poly:.4f}",  "WordNet synsets",      "0 = 1 sense; 1 = many"),
                ("IdentityHit", str(int(identity_success)),
                                "LLM identity",    "1 = success; 0 = failure"),
            ]

            print(f"\n===== Trial {trial_id}/{N_TRIALS} =====")
            _print_table(f"A Word: {a_word}", rows_A)
            _print_table(f"B Word: {b_word}", rows_B)

            # Also log a concise line to stdout
            print(
                f"[summary] A={a_word!r}, B={b_word!r}, "
                f"answer={answer!r}, success={identity_success}"
            )

            # ---- Write one CSV row per pair ----
            writer.writerow([
                trial_id,
                MODEL,
                a_word,
                b_word,
                a_len,
                b_len,
                f"{a_pop:.6f}",
                f"{b_pop:.6f}",
                a_upos,
                b_upos,
                a_dom,
                b_dom,
                f"{a_abstr:.6f}",
                f"{b_abstr:.6f}",
                f"{a_poly:.6f}",
                f"{b_poly:.6f}",
                int(identity_success),
                prompt,
                raw_response,
                answer,
            ])

    accuracy = hits / N_TRIALS if N_TRIALS else 0.0
    print(f"\nFinished {N_TRIALS} identity trials.")
    print(f"Accuracy: {hits}/{N_TRIALS} = {accuracy:.3f}")
    print(f"Results written to: {csv_path}")
