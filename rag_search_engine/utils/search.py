import json, math
from typing import List, Set, Iterable, Iterator, Dict
from pathlib import Path
import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex


# ------------------------
# 1) load spaCy (tagger + attribute_ruler + lemmatizer kept; parser/ner disabled)
# ------------------------
def get_nlp():
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])


NLP = get_nlp()

# ------------------------
# 2) tokenizer customization
#    - split letters '.' letters via INFIX
#    - override token_match so alpha.dot.alpha isn't glued into one token
# ------------------------
# infix: split on '.' between letters
infixes = list(NLP.Defaults.infixes) + [r"(?<=[A-Za-z])\.(?=[A-Za-z])"]
infix_re = compile_infix_regex(infixes)

# default token_match glues domain-like tokens; relax it for alpha.dot.alpha(+)
orig_token_match = NLP.tokenizer.token_match
ALPHA_DOT_ALPHA_FULLMATCH = re.compile(r"^[A-Za-z]+(?:\.[A-Za-z]+)+$").fullmatch


def custom_token_match(text: str):
    # let infix split alpha.alpha(.alpha) tokens
    if ALPHA_DOT_ALPHA_FULLMATCH(text):
        return None
    return orig_token_match(text) if orig_token_match else None


# rebuild tokenizer
NLP.tokenizer = Tokenizer(
    NLP.vocab,
    rules=NLP.tokenizer.rules,
    prefix_search=NLP.tokenizer.prefix_search,
    suffix_search=NLP.tokenizer.suffix_search,
    infix_finditer=infix_re.finditer,
    token_match=custom_token_match,
)

# ------------------------
# 3) normalizers
#    INDEX: ordered tokens, keep stopwords (so phrase positions still work)
#    QUERY: set of tokens, drop stopwords for quick overlap checks
# ------------------------

# helpers for post-token fixes
INIT_DOTS = re.compile(r"^(?:[A-Za-z]\.){2,}[A-Za-z]\.?$")  # e.g., M.A.S.H.
ONLY_LETTERS_AND_DOTS = re.compile(r"^[A-Za-z.]+$").match
POSSESSIVE_FORMS = {"'s", "’s"}  # quick clitic filter


def _emit_pieces_from_dot_word(lemma: str) -> List[str]:
    """
    Handle tokens containing dots:
    - 'M.A.S.H.'      -> ['mash']
    - 'ryan.flanagan' -> ['ryan', 'flanagan']
    - 'co.uk'         -> ['co', 'uk']
    """
    # collapse initialisms like M.A.S.H. → mash
    if INIT_DOTS.match(lemma.upper()):
        return ["".join(ch for ch in lemma if ch.isalpha())]
    # otherwise split if it's letters and dots only
    if ONLY_LETTERS_AND_DOTS(lemma):
        return [p for p in lemma.split(".") if p]
    return [lemma]


def tokens_for_index(text: str, split_hyphens: bool = True) -> List[str]:
    """
    Single-text normalizer for indexing (ordered, keeps stopwords).
    - lowercase, lemma
    - drop spaces, punctuation, URLs/emails
    - split alpha.alpha, collapse initialisms
    - split hyphens optionally
    - drop possessive clitic "'s"
    """
    doc = NLP(text.lower())
    out: List[str] = []
    for t in doc:
        if t.is_space or t.is_punct or t.like_url or t.like_email:
            continue
        lem = t.lemma_.lower()

        # drop possessive clitic
        if lem in POSSESSIVE_FORMS or t.tag_ == "POS":
            continue

        # handle tokens with dots
        if "." in lem:
            out.extend(_emit_pieces_from_dot_word(lem))
            continue

        # hyphen handling
        if split_hyphens and "-" in lem and lem.replace("-", "").isalpha():
            out.extend(p for p in lem.split("-") if p)
            continue

        # keep alphabetic words and numbers; skip everything else
        if t.is_alpha:
            out.append(lem)
        elif t.like_num:
            out.append(t.text)  # keep raw numeric literal
        # else: drop symbols/mixed junk
    return out


def normalize_for_index(
    texts: Iterable[str],
    batch_size: int = 256,
    n_process: int = 1,
    *,
    split_hyphens: bool = True,
) -> Iterator[List[str]]:
    """
    Batched version for building an inverted index. WSL2 tip: keep n_process=1.
    """
    for doc in NLP.pipe(
        (s.lower() for s in texts), batch_size=batch_size, n_process=n_process
    ):
        out: List[str] = []
        for t in doc:
            if t.is_space or t.is_punct or t.like_url or t.like_email:
                continue
            lem = t.lemma_.lower()
            if lem in POSSESSIVE_FORMS or t.tag_ == "POS":
                continue
            if "." in lem:
                out.extend(_emit_pieces_from_dot_word(lem))
                continue
            if split_hyphens and "-" in lem and lem.replace("-", "").isalpha():
                out.extend(p for p in lem.split("-") if p)
                continue
            if t.is_alpha:
                out.append(lem)
            elif t.like_num:
                out.append(t.text)
        yield out


def normalize_for_query(text: str) -> list[str]:
    """
    Query-time normalizer: drop stopwords for cheap set-overlap, keep same dot/hyphen logic.
    """
    doc = NLP(text.lower())
    out: List[str] = []
    for t in doc:
        if t.is_space or t.is_punct or t.is_stop or t.like_url or t.like_email:
            continue
        lem = t.lemma_.lower()
        if lem in POSSESSIVE_FORMS or t.tag_ == "POS":
            continue
        if "." in lem:
            out.extend(_emit_pieces_from_dot_word(lem))
            continue
        if "-" in lem and lem.replace("-", "").isalpha():
            out.extend(p for p in lem.split("-") if p)
            continue
        if t.is_alpha:
            out.append(lem)
        elif t.like_num:
            out.append(t.text)
    return out


# ________________________________________________________________________________


# ________________________________________________________________________________
# ____________________Search Algorithm____________________________________________
# ________________________________________________________________________________
def basic_search(text: str) -> List[str]:
    """
    Case insensitivity: Convert all text to lowercase
        "The Matrix" becomes "the matrix"
    Remove punctuation: We don't care about periods, commas, etc
        "Hello, world!" becomes "hello world"
    Tokenization: Break text into individual words
        "the matrix" becomes ["the", "matrix"]
    Stop words: Remove common stop words that don't add much meaning
        ["the", "matrix"] becomes ["matrix"]
    Stemming: Keep only the stem of words
        ["running", "jumping"] becomes ["run", "jump"]
    """

    # define path to data
    DATA = Path(__file__).resolve().parents[2] / "data" / "movies.json"

    # use cache to speed up search
    global _PRECOMPUTED_TITLES

    # sanatize the query text
    sanitized_text = normalize_for_query(text)
    st = set(*sanitized_text)
    # sys.exit()
    # 1) load titles
    with DATA.open("r", encoding="utf-8") as file:
        data = json.load(fp=file)
        titles = [e["title"] for e in data["movies"]]

    # pipe over titles in batches; use all cores
    norm_sets = normalize_for_query(titles)
    _PRECOMPUTED_TITLES = list(zip(titles, norm_sets))

    # return if any part of sanitized text is present at all...
    def any_substring(qset: set[str], tset: set[str]) -> bool:
        # print(">>",qset, tset)
        return bool(qset & tset)

    # more permisive return statement because substrings can now match
    return [title for title, tset in _PRECOMPUTED_TITLES if any_substring(st, tset)]


# ________________________________________________________________________________

# ________________________________________________________________________________
# ____________________Search Algorithm____________________________________________
# ________________________________________________________________________________


def search_inverted_index(
    text: str,
    postings: Dict[str, Dict[int, List[int]]],
    docmap: Dict[int, Dict[str, str]],
) -> set[str]:
    """
    Case insensitivity: Convert all text to lowercase
        "The Matrix" becomes "the matrix"
    Remove punctuation: We don't care about periods, commas, etc
        "Hello, world!" becomes "hello world"
    Tokenization: Break text into individual words
        "the matrix" becomes ["the", "matrix"]
    Stop words: Remove common stop words that don't add much meaning
        ["the", "matrix"] becomes ["matrix"]
    Stemming: Keep only the stem of words
        ["running", "jumping"] becomes ["run", "jump"]
    """
    # sanatize the query text
    sanitized_text = normalize_for_query(text)
    first_five = []
    for word in sanitized_text:
        if postings.get(word):
            first_five += list(postings[word].keys())[:5]

    return set(first_five)


def calc_idf(
    text: str,
    postings: Dict[str, Dict[int, List[int]]],
    docmap: Dict[int, Dict[str, str]],
) -> float:
    # sanatize the query text
    sanitized_text = normalize_for_query(text)
    idf = math.log(
        (len(docmap.keys()) + 1) / (len(postings[sanitized_text[0]].keys()) + 1)
    )
    print(f"Inverse document frequency of {sanitized_text[0]}: {idf:.2f}")

    return idf


# ________________________________________________________________________________
