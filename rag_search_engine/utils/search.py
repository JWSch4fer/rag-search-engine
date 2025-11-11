import json
from typing import List, Set, Iterable, Iterator
from pathlib import Path
import sys


# avoid redownloading this package
# Keep tagger+lemmatizer for good lemmas; disable parser/ner.
def get_nlp():
    import spacy

    try:
        return spacy.load(
            "en_core_web_sm", disable=["parser", "ner"]
        )
    except OSError:
        # Not installed → download once, then load
        from spacy.cli.download import download

        download("en_core_web_sm")
        return spacy.load(
            "en_core_web_sm", disable=["parser", "ner"]
        )


# predifine for spacy
NLP = get_nlp()


# ________________________________________________________________________________
# ____________________Normalizer__________________________________________________
# ________________________________________________________________________________
def normalize_for_index(
    texts: Iterable[str],
    batch_size: int = 256,
    n_process: int = 1,  #WSL2 doesn't play nice with more processes...
) -> Iterator[List[str]]:
    """
    Yield token lists for each text using spaCy.pipe (fast, parallel).
    """
    lowered = (s.lower() for s in texts)
    for doc in NLP.pipe(lowered, batch_size=batch_size, n_process=n_process):
        yield [t.lemma_.lower() for t in doc if not t.is_space and not t.is_punct]


def normalize_for_query(
    texts: Iterable[str],
    batch_size: int = 256,
    n_process: int = 1,  #WSL2 doesn't play nice with more processes...
) -> Iterator[Set[str]]:
    """
    return set
    """
    lowered = (s.lower() for s in texts)
    for doc in NLP.pipe(lowered, batch_size=batch_size, n_process=n_process):
        yield {
            t.lemma_.lower()
            for t in doc
            if not t.is_space and not t.is_punct and not t.is_stop
        }


# ________________________________________________________________________________

# ________________________________________________________________________________
# ____________________Search Algorithm____________________________________________
# ________________________________________________________________________________
_PRECOMPUTED_TITLES = None


def basic_search(text: str | List[str]) -> List[str]:
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
    # ensure title is the proper type
    match text:
        case str():
            data = [text]
        case list():
            data = text
        case _:
            raise TypeError("text is not the correct type. Expected str or list[str]")

    # define path to data
    DATA = Path(__file__).resolve().parents[2] / "data" / "movies.json"

    # use cache to speed up search
    global _PRECOMPUTED_TITLES

    # sanatize the query text
    sanitized_text = normalize_for_query(data)
    st = set(*sanitized_text)
    # sys.exit()
    # 1) load titles
    with DATA.open("r", encoding="utf-8") as file:
        data = json.load(fp=file)
        titles = [e["title"] for e in data["movies"]]

    # 2) precompute title normalization once per process
    if _PRECOMPUTED_TITLES is None:
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
