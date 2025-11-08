#!/usr/bin/env python3

import argparse
import json
from typing import List
from pathlib import Path
import string


# avoid redownloading this package
def get_nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Not installed → download once, then load
        from spacy.cli.download import download

        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


ROOT = Path(__file__).resolve().parents[1]

_PRECOMPUTED_TITLES = None


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
    DATA = ROOT / "data" / "movies.json"

    # define punctuation to keep/remove
    KEEP = {"'", "-"}
    DROP = "".join(ch for ch in string.punctuation if ch not in KEEP)
    TRANS_SEL = str.maketrans("", "", DROP)

    # predifine spacy function
    NLP = get_nlp()

    # use cache to speed up search
    global _PRECOMPUTED_TITLES

    def sanitize_doc(doc) -> set[str]:
        return {
            t.lemma_.lower()
            for t in doc
            if not t.is_stop and not t.is_punct and not t.is_space
        }

    def sanitize_text(text: str) -> set[str]:
        pre = text.lower().translate(TRANS_SEL)
        return sanitize_doc(NLP(pre))

    # sanatize the query text
    sanitized_text = sanitize_text(text)

    # 1) load titles
    with DATA.open("r", encoding="utf-8") as file:
        data = json.load(fp=file)
        titles = [e["title"] for e in data["movies"]]

    # 2) precompute title normalization once per process
    if _PRECOMPUTED_TITLES is None:
        # pipe over titles in batches; use all cores
        pre_iter = (t.lower().translate(TRANS_SEL) for t in titles)
        docs = NLP.pipe(pre_iter, batch_size=256, n_process=-1)
        norm_sets = [sanitize_doc(d) for d in docs]
        _PRECOMPUTED_TITLES = list(zip(titles, norm_sets))

    # if there is any overlap we return true
    # return [title for title, tset in _PRECOMPUTED_TITLES if sanitized_text & tset]

    # return if any part of sanitized text is present at all...
    def any_substring(qset: set[str], tset: set[str]) -> bool:
        return any(q and (q in t) for q in qset for t in tset)
    # more permisive return statement because substrings can now match
    return [title for title, tset in _PRECOMPUTED_TITLES if any_substring(sanitized_text, tset)]

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25"
    )  # args.command == 'search'
    search_parser.add_argument(
        "query", type=str, help="Search query"
    )  # args.query == 'query'

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = basic_search(args.query.lower())
            for idx, r in enumerate(results):
                print(f"{idx}. {r}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
