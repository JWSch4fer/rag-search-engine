#!/usr/bin/env python3

import argparse
import json
from typing import List
from pathlib import Path
import string

# one time download
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def remove_stops(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.lower() not in stops]


ROOT = Path(__file__).resolve().parents[1]  # adapt to your tree
FILE = ROOT / "resources" / "index.faiss"


def basic_search(text: str) -> List[str]:
    """
    Case insensitivity: Convert all text to lowercase
        "The Matrix" becomes "the matrix"
        "HE IS HERE" becomes "he is here"
    Remove punctuation: We don't care about periods, commas, etc
        "Hello, world!" becomes "hello world"
        "sci-fi" becomes "scifi"
    Tokenization: Break text into individual words
        "the matrix" becomes ["the", "matrix"]
        "hello world" becomes ["hello", "world"]
    Stop words: Remove common stop words that don't add much meaning
        ["the", "matrix"] becomes ["matrix"]
        ["a", "puppy"] becomes ["puppy"]
    Stemming: Keep only the stem of words
        ["running", "jumping"] becomes ["run", "jump"]
        ["watching", "windmills"] becomes ["watch", "windmill"]
    """
    # define path to data
    DATA = ROOT / "data" / "movies.json"

    # define punctuation to keep/remove
    KEEP = {"'", "-"}
    DROP = "".join(ch for ch in string.punctuation if ch not in KEEP)
    TRANS_SEL = str.maketrans("", "", DROP)

    # define stopword dictionary and stemming
    SNOW = SnowballStemmer(language="english")
    STOPWORDS = set(stopwords.words("english"))
    
    def sanitize(t: str) -> set[str]:
        """
         prep the text
         case insensitivity
         remove punctuation
         stopwords + stemming
        """
        t = t.lower().translate(TRANS_SEL)
        return set(
                    [SNOW.stem(t) for t in t.split(" ") if t not in STOPWORDS]
                )
    
    sanitized_text = sanitize(text)
    with DATA.open("r", encoding="utf-8") as file:
        data = json.load(fp=file)
        results = []
        for entry in data["movies"]:
            sanitized_title = sanitize(entry["title"])
            if bool(sanitized_text & sanitized_title):  # if there is any overlap we return true
                results.append(entry["title"])

    return results


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
