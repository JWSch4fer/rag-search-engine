import math
from typing import List, Dict
from pathlib import Path
from sortedcontainers import SortedList
from rag_search_engine.utils.utils import fix_text, load_data, normalize_token_semantic


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

    # sanatize the query text
    sanitized_text = fix_text(text)
    st = set([sanitized_text])

    # 1) load titles
    data = load_data(DATA)
    titles = [(e["title"], set(e["title"].lower().split())) for e in data]

    # return if any part of sanitized text is present at all...
    def any_substring(qset: set[str], tset: set[str]) -> bool:
        # print(">>",qset, tset)
        return bool(qset & tset)

    # more permisive return statement because substrings can now match
    return [title for title, tset in titles if any_substring(st, tset)]


# ________________________________________________________________________________

# ________________________________________________________________________________
# ____________________Search Algorithm____________________________________________
# ________________________________________________________________________________


def search_inverted_index(
    text: str,
    postings: Dict[str, Dict[int, List[int]]],
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
    sanitized_text = [
        normalize_token_semantic(t) for t in fix_text(text).lower().split()
    ]
    first_five = []
    for word in sanitized_text:
        if postings.get(word):
            first_five += list(postings[word].keys())[:5]
    return set(first_five)


def bm25_search(
    text: str,
    postings: Dict[str, Dict[int, List[int]]],
    docmap: Dict[int, Dict[str, str]],
    doclen: Dict[int, int],
) -> List:
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
    sanitized_text = [
        normalize_token_semantic(t) for t in fix_text(text).lower().split()
    ]

    # results based on inverted index
    prelim_results = set()
    for word in sanitized_text:
        if postings.get(word):
            prelim_results = prelim_results | set(postings[word].keys())

    # rank order results based on bm25
    doc_names = SortedList()  # (bm25score, title)
    for id in prelim_results:
        word_score = 0
        for word in sanitized_text:
            word_score += calc_bm25_freq(word, id, docmap, doclen) * calc_bm25(
                word, postings, docmap
            )
        doc_names += [(word_score, id)]

    return doc_names[-15:]


def calc_idf(
    text: str,
    postings: Dict[str, Dict[int, List[int]]],
    docmap: Dict[int, Dict[str, str]],
) -> float:
    # sanatize the query text
    sanitized_text = normalize_token_semantic(fix_text(text).lower())
    idf = math.log(
        (len(docmap.keys()) + 1) / (len(postings[sanitized_text].keys()) + 1)
    )
    return idf


def calc_freq(
    text: str,
    doc_id: int,
    docmap: Dict[int, Dict[str, str]],
) -> int:

    sanitized_text = normalize_token_semantic(fix_text(text).lower())
    return docmap[doc_id]["description"].lower().count(sanitized_text) + docmap[doc_id][
        "title"
    ].lower().count(sanitized_text)


def calc_tf_idf(
    doc_id: int,
    text: str,
    postings: Dict[str, Dict[int, List[int]]],
    docmap: Dict[int, Dict[str, str]],
) -> float:
    # sanatize the query text
    # sanitized_text = fix_text(text).lower()
    idf = calc_idf(text, postings, docmap)
    tf = calc_freq(text, doc_id, docmap)
    return tf * idf


def calc_bm25(
    text: str,
    postings: Dict[str, Dict[int, List[int]]],
    docmap: Dict[int, Dict[str, str]],
) -> float:
    # sanatize the query text
    sanitized_text = normalize_token_semantic(fix_text(text).lower())
    N = len(docmap.keys())
    df = len(postings[sanitized_text].keys())
    return math.log(((N - df + 0.5) / (df + 0.5)) + 1)


def calc_bm25_freq(
    text: str,
    doc_id: int,
    docmap: Dict[int, Dict[str, str]],
    doclen: Dict[int, int],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    BM25_K1 = k1
    sanitized_text = normalize_token_semantic(fix_text(text).lower())
    tf = docmap[doc_id]["description"].lower().count(sanitized_text) + docmap[doc_id][
        "title"
    ].lower().count(sanitized_text)

    avg_len = sum(doclen.values()) / len(doclen)

    length_norm = 1 - b + b * (doclen[doc_id] / avg_len)
    return (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * length_norm)


# ________________________________________________________________________________
