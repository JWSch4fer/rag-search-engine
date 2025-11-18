import re, html, unicodedata, codecs, json
from functools import lru_cache
from typing import Dict, List
from pathlib import Path
# import numpy as np
# import numpy.typing as npt
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from rapidfuzz.fuzz import partial_ratio
import unicodedata
from rapidfuzz import process

ROOT = Path(__file__).resolve().parents[2]

CANONICAL_VOCAB = {
    # core genres
    "sciencefiction",
    "cyberpunk",
    "fantasy",
    "horror",
    "thriller",
    "comedy",
    "drama",
    "romance",
    "action",
    "adventure",
    "mystery",
    "crime",
    "documentary",
    "biography",
    "war",
    "western",
    "musical",
    "family",
    # subgenres / styles
    "noir",
    "heist",
    "gangster",
    "spaghettiwestern",
    "martialarts",
    "psychologicalthriller",
    "psychologicalhorror",
    "technothriller",
    "foundfootage",
    "slasher",
    "splatter",
    "romcom",
    "screwballcomedy",
    "slapstickcomedy",
    "comingofage",
    "sliceoflife",
    "period",
    "arthouse",
    "blackandwhite",
    "documentary",
    "youngadult",
    # formats
    "anime",
    "animation",
    "liveaction",
    "stopmotion",
    "cgi",
    "3d",
    "2d",
    "series",
    "television",
    "televisionfilm",
    "miniseries",
    "documentaryseries",
    "movie",
    "film",
    "short",
    "episode",
    # comics/superheroes
    "superhero",
    "comicbook",
    # anime subtypes
    "shonen",
    "shojo",
    "seinen",
    "josei",
    "mecha",
    "isekai",
    "magicalgirl",
    "ova",
    "ona",
    # audience / misc
}

NORMALIZATION_MAP = {
    # sci-fi & tech
    "scifi": "sciencefiction",
    "sci-fi": "sciencefiction",
    "sf": "sciencefiction",
    "sci fi": "sciencefiction",
    "sci_fi": "sciencefiction",
    "spaceopera": "sciencefiction",
    # animation / format
    "animated": "anime",
    "animation": "anime",
    "animations": "anime",
    "animator": "anime",
    "animators": "anime",
    "animate": "anime",
    "cartoon": "anime",
    "cartoons": "anime",
    "live-action": "liveaction",
    "stop-motion": "stopmotion",
    # tv / series
    "t.v.": "television",
    "tv": "television",
    "tvmovie": "television",
    "tv-movie": "television",
    "limitedseries": "miniseries",
    "mini-series": "miniseries",
    "docuseries": "documentaryseries",
    "docu-series": "documentaryseries",
    "episode": "episode",
    "ep": "episode",
    # documentary / bio
    "docu": "documentary",
    "biopic": "documentary",
    "bio-pic": "documentary",
    # romance/comedy
    "rom-com": "romcom",
    "rom com": "romcom",
    "romcoms": "romcom",
    "screwball": "comedy",
    "slapstick": "comedy",
    # horror & thriller
    "found-footage": "horror",
    "psychological thriller": "horror",
    "psychological horror": "horror",
    "techno-thriller": "horror",
    "technothriller": "horror",
    "splatter": "horror",
    "slasher": "horror",
    # crime/noir
    "film-noir": "noir",
    "filmnoir": "noir",
    # western
    "spaghetti-western": "western",
    # action/martial arts
    "martial-arts": "martialarts",
    # period/style
    "period piece": "period",
    "period-piece": "period",
    "coming-of-age": "comingofage",
    "slice-of-life": "sliceoflife",
    "arthouse": "arthouse",
    "art-house": "arthouse",
    "black-and-white": "blackandwhite",
    "b&w": "blackandwhite",
    # superhero/comics
    "super-hero": "superhero",
    "comic-book": "comicbook",
    "comic book": "comicbook",
    # audience
    "family-friendly": "family",
    "young-adult": "youngadult",
    "ya": "youngadult",
}


FUZZY_SCORE_CUTOFF = 85  # a bit looser to catch 'animé'
MIN_LEN_FOR_FUZZY = 3

ALLOWLIST = {"go", "get", "make"}
STOPWORDS = set(STOP_WORDS) - ALLOWLIST


# --------- text normalization (pre-spaCy) ---------
_u_escape = re.compile(r"\\u[0-9a-fA-F]{4}")

def fix_text(s: str) -> str:
    # NOTE: case is not changed!!

    # Decode JSON-style unicode escapes if they appear literally (double-escaped case)
    if _u_escape.search(s):
        try:
            s = codecs.decode(s, "unicode_escape")
        except Exception:
            pass
    # Decode HTML entities (&eacute; -> é)
    s = html.unescape(s)
    # Normalize composed form
    return unicodedata.normalize("NFC", s)


def load_data(file_path: Path | str) -> List[Dict[str, str]]:
    file_path = Path(file_path)  # ensure pure path
    data = json.loads(file_path.read_text(encoding="utf-8"))
    data = data["movies"]  # now a list of dicts
    for idx, movie in enumerate(data):
        data[idx]["title"] = fix_text(movie["title"])
        data[idx]["description"] = fix_text(movie["description"])
    return data


def fold_diacritics(s: str) -> str:
    # "animé" -> "anime"
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)
    )


@lru_cache(maxsize=65536)
def normalize_token_semantic(tok: str) -> str:
    # Rule 1: fold diacritics for fuzzy (but don't change the visible token unless matched)
    folded = fold_diacritics(tok)

    # Rule 2: explicit dictionary to improve inverted dictionary
    norm = NORMALIZATION_MAP.get(folded, folded)

    # Rule 3: fuzzy to canonical vocab
    if norm not in CANONICAL_VOCAB and len(norm) >= MIN_LEN_FOR_FUZZY:
        match = process.extractOne(
            norm,
            CANONICAL_VOCAB,
            score_cutoff=FUZZY_SCORE_CUTOFF,
            scorer=partial_ratio,
        )

        # print(match, folded)
        if match:
            return match[0]  # return canonical label
    return norm


def preprocess(texts: str | List[str], n_process=1, batch_size=256) -> List[str]:
    """
    texts: iterable[str]
    returns: list[list[str]]  (tokens per text)
    """
    try:
        # 1) Load spaCy (keep POS for lemmatizer; drop NER for speed)
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        import spacy.cli.download as download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # 1) normalize outside spaCy (fast, vectorizable)
    match texts:
        case str():
            fixed = [fix_text(texts).lower()]
        case list():
            fixed = [fix_text(t).lower() for t in texts]

    # 2) pipe through spaCy in batches (parallel if n_process>1)
    out_all = []
    sw = STOPWORDS  # local refs are quicker in hot loops
    for doc in nlp.pipe(fixed, n_process=n_process, batch_size=batch_size):
        out = []
        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue

            lemma = tok.lemma_
            if not lemma:
                lemma = tok.text
            lemma = lemma.strip().lower()

            # skip stopwords (allowlist already removed from sw)
            if lemma in sw:
                continue

            # keep only alnum chars (light punctuation strip)
            # (keeps numbers; remove .isdigit() if you don't want them)
            cleaned = "".join(ch for ch in lemma if ch.isalnum())

            if cleaned:
                # semantic normalization (synonyms + fuzzy)
                out.append(normalize_token_semantic(cleaned))

        out_all.append(out)
    return out_all


# def cosine_similarity(
#     vec_db: npt.NDArray[np.float64],  # shape (N, D)
#     vec: npt.NDArray[np.float64],  # shape (D,)
# ) -> npt.NDArray[np.int32]:  # shape (N,)
#     """
#     Compute cosine similarity between each row in vec_db and vec.

#     vec_db: 2D array of shape (N, D)
#     vec:    1D array of shape (D,)
#     returns: 1D array of shape (N,) with cosine similarities
#     """
#     if vec_db.ndim != 2:
#         raise ValueError(f"vec_db must be 2D (N, D); got shape {vec_db.shape}")
#     if vec.ndim != 1:
#         raise ValueError(f"vec must be 1D (D,); got shape {vec.shape}")
#     if vec_db.shape[1] != vec.shape[0]:
#         raise ValueError(
#             f"Dimension mismatch: vec_db has D={vec_db.shape[1]}, "
#             f"vec has D={vec.shape[0]}"
#         )

#     # Norm of the query vector
#     norm_vec = np.linalg.norm(vec)
#     if norm_vec == 0:
#         # If the query is the zero vector, all similarities are 0
#         return np.zeros(vec_db.shape[0], dtype=np.float64)

#     # Norms of each row in vec_db
#     norms_db = np.linalg.norm(vec_db, axis=1)  # shape (N,)

#     # Dot products between each row and vec
#     dots = vec_db @ vec  # shape (N,)

#     # Avoid division by zero for zero rows in vec_db
#     sims = np.zeros(vec_db.shape[0], dtype=np.float64)

#     sims = dots / (norms_db * norm_vec)

#     # track which documents are the best match to the query
#     dt = np.dtype([("id", np.int32), ("sim", np.float64)])
#     id_sims = np.zeros(sims.shape[0], dtype=dt)

#     id_sims["id"] = np.arange(sims.shape[0], dtype=np.int32)
#     id_sims["sim"] = sims

#     return_idx = np.argsort(id_sims["sim"])
#     return id_sims["id"][return_idx[-10:]]


def chunk(text: str | List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    """
    chunk input text into a List

        text: input string to be chunked
        chunk_size: number of words per chunk
        overlap: number of overlapping words between adjacent chunks
    """
    match text:
        case str():
            # if a raw string is the input we split on white space
            text = text.strip().split()
        case list():
            # if a list of strings is input we assume it was pre-split
            pass

    chunked_text, chunk = [text[:chunk_size]], []
    for phrase in text[chunk_size:]:
        if overlap > 0 and len(chunk) == 0 and overlap < chunk_size:
            chunk = chunked_text[-1][-overlap:]
        chunk.append(phrase)

        if len(chunk) == chunk_size:
            chunked_text.append(chunk)
            chunk = []

    if chunk:
        chunked_text.append(chunk)

    return chunked_text


def semantic_chunk(
    text: str | List[str], max_chunk_size: int, overlap: int
) -> List[List[str]]:
    """
    chunk input text into a List based on punctuaion

        text: input string to be chunked
        max_chunk_size: maximum number of sentences per chunk
        overlap: number of overlapping sentences between adjacent chunks
    """
    # precompiled at import time
    SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    match text:
        case str():
            return chunk(SPLIT_RE.split(text.strip()), max_chunk_size, overlap)
        case list():
            # double list comprehension to return List[str], not nested
            return [
                s
                for t in text
                for s in chunk(SPLIT_RE.split(t.strip()), max_chunk_size, overlap)
            ]


def min_max_norm(nums: List[float]) -> List[float]:
    """
    return the normalized list of numbers based on min/max
    """
    min_score = min(nums)
    max_score = max(nums)
    if min_score == max_score:
        return [1.0] * len(nums)
    minmax = lambda x: (x - min_score) / (max_score - min_score)
    return [minmax(n) for n in nums]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5):
    """
    α = 1.0: [████████████████████] 100% Keyword
    α = 0.7: [██████████████------] 70% Keyword, 30% Semantic
    α = 0.5: [██████████----------] 50/50 Split
    α = 0.2: [████----------------] 20% Keyword, 80% Semantic
    α = 0.0: [--------------------] 100% Semantic
    """
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k=60) -> float:
    return 1 / (k + rank)
