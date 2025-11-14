import re, html, unicodedata, codecs, json
from functools import lru_cache
from typing import Dict, List
from pathlib import Path
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from rapidfuzz.fuzz import partial_ratio
import unicodedata
from rapidfuzz import process


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
_intra_dash = re.compile(r"(?<=\w)[\-\u2010-\u2015](?=\w)")


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
        from spacy.cli import download

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
