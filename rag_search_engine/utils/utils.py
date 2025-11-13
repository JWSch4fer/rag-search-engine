import re, html, unicodedata, codecs, json
from typing import Dict, List
from pathlib import Path
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

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

    # 2) Build a gentle stopword set and protect words you want to keep
    ALLOWLIST = {"go", "get", "make"}
    STOPWORDS = set(STOP_WORDS) - ALLOWLIST

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
                out.append(cleaned)
        out_all.append(out)
    return out_all
