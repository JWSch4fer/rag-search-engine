# rag_search_engine/config.py
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, at import time (for CLI / local dev)
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# root of project
ROOT = Path(__file__).resolve().parents[1]
# database directory
DEFAULT_DB_PATH = ROOT / "rag_search_engine" / "cache" / "movies.db"


# natural language processing parameters
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
