import pickle, json
from pathlib import Path
from typing import Callable, Dict, List, Iterator
from collections import defaultdict
from sortedcontainers import SortedDict

# from rag_search_engine.utils.search import normalize_for_index
from rag_search_engine.utils.utils import load_data, preprocess

ROOT = Path(__file__).resolve().parents[1]

# ________________________________________________________________________________
# ____________________Inverted Index______________________________________________
# ________________________________________________________________________________


def ddlist() -> Dict[int, List[int]]:
    """
    define for nesting default dict and pickle compatibility
    """
    return defaultdict(list)


class InvertedIndex:
    """
    postings: term -> {doc_id -> [positions]}
    docmap:   doc_id -> raw text (or metadata)
    """

    INDEX_NAME = "index.pkl"
    DOCMAP_NAME = "docmap.pkl"
    DOCLEN_NAME = "doclen.pkl"
    META_NAME = "meta.json"

    TITLE_END_TOKEN = "[TITLE_END]"  # field boundary token

    def __init__(
        self,
        file_path: Path | str | None = None,
        cache_dir: Path | str = ROOT / "cache",
        normalizer: Callable[[str], Iterator[List[str]]] = preprocess,
    ) -> None:
        self.file_path = Path(file_path) if file_path else None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._postings: Dict[str, Dict[int, List[int]]] = defaultdict(ddlist)
        self._docmap: Dict[int, Dict[str, str]] = SortedDict()
        self._doclen: Dict[int, int] = SortedDict()
        self._normalizer = normalizer
        self._built = False

    def __repr__(self) -> str:

        repr = f"--------------------------------------\n"
        repr += f"Cache directory path: {self.cache_dir}\n"
        repr += f"Input data file path: {self.file_path}\n"
        repr += f"--------------------------------------\n"
        return repr

    # --- accessors ---
    def index(self) -> Dict[str, Dict[int, List[int]]]:
        return self._postings

    def docmap(self) -> Dict[int, Dict[str, str]]:
        return self._docmap

    def doclen(self) -> Dict[int, int]:
        return self._doclen

    # ---- simple (per-doc) build ----
    def build(self) -> None:
        """
        Simple build: iterate JSON list of {id,title,description} and index each.
        """
        if not self.file_path:
            raise Exception("Need a valid source to build cache")

        # data = json.loads(self.file_path.read_text(encoding="utf-8"))
        data = load_data(self.file_path)
        titles = [t["title"] for t in data]
        descriptions = [t["description"] for t in data]

        title_tok_lists = preprocess(titles)
        body_tok_lists = preprocess(descriptions)

        # create fresh iterator
        data_iter = iter(data)
        for d, t_tokens, b_tokens in zip(data_iter, title_tok_lists, body_tok_lists):
            doc_id = int(d["id"])
            tokens = t_tokens + [self.TITLE_END_TOKEN] + b_tokens
            self._docmap[doc_id] = {
                "title": d["title"],
                "description": d["description"],
            }
            self._doclen[doc_id] = len(d["description"])

            for pos, tok in enumerate(tokens):
                self._postings[tok][doc_id].append(pos)
        self._built = True

    # --- persistence (cache) ---
    def _sig_for_docs(self, docs_path: Path | None) -> dict:
        """
        build from a JSON/CSV file, pass that path here when saving.
        Used to auto-invalidate caches when the source changes.
        """
        if not docs_path or not docs_path.exists():
            return {}
        st = docs_path.stat()
        return {
            "src": str(docs_path),
            "mtime_ns": int(st.st_mtime_ns),
            "size": st.st_size,
        }

    def save(self) -> None:
        """
        Save postings + docmap + meta into cache_dir using gzip pickles.
        """
        if not self._built and not self._postings:
            # allow saving incremental states but warn mentally
            pass

        idx_file = self.cache_dir / self.INDEX_NAME
        dmap_file = self.cache_dir / self.DOCMAP_NAME
        len_file = self.cache_dir / self.DOCLEN_NAME
        meta_file = self.cache_dir / self.META_NAME

        with open(idx_file, "wb") as f:
            pickle.dump(self._postings, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(dmap_file, "wb") as f:
            pickle.dump(self._docmap, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(len_file, "wb") as f:
            pickle.dump(self._doclen, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta = {
            "normalizer": getattr(self._normalizer, "__name__", "callable"),
            "signature_dmap": self._sig_for_docs(dmap_file),
            "signature_dlen": self._sig_for_docs(len_file),
            "signature_idx": self._sig_for_docs(idx_file),
            "counts": {"terms": len(self._postings), "docs": len(self._docmap)},
            "version": 1,
        }
        meta_file.write_text(json.dumps(meta, indent=2))

    def load(self) -> bool:
        """
        Load postings + docmap from cache_dir if present.
        Returns True if loaded, False otherwise.
        """
        idx_file = self.cache_dir / self.INDEX_NAME
        dmap_file = self.cache_dir / self.DOCMAP_NAME
        len_file = self.cache_dir / self.DOCLEN_NAME

        if not idx_file.exists() or not dmap_file.exists():
            return False

        with open(idx_file, "rb") as f:
            self._postings = pickle.load(f)
        with open(dmap_file, "rb") as f:
            self._docmap = pickle.load(f)
        with open(len_file, "rb") as f:
            self._doclen = pickle.load(f)

        self._built = True
        return True

    def exists(self) -> bool:
        """
        check postings + docmap from cache_dir if present.
        Returns True if exist, False otherwise.
        """
        idx_file = self.cache_dir / self.INDEX_NAME
        dmap_file = self.cache_dir / self.DOCMAP_NAME

        if not idx_file.exists() or not dmap_file.exists():
            return False

        return True


# ________________________________________________________________________________
