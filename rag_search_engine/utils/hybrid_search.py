import os

from rag_search_engine.utils.inv_idx import InvertedIndex
from rag_search_engine.utils.semantic_search import SemanticSearch
from rag_search_engine.utils.search import bm25_search
from rag_search_engine.utils.utils import load_data, min_max_norm, rrf_score


class HybridSearch:
    def __init__(self, document):
        data = load_data(document)
        self.semantic_search = SemanticSearch()
        self.semantic_search.build_or_sync(data)

        self.invidx = InvertedIndex(document)
        if not os.path.exists(self.invidx.exists()):
            self.invidx.build()
            self.invidx.save()

    def _bm25_search(self, query: str, limit: int):
        self.invidx.load()
        return bm25_search(
            query,
            self.invidx.index(),
            self.invidx.docmap(),
            self.invidx.doclen(),
            limit,
        )

    def _semantic_search(self, query: str, limit: int):
        return self.semantic_search.query_top_k(query_text=query, k=limit)

    def weighted_search(self, query, alpha, limit=5):
        # [(score, id),...]
        bm25 = self._bm25_search(query=query, limit=limit)
        scores = [s for s, _ in bm25]
        new_scores = min_max_norm(scores)

        for idx, (_, doc_id) in enumerate(bm25):
            bm25[idx] = (doc_id, new_scores[idx], self.invidx.docmap()[doc_id])

        # [(id, score, {'title', 'description'}),...]
        semantic = self._semantic_search(query=query, limit=limit)
        scores = [s for _, s, _ in semantic]
        new_scores = min_max_norm(scores)
        for idx, (doc_id, _, info) in enumerate(semantic):
            semantic[idx] = (doc_id, new_scores[idx], info)

        result = semantic + bm25
        result.sort(key=lambda x: -x[1])
        return result
        # raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        """
        This one doesn't really work it needs a refactor
        """
        # [(score, id),...]
        bm25 = self._bm25_search(query=query, limit=limit)
        bm25.sort(key=lambda x: -x[0])
        bm25_rrf = {id: idx for idx, (_, id) in enumerate(bm25)}

        # [(id, score, {'title', 'description'}),...]
        semantic = self._semantic_search(query=query, limit=limit)
        semantic.sort(key=lambda x: -x[1])
        semantic_rrf = {id: idx for idx, (id, _, _) in enumerate(semantic)}

        result = []
        for d in list(bm25_rrf.keys()) + list(semantic_rrf.keys()):
            NOT_FOUND = 10000  # contribute very little
            score = rrf_score(bm25_rrf.get(d, NOT_FOUND)) + rrf_score(
                semantic_rrf.get(d, NOT_FOUND)
            )
            result.append((d, score, self.invidx.docmap()[d]))

        result.sort(key=lambda x: -x[1])
        return result
