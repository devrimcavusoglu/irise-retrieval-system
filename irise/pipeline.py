from typing import Dict, Any

from whoosh.fields import SchemaClass

from irise.indexer import Indexer
from irise.neural_search import NeuralSearch


class SearchPipeline:
    def __init__(self, index_path: str = None, index_schema: SchemaClass = None, search_kwargs: Dict[str, Any] = None):
        search_kwargs = search_kwargs or {}
        self._indexer = Indexer(index_path, index_schema)
        self._neural_search = NeuralSearch(**search_kwargs)

    def __call__(self, query: str, initial_search_limit: int = 200, top_k: int = 10, weighting="tfidf", return_dict: bool = True):
        results = self._indexer.search(query, weighting=weighting, limit=initial_search_limit, return_ids_only=True)
        results = self._neural_search(query, results, top_k=top_k, return_dict=return_dict)
        return results
