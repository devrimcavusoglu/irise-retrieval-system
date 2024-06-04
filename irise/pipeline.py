from typing import Dict, Any

from whoosh.fields import SchemaClass

from irise.indexer import Indexer
from irise.neural_search import NeuralSearch


class SearchPipeline:
    def __init__(self, index_path: str = None, index_schema: SchemaClass = None, search_kwargs: Dict[str, Any] = None):
        search_kwargs = search_kwargs or {}
        self._indexer = Indexer(index_path, index_schema)
        self._neural_search = NeuralSearch(**search_kwargs)

    def __call__(self, query: str, initial_search_limit: int = 200, top_k: int = 5):
        results = self._indexer.search(query, limit=initial_search_limit, return_ids_only=True)
        results = self._neural_search(query, results, top_k=top_k)
        return results


if __name__ == "__main__":
    from irise import INDEX_DIR
    pipe = SearchPipeline(index_path=INDEX_DIR / "irise_index_advanced")
    r = pipe("health environment")
    print(r)
