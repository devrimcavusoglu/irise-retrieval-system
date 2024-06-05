from typing import List, Tuple

import ir_datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class NeuralSearch:
    """
    Class for performing neural search using SentenceTransformer (BERT variants).

    Args:
        model_name_or_path (str): Path to the pre-trained model or URL. [default: 'thenlper/gte-base']
        dataset (str): Dataset name. [default: 'beir/msmarco/test']
    """

    def __init__(self, model_name_or_path: str = "thenlper/gte-base", dataset: str = "beir/msmarco/test"):
        self._model = SentenceTransformer(model_name_or_path)
        self._dataset = ir_datasets.load(dataset)
        self._docs_store = self._dataset.docs_store()

    def _get_text_from_dataset(self, doc_ids: list[str]):
        instance_text = [self._docs_store.get(doc_id).text for doc_id in doc_ids]
        return instance_text

    def _get_similarity_results(self, query_embedding, passage_embeddings, initial_results) -> list[Tuple[str, float]]:
        result = cos_sim(query_embedding, passage_embeddings)
        result_dict = {doc_id: score for doc_id, score in zip(initial_results, result.flatten().tolist())}
        sorted_results = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def __call__(self, query: str, initial_results: list[str], top_k: int = 10, return_dict: bool = False):
        """
        Performs neural search using sentence-transformer embedding model.

        Args:
            query (str): Query sentence.
            initial_results (list[str]): Initial search results, list of doc IDs.
            top_k (int): Number of results to return.
            return_dict (bool): Whether to return result dict or not.

        Returns:
            Result list, sorted by similarity score.
        """
        passages = self._get_text_from_dataset(initial_results)
        passage_embeddings = self._model.encode(passages)
        query_embedding = self._model.encode(query)
        result = self._get_similarity_results(query_embedding, passage_embeddings, initial_results)
        result = result[:top_k]
        if return_dict:
            results = []
            retrieved_passages = self._get_text_from_dataset([doc_id for doc_id, _ in result])
            for (doc_id, score), passage in zip(result, retrieved_passages):
                results.append({
                    "doc_id": doc_id,
                    "passage": passage,
                    "score": score
                })
            return results
        return result

