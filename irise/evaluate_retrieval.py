# Use TREC Eval
# https://huggingface.co/spaces/evaluate-metric/trec_eval
import evaluate
from ir_datasets import Dataset
from tqdm import tqdm

from irise.indexer import Indexer


def retrieve_from_index(dataset: Dataset, indexer: Indexer, system: str):
    """Retrieve the relevant documents from queries from the (inverted) index."""
    predictions = {
        "query": [],
        "q0": [],
        "docid": [],
        "score": [],
        "rank": [],
        "system": [],
    }
    for query in tqdm(dataset.queries_iter(), total=dataset.queries_count()):
        results = indexer.search(query.text, limit=dataset.docs_count(), weighting=system)
        if not results:
            predictions["query"].append(int(query.query_id))
            predictions["q0"].append("q0")
            predictions["docid"].append(str(-1))
            predictions["score"].append(-1)
            predictions["rank"].append(-1)
            predictions["system"].append(system)
        else:
            for rank, (score, docid) in enumerate(results.top_n):
                predictions["query"].append(int(query.query_id))
                predictions["q0"].append(f"q{query.query_id}")
                predictions["docid"].append(str(docid))
                predictions["score"].append(score)
                predictions["rank"].append(rank)
                predictions["system"].append(system)


def run_evaluation():
    indexer = Indexer()
    trec_eval = evaluate.load("trec_eval")

    results = trec_eval.compute(references=[qrel], predictions=[run])