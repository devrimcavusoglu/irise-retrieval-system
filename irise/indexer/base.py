import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import ir_datasets
from tqdm import tqdm
from whoosh import index
from whoosh.fields import SchemaClass
from whoosh.multiproc import MpWriter
from whoosh.qparser import QueryParser, OrGroup
from whoosh.scoring import BM25F, TF_IDF
from whoosh.searching import Searcher
from whoosh.writing import SegmentWriter

from irise import DEFAULT_INDEX_DIR
from irise.indexer.custom_scoring import MixedWeighting
from irise.utils import PathOrStr
from irise.utils.schema import IriseSchema


class BaseIndexer(ABC):
    def __init__(self, path: Optional[PathOrStr] = None, schema: Optional[SchemaClass] = None):
        path = path or DEFAULT_INDEX_DIR
        schema = schema or IriseSchema()
        self.path = Path(path)
        self.schema = schema
        self.index = None
        self._writer = None
        self._searcher = None

    def _get_writer(self, **kwargs) -> MpWriter | SegmentWriter:
        if self._writer is None:
            self.create_index()
            self._writer = self.index.writer(**kwargs)
        return self._writer

    def _get_searcher(self, **kwargs) -> Searcher:
        if self._searcher is None:
            self.create_index()
            self._searcher = self.index.searcher(**kwargs)
        return self._searcher

    def create_index(self) -> None:
        if self.index is not None:
            return
        elif self.path.is_dir():
            self.index = index.open_dir(self.path)
            return

        self.path.mkdir(parents=True)
        self.index = index.create_in(self.path, self.schema)

    @abstractmethod
    def preprocess(self, text: str, **kwargs):
        """Preprocess given text."""
        raise NotImplementedError

    def __call__(self, dataset: str = "beir/msmarco/test", procs: int = -1, subset: bool = False, *args, **kwargs):
        """Indexes the given dataset."""
        if self.path.is_dir():
            raise FileExistsError("The index exists. To create a new index, remove the current index directory first.")
        self.create_index()
        if procs == -1:
            procs = os.cpu_count()
        dataset = ir_datasets.load(dataset)
        docs_store = dataset.docs_store()
        writer = self._get_writer(procs=procs, **kwargs)
        if subset:
            print("Indexing query relevance entries.")
            qrels = [qrel.doc_id for qrel in dataset.qrels_iter()]
            print("Indexing documents.")
            all_docs = [doc.doc_id for doc in dataset.docs_iter()]
            print("Creating the subset.")
            non_qrels = list(set(all_docs) - set(qrels))
            subset = qrels + []
            for _ in range(len(qrels)):
                idx = random.randint(0, len(non_qrels))
                selected_doc = non_qrels.pop(idx)
                subset.append(selected_doc)
            for doc_id in tqdm(subset, desc="Indexing documents in the subset", total=len(subset)):
                doc = docs_store.get(doc_id)
                text = self.preprocess(doc.text)
                writer.add_document(doc_id=doc.doc_id, text=text)  # MSMarcoSchema
        else:
            for doc in tqdm(dataset.docs_iter(), desc="Indexing documents in the dataset", total=dataset.docs_count()):
                text = self.preprocess(doc.text)
                writer.add_document(doc_id=doc.doc_id, text=text)  # MSMarcoSchema
        writer.commit()

    def search(self, query: str, beta: float = 0.5, or_group: bool = True, return_ids_only: bool = False, **kwargs):
        if self.index is None:
            self.create_index()

        qp_kwargs = {}
        if or_group:
            qp_kwargs["group"] = OrGroup
        query = self.preprocess(query)
        qp = QueryParser("text", schema=self.index.schema, **qp_kwargs)
        q = qp.parse(query)
        w = MixedWeighting(beta=beta)
        searcher = self._get_searcher(weighting=w)
        search_results = searcher.search(q, **kwargs)
        if return_ids_only:
            results = []
            for result in search_results:
                results.append(result["doc_id"].split("_")[-1])
            return results
        return search_results


class Indexer(BaseIndexer):
    def preprocess(self, text: str, **kwargs):
        return text
