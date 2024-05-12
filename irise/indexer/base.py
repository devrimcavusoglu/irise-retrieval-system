import os
from abc import ABC, abstractmethod
from pathlib import Path

import ir_datasets
from tqdm import tqdm
from whoosh import index
from whoosh.fields import SchemaClass
from whoosh.multiproc import MpWriter
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F, TF_IDF
from whoosh.searching import Searcher
from whoosh.writing import SegmentWriter

from irise import DEFAULT_INDEX_DIR, PROJECT_ROOT
from irise.utils import PathOrStr
from irise.utils.schema import IriseSchema


class BaseIndexer(ABC):
    def __init__(self, path: PathOrStr = DEFAULT_INDEX_DIR, schema: SchemaClass = IriseSchema(), **kwargs):
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

    def __call__(self, dataset: str = "beir/msmarco/test", procs: int = -1, *args, **kwargs):
        """Indexes the given dataset."""
        if self.path.is_dir():
            raise FileExistsError("The index exists. To create a new index, remove the current index directory first.")
        self.create_index()
        if procs == -1:
            procs = os.cpu_count()
        dataset = ir_datasets.load(dataset)
        writer = self._get_writer(procs=procs, **kwargs)
        for doc in tqdm(dataset.docs_iter(), desc="Indexing documents:", total=dataset.docs_count()):
            text = self.preprocess(doc.text)
            writer.add_document(doc_id=f"doc_{doc.doc_id}", text=text)  # MSMarcoSchema
        writer.commit()

    def search(self, query: str, limit: int = 5, weighting: str = "tfidf", **kwargs):
        if self.index is None:
            self.create_index()
        qp = QueryParser("text", schema=self.index.schema)
        q = qp.parse(query)
        weighting = weighting.lower()
        if weighting in ["bm25", "bm25f"]:
            w = BM25F()
        elif weighting in ["tfidf", "tf_idf", "tf-idf"]:
            w = TF_IDF()
        else:
            raise ValueError("Unknown weighting scheme.")
        searcher = self._get_searcher(weighting=w, **kwargs)
        return searcher.search(q, limit=limit)


class Indexer(BaseIndexer):
    def preprocess(self, text: str, **kwargs):
        return text


if __name__ == "__main__":
    from irise import INDEX_DIR
    indexer = Indexer(path=INDEX_DIR / "irise_index_advanced")
    indexer()
