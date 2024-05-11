from abc import ABC, abstractmethod
from pathlib import Path

from tqdm import tqdm
from whoosh import index
from whoosh.fields import SchemaClass
from whoosh.multiproc import MpWriter
from whoosh.qparser import QueryParser
from whoosh.searching import Searcher
from whoosh.writing import SegmentWriter

from irise import DEFAULT_INDEX_DIR
from irise.utils import PathOrStr


class BaseIndexer(ABC):
    def __init__(self, schema: SchemaClass, path: PathOrStr = DEFAULT_INDEX_DIR, create_on_init: bool = True, n_procs: int = 1, **kwargs):
        self.path = Path(path)
        self.schema = schema
        self.n_procs = n_procs
        self.index = None
        self._writer: MpWriter | SegmentWriter = None
        self._searcher: Searcher = None
        if create_on_init:
            self.create_index()

    @property
    def writer(self):
        if self._writer is None:
            self._writer = self.index.writer(procs=self.n_procs)
        return self._writer

    @property
    def searcher(self):
        if self._searcher is None:
            self._searcher = self.index.searcher()
        return self._searcher

    def create_index(self) -> None:
        if self.path.is_dir():
            self.index = index.open_dir(self.path)
            return

        self.path.mkdir()
        self.index = index.create_in(self.path, self.schema)

    @abstractmethod
    def preprocess(self, text: str, **kwargs):
        """Preprocess given text."""
        raise NotImplementedError

    def __call__(self, dataset: str = "beir/msmarco/test", *args, **kwargs):
        """Indexes the given dataset."""
        if self.path.is_dir():
            raise FileExistsError("The index exists. To create a new index, remove the current index directory first.")
        dataset = ir_datasets.load(dataset)
        for doc in tqdm(dataset.docs_iter(), desc="Indexing documents:", total=dataset.docs_count()):
            text = self.preprocess(doc.text)
            self.writer.add_document(doc_id=f"doc_{doc.doc_id}", text=text)  # MSMarcoSchema
        self.writer.commit()

    def search(self, query: str, limit: int):
        qp = QueryParser("text", schema=self.schema)
        q = qp.parse(query)
        return self.searcher.search(q, limit=limit)


class Indexer(BaseIndexer):
    def preprocess(self, text: str, **kwargs):
        return text


if __name__ == "__main__":
    import ir_datasets
    from irise.utils.schema import IriseSchema
    indexer = Indexer(IriseSchema)
    indexer()
