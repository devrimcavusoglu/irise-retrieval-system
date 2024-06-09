import fire

from irise.indexer import Indexer

if __name__ == "__main__":
    fire.Fire({
        "index": Indexer
    })
