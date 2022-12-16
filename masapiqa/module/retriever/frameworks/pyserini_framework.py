# -*- coding: UTF-8 -*-
"""
Created on 07.10.22
Retriever that uses pyserini.

:author:     Martin Dočekal
"""
import json
from pathlib import Path

from masapiqa.database import Database
from masapiqa.module.retriever.frameworks.base import BaseRetrieverFramework
from pyserini.search import LuceneSearcher, FaissSearcher, DprQueryEncoder


class PyseriniRetrieverFramework(BaseRetrieverFramework):

    def __init__(self, database_path: str):
        directory_path = Path(database_path)
        config_path = directory_path.joinpath("config.json")
        self.index_path = str(directory_path.joinpath('index'))
        self.q_enc = None

        with open(config_path) as f:
            self.config = json.load(f)

        if self.config["index_type"] == "BM25":
            self.searcher = LuceneSearcher(self.index_path)
        elif self.config["index_type"] == "DPR":
            self.q_enc = DprQueryEncoder(self.config["suggested_query_encoder"])
            self.searcher = FaissSearcher(self.index_path, self.q_enc)
        else:
            raise RuntimeError("Loading unknown index.")

        self.database = Database(str(directory_path.joinpath("database.jsonl")))

    def predict(self, question: str, config: dict) -> dict:
        indices = []
        scores = []
        passages = []
        titles = []
        with self.database:
            for res in self.searcher.search(question, config["top_k"]):
                indices.append(res.docid)
                scores.append(res.score)
                db_record = self.database[int(res.docid)]
                passages.append(db_record.contents)
                titles.append(db_record.title)

        return {
            "question": question,
            "indices": indices,
            "scores": scores,
            "passages": passages,
            "titles": titles
        }

    def to(self, device):
        if self.q_enc is not None:
            self.q_enc.model.to(device)
            self.q_enc.device = device
