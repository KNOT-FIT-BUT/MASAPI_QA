# -*- coding: UTF-8 -*-
"""
Created on 02.11.22
Dump retriever model that returns first-k entries in database.

:author:     Martin Dočekal
"""
from pathlib import Path

from masapiqa.database import Database

from masapiqa.module.retriever.frameworks.base import BaseRetrieverFramework


class FirstKFramework(BaseRetrieverFramework):
    """
    Dump retriever model that returns first-k entries in database.
    The score, for each passage, is always 1.0.
    """

    def __init__(self, database_path: str):
        directory_path = Path(database_path)
        self.database = Database(str(directory_path.joinpath("database.jsonl")))

    def predict(self, question: str, config: dict) -> dict:
        indices = []
        scores = []
        passages = []
        titles = []
        with self.database:
            for i, record in enumerate(self.database[:config["top_k"]]):
                indices.append(i)
                scores.append(1.0)
                passages.append(record.contents)
                titles.append(record.title)

        return {
            "question": question,
            "indices": indices,
            "scores": scores,
            "passages": passages,
            "titles": titles
        }

    def to(self, device):
        ...

