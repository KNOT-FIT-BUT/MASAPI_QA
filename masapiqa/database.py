# -*- coding: UTF-8 -*-
""""
Created on 12.08.22
Module containing tools for reading and creating databases.

:author:     Martin Dočekal
"""
import csv
from dataclasses import dataclass
from typing import List, Union, Iterable

from tqdm import tqdm
from windpyutils.files import JsonRecord, RecordFile


@dataclass
class PassageRecord(JsonRecord):
    """
    Single database record.
    """
    id: int
    split_index: int
    dataset_sample_id: str
    split_start_char_offset: int
    title: str
    contents: str


class Database:
    """
    Class for reading records from database containing all contexts.

    Example:
        >>>with Database("path_to/database/folder") as db:
        >>>    db[1]
        "PassageRecord(...)

    """

    def __init__(self, path_to: str, verbose: bool = False):
        """
        initialize database from its folder.

        :param path_to: path to database jsonl file
        :param verbose: whether the progress bars should be shown
        """

        self.passages_file: RecordFile[PassageRecord] = RecordFile(path_to, PassageRecord)
        self.verbose = verbose

    def _get_line_offsets(self, path: str) -> List[int]:
        """
        Creates list of offsets from index file.

        :param path: path to index file
        :return: list of offsets
        """
        res = []
        with open(path, newline='') as f:
            for r in tqdm(csv.DictReader(f, delimiter="\t"), desc="Indexing database", disable=not self.verbose):
                res.append(int(r["file_line_offset"]))
        return res

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> "Database":
        """
        opens database

        :return: Returns the object itself.
        """
        self.passages_file.open()
        return self

    def close(self):
        """
        closes database
        """
        self.passages_file.close()

    def __len__(self):
        return len(self.passages_file)

    def __getitem__(self, item: Union[int, slice, Iterable]) -> Union[PassageRecord, List[PassageRecord]]:
        """
        Get passage record on given index.

        :param item: index of a record
        :return: the record on given index
        """
        return self.passages_file[item]

