# -*- coding: UTF-8 -*-
""""
Created on 12.08.22
Module for retrieval indexes.

:author:     Martin Dočekal
"""
from abc import abstractmethod
from typing import Sequence, TypeVar, Generic

import numpy as np

from masapiqa.retrieval.encoding import Encoder
from masapiqa.retrieval.search import SearchApproach

K = TypeVar("K")


class Index(Generic[K]):
    """
    Base class for retrieval indexes.
    An index is a mapping to sequence of integers that are indices of most associated records in a database. Therefore,
    in index is always associated with some kind of datasource.

    :ivar k: top k indices
    """

    def __init__(self, k: int):
        """
        initialization of an index

        :param k: how many top-k indices should be selected (maximally)
        """
        self.k = k

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __call__(self, keys: Sequence[K]) -> Sequence[Sequence[int]]:
        """
        Maps a sequence of keys into indices of most associated records to each of them.

        :param keys: sequence of keys used for search
        :return: Sequence of sequence for each key with associated indices. The most associated one is first.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, from_file: str) -> "Index":
        """
        Load index from file.

        :param from_file: path to file where the index is saved
        :return: loaded index
        """
        ...

    @abstractmethod
    def save(self, to_file: str):
        """
        Save index into given file.

        :param to_file: path where the index will be saved
        """
        ...


class EncoderIndex(Index[str]):
    """
    An index that uses encoder to obtain embeddings of string keys to search associated records.

    Example:
        >>> index = EncoderIndex(my_encoder, embeddings_for_search, 3, my_search_approach)
        >>> index(["Who sells cars?"])
        [[1,5, 20]]
    """
    def __init__(self, encoder: Encoder, embeddings: np.array, k: int, search_approach: SearchApproach):
        """
        initialization of index

        :param embeddings: the embedding matrix with an embedding for each record
            shape: RECORDS X FEATURES
        :param k: how many top-k indices should be selected (maximally)
        :param search_approach: strategy that is used for searching the most associated embeddings to a key
        """
        super().__init__(k)
        self.embeddings = embeddings
        self.encoder = encoder
        self.search_approach = search_approach

    def __len__(self):
        return self.embeddings.shape[0]

    def __call__(self, keys: Sequence[str]) -> Sequence[Sequence[int]]:
        key_embeddings = self.encoder(keys)
        return self.search_approach(key_embeddings, self.embeddings)

