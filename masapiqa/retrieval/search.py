# -*- coding: UTF-8 -*-
""""
Created on 12.08.22

Module containing search approaches for searching associated embeddings.

:author:     Martin Dočekal
"""
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class SearchApproach(ABC):
    """
    Abstract base class for search approaches for searching associated embeddings.
    Every search approach must be a functor that is able to provide the most associated indices.
    """
    def __init__(self, k: int):
        """
        initialization of an approach for searching

        :param k: how many top-k ones should be searched (maximally)
        """
        self.k = k

    @abstractmethod
    def __call__(self, keys: np.array, emb_matrix: np.array) -> Sequence[Sequence[int]]:
        """
        searches indices of the most associated embeddings, in embeddings matrix, for each given key

        :param keys: keys embeddings for which we are searching the associated ones
            shape KEYS X FEATURES
        :param emb_matrix: embeddings matrix in which we are searching
            RECORDS X FEATURES
        :return: indices of top-k most associated records for each key
            sorted in descending order (most associated one is first)
        """
        ...


class DotProductSearch(SearchApproach):
    """
    Uses dot product to search the associated embeddings.
    """

    def __call__(self, keys: np.array, emb_matrix: np.array) -> Sequence[Sequence[int]]:
        return np.argsort(-np.dot(keys, emb_matrix.T))[:, :self.k]
