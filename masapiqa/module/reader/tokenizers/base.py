import torch

from typing import List, Optional


class BaseReaderTokenizer(object):

    def __init__(self, tokenizer, config):
        self._model_config = config
        self._tokenizer = tokenizer
        self._max_length = config["max_length"] if config.get("max_length", None) else tokenizer.model_max_length
        self._device = torch.device("cpu")

    @property
    def model_config(self) -> dict:
        return self._model_config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_length(self) -> int:
        return self._max_length

    def tokenize(self, question: str, passages: List[str], titles: Optional[List[str]] = None) -> dict:
        raise NotImplementedError

    def to(self, device: torch.device):
        self._device = device

    def __len__(self) -> int:
        return len(self._tokenizer)

    def __getattr__(self, key: str) -> object:
        return getattr(self._tokenizer, key)
