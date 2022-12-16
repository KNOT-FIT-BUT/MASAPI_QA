import torch


class BaseRetrieverTokenizer(object):

    def __init__(self, config):
        self._model_config = config
        self._tokenizer = None
        self._max_length = None
        self._device = torch.device("cpu")

    @property
    def model_config(self):
        return self._model_config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_length(self):
        return self._max_length

    def tokenize(self, question: str) -> dict:
        raise NotImplementedError

    def to(self, device):
        self._device = device
