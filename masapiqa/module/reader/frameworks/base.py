import torch

from ..tokenizers.base import BaseReaderTokenizer


class BaseReaderFramework(object):

    def __init__(self, model: torch.nn.Module, tokenizer: BaseReaderTokenizer, config):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config

        self._model.eval()

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config(self):
        return self._config

    @property
    def is_extractive(self):
        return False

    @property
    def is_abstractive(self):
        return False

    @torch.no_grad()
    def predict(self, question, passages, titles, config):
        raise NotImplementedError

    def half(self):
        self._model.half()

    def to(self, device):
        self._model.to(device)
        self._tokenizer.to(device)


class BaseExtractiveReaderFramework(BaseReaderFramework):
    @property
    def is_extractive(self):
        return True


class BaseAbstractiveReaderFramework(BaseReaderFramework):
    @property
    def is_abstractive(self):
        return True

    @torch.no_grad()
    def rerank(self, question, answers, passages, titles, config):
        raise NotImplementedError
