import torch

from ..tokenizers.base import BaseRerankerTokenizer


class BaseRerankerFramework(object):

    def __init__(self, model: torch.nn.Module, tokenizer: BaseRerankerTokenizer, config: dict):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        self.model.eval()

    def predict(self, question: str, passages: list, titles: list, config: dict) -> dict:
        raise NotImplementedError

    def to(self, device):
        self.model.to(device)
        self.tokenizer.to(device)