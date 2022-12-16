import torch

from .base import BaseRerankerFramework
from ..models import PassageReranker
from ..tokenizers import PassageRerankerTokenizer


class PassageRerankerFramework(BaseRerankerFramework):

    def __init__(self, model: PassageReranker, tokenizer: PassageRerankerTokenizer,
                 config: dict):
        super().__init__(model=model,
                         tokenizer=tokenizer,
                         config=config)
        
        self.batch_size = 8

    def predict(self, question: str, passages: list, titles: list, config: dict) -> dict:
        scores = []
        for i in range(0, len(passages), self.batch_size):

            passages_sublist = passages[i:i+self.batch_size]
            titles_sublist = titles[i:i+self.batch_size] if titles else None

            batch = self.tokenizer.tokenize(question, passages_sublist, 
                                            titles_sublist)

            batch_scores = self.model(batch)
            batch_scores = batch_scores.view(-1)
            scores += batch_scores.tolist()

        return {
            "reranked_scores": scores
        }

