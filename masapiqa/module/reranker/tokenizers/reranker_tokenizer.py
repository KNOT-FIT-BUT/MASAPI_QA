from transformers import AutoTokenizer
from .base import BaseRerankerTokenizer
import math

from scalingqa.reranker.datasets import BaselineRerankerQueryBuilder


class PassageRerankerTokenizer(BaseRerankerTokenizer):

    def __init__(self, config):
        super().__init__(config)

        self._tokenizer = AutoTokenizer.from_pretrained(config['encoder'],
                                                        do_lower_case=True,
                                                        cache_dir=config["cache_dir"])

        if "max_length" not in config or not config["max_length"] :
            self._max_length = 256
        else:
            self._max_length = config["max_length"]

        self._query_builder = BaselineRerankerQueryBuilder(self._tokenizer, 
                                                           max_seq_length=self._max_length)

    def tokenize(self, question: str, passages: [str], titles: [str]) -> dict:
        title_and_context_pair = [(titles[i], passages[i]) for i in range(len(passages))]

        batch = self._query_builder(question, title_and_context_pair, False)
        batch = {key: batch[key].to(self._device) for key in ["input_ids", "attention_mask"]}

        return batch
