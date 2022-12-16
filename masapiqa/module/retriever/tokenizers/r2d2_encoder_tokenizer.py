from transformers import AutoTokenizer
from .base import BaseRetrieverTokenizer


class R2D2EncoderTokenizer(BaseRetrieverTokenizer):

    def __init__(self, config):
        super().__init__(config)

        self._tokenizer = AutoTokenizer.from_pretrained(config['model_type'],
                                                        do_lower_case=True,
                                                        cache_dir=config["cache_dir"])

        if "max_length" not in config:
            self._max_length = 512

    def tokenize(self, question: str) -> dict:
        features = self.tokenizer.encode_plus(question,
                                              add_special_tokens=True,
                                              return_token_type_ids=True,   # Why there was False?
                                              truncation=True,
                                              max_length=self._max_length,
                                              return_tensors="pt")

        features = {key: value.to(self._device) for key, value in features.items()}

        features["raw_question"] = question

        return features
