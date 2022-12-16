
import torch

from scalingqa.generative_reader.dataset.fid_generative_reader_dataset import FusionInDecoderDataset
from transformers import AutoTokenizer
from typing import List, Optional, Tuple

from .base import BaseReaderTokenizer


class T5FusionInDecoderTokenizer(BaseReaderTokenizer):

    def __init__(self, config, cache_dir=None):

        tokenizer = AutoTokenizer.from_pretrained(config['encoder'],
                                                  cache_dir=cache_dir, use_fast=True)

        super().__init__(tokenizer, config)

        self._tokenizer.question_special_token = '<question>'
        self._tokenizer.passage_special_token = '<passage>'
        self._tokenizer.title_special_token = '<title>'
        self._tokenizer.add_tokens(
            [self._tokenizer.question_special_token, self._tokenizer.passage_special_token,
             self._tokenizer.title_special_token], special_tokens=True)

        self._context_size = config["context_size"]
        self._preprocessing_truncation = config["preprocessing_truncation"]

    def tokenize(self, question, passages, titles=None):

        titles_tokens = []
        titles_raw = []

        top_k_passages_tokens = []
        top_k_passages_raw = []

        # take rest of the passages as top-k, if available
        for _, (p, t) in enumerate(zip(passages, titles)):
            if len(top_k_passages_tokens) == self._context_size:
                break

            # sometimes, there can be duplicate passages inside text (e.g. DPR passages), remove these
            if t in titles_raw and p in top_k_passages_raw:
                continue

            titles_tokens.append(self._tokenizer.encode(t, add_special_tokens=False))
            titles_raw.append(t)
            passage = " " + p
            tokenized_passage = self._tokenizer.encode(passage, add_special_tokens=False)
            top_k_passages_tokens.append(tokenized_passage)
            top_k_passages_raw.append(passage)

        assert len(top_k_passages_tokens) == self._context_size, f"Passages: {len(top_k_passages_tokens)}, Context size: {self._context_size}"
        question_r = question + " ?" if not question.endswith("?") else question
        question_tokens = self._tokenizer.encode(question_r, add_special_tokens=False)

        input_sequences, document_masks = FusionInDecoderDataset.assemble_input_sequences(question=question_tokens,
                                                                                          passages=top_k_passages_tokens,
                                                                                          titles=titles_tokens,
                                                                                          tokenizer=self._tokenizer,
                                                                                          max_passage_length=self._max_length,
                                                                                          preprocessing_truncation=self._preprocessing_truncation)

        input_sequences = self._pad_nested_field(input_sequences, self._tokenizer.pad_token_id)
        src_mask = self._pad_nested_field([[1] * len(x) for x in input_sequences], 0)
        doc_mask = self._pad_nested_field(document_masks, 1.)

        return {
            "src": torch.tensor(input_sequences, device=self._device),
            "src_mask": torch.tensor(src_mask, device=self._device),
            "doc_mask": torch.tensor(doc_mask, device=self._device),
            "target": torch.tensor([self._tokenizer.pad_token_id], device=self._device),
            "target_mask": torch.tensor([1], device=self._device)
        }

    def _pad_nested_field(self, sequences, pad):
        max_ = max(len(item) for item in sequences)

        padded_sequences = [s + [pad] * (max_ - len(s)) for s in sequences]
        return padded_sequences
