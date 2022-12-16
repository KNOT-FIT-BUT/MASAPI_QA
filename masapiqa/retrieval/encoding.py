# -*- coding: UTF-8 -*-
""""
Created on 12.08.22

Tools for obtaining context embeddings.

:author:     Martin Dočekal
"""
from abc import abstractmethod, ABC
from typing import Sequence, List

import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRQuestionEncoder, \
    DPRPretrainedContextEncoder, PreTrainedTokenizer

import numpy as np


class Encoder(ABC):
    """
    Base class for all encoders providing embeddings for retrieval.
    """

    @abstractmethod
    def __call__(self, contexts: Sequence[str]) -> np.array:
        """
        Performs encoding of given contexts.

        :param contexts: context that should be encoded
        :return: Embeddings of context in form of matrix with shape CONT_LEN X FEATURES
        """
        ...


class TorchModelEncoder(Encoder):
    """
    Base class for encoder that use already trained torch model.
    """

    def __init__(self, model: torch.nn.Module, use_device: torch.device = torch.device("cpu")):
        """
        initialization of encoder

        :param model: torch model that would be used for encoding
        :param use_device: which device should be used for model
        """
        self.model = model
        self.model.to(use_device)

    @abstractmethod
    def create_inputs(self, contexts: Sequence[str]) -> Any:
        """
        Transforms sequence of contexts into inputs that could be passed into a model

        :param contexts: texts for transformation
        :return: inputs for a model
        """
        ...

    @abstractmethod
    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        transformation of inputs into embeddings

        :param inputs: inputs for a model
            shape INPUTS X TOKENS
        :return: embeddings of inputs
            shape INPUTS X FEATURES
        """
        ...

    @torch.no_grad()
    def __call__(self, contexts: Sequence[str]) -> np.array:
        inputs = self.create_inputs(contexts)
        return self.embed(inputs).numpy()


class DPREncoder(TorchModelEncoder):
    """
    Base class for all trained DPR based encoders.
    """

    def __init__(self, model: DPRPretrainedContextEncoder, tokenizer: PreTrainedTokenizer,
                 use_device: torch.device = torch.device("cpu")):
        """
        initialization of encoder

        :param model: torch model that would be used for encoding
        :param tokenizer: tokenizer that would be used for creating model inputs
        :param use_device: which device should be used for model
        """
        super().__init__(model, use_device)
        self.tokenizer = tokenizer

    def create_inputs(self, contexts: List[str]):
        return self.tokenizer(contexts, return_tensors="pt", padding=True, truncation=True)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).pooler_output


class DPRCtxEncoder(DPREncoder):
    """
    Trained DPR encoder of contexts. It was introduced in Dense Passage Retrieval for Open-Domain Question Answering:
        https://arxiv.org/abs/2004.04906
    """

    def __init__(self, use_device: torch.device = torch.device("cpu")):
        """
        initialization of encoder

        :param use_device: which device should be used for model
        """
        super().__init__(model=DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
                         tokenizer=DPRContextEncoderTokenizer.from_pretrained(
                             "facebook/dpr-ctx_encoder-single-nq-base"
                         ),
                         use_device=use_device)


class DPRQueEncoder(DPREncoder):
    """
    Trained DPR encoder of questions. It was introduced in Dense Passage Retrieval for Open-Domain Question Answering:
        https://arxiv.org/abs/2004.04906
    """
    def __init__(self, use_device: torch.device = torch.device("cpu")):
        """
        initialization of encoder

        :param use_device: which device should be used for model
        """
        super().__init__(model=DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
                         tokenizer=DPRQuestionEncoderTokenizer.from_pretrained(
                             "facebook/dpr-question_encoder-single-nq-base"
                         ),
                         use_device=use_device)

