
import torch

from transformers import AutoTokenizer
from typing import List, Optional, Tuple

from .base import BaseReaderTokenizer


class R2D2ExtractiveReaderTokenizer(BaseReaderTokenizer):

    def __init__(self, config, cache_dir=None):
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_type'],
                                                  cache_dir=cache_dir, use_fast=True)
        super().__init__(tokenizer, config)
        self.articleTitle = config['include_doc_title']

    @property
    def articleTitle(self) -> bool:
        """
        True causes that the title will be added into the input.
        """
        return self._articleTitle

    def is_extractive(self):
        return True

    def is_abstractive(self):
        return False

    @articleTitle.setter
    def articleTitle(self, use: bool):
        """
        Activates or deactivates title in the input.
        :param use: True use title in input. False otherwise.
        :type use: bool
        """

        self._articleTitle = use

        # the number of special tokens s probably changed
        self._numOfSpecTokInSeq = self._calcNumOfSpecTokInSeq()

    def tokenize(self, question, passages, titles=None):

        title_tokens = [] if self.articleTitle else None
        tokens2CharMap = []
        tokens = []

        for idx, (p, t) in enumerate(zip(passages, titles)):
            if self.articleTitle:
                title_tokens.append(self._tokenizer.encode(t, add_special_tokens=False))

            actPsg = " " + p
            tokenizationRes = self._tokenizer.encode_plus(actPsg, add_special_tokens=False,
                                                          return_offsets_mapping=True)
            tokens.append(tokenizationRes['input_ids'])
            tokens2CharMap.append(tokenizationRes['offset_mapping'])

        # we want to make input sequence with format (shown without special tokens):
        #   PASSAGE QUESTION <TITLE>
        q = self._tokenizer.encode(question, add_special_tokens=False)
        questions = [q.copy() for _ in range(len(tokens))]  # question should be the same for all inputs

        self._truncateBatch(questions=questions, passages=tokens, titles=title_tokens)

        longestPassage = max(len(t) for t in tokens)

        passageMask = torch.zeros(len(tokens), longestPassage, dtype=torch.bool)
        for i, t in enumerate(tokens):
            passageMask[i, :len(t)] = 1

        inputSequences, tokenTypeIds = self._assembleInputSequences(questions, tokens, title_tokens)

        inputSequences = self._tokenizer.pad({"input_ids": inputSequences})

        inputSequencesAttentionMask = torch.tensor(inputSequences["attention_mask"], device=self._device)

        inputSequences = torch.tensor(inputSequences["input_ids"], device=self._device)

        # let's pad the token types with value of last type id
        for tTypeIds in tokenTypeIds:
            tTypeIds += [tTypeIds[-1]] * (inputSequences.shape[1] - len(tTypeIds))
        tokenTypeIds = torch.tensor(tokenTypeIds, device=self._device)

        return {
            "inputSequences": inputSequences,
            "inputSequencesAttentionMask": inputSequencesAttentionMask,
            "passageMask": passageMask,
            "longestPassage": longestPassage,
            "tokenType": tokenTypeIds,
            "tokensOffsetMap": tokens2CharMap
        }

    def _calcNumOfSpecTokInSeq(self) -> int:
        """
        Calculates number of special tokens in an input sequence.
        :return: Number of special tokens in an input sequence.
        :rtype: int
        """

        assembled, _ = self._assembleInputSequences([[1]], [[2]], [[3]] if self.articleTitle else None)
        return len(assembled[0]) - (3 if self.articleTitle else 2)

    def _truncateBatch(self, questions: List[List[int]], passages: List[List[int]], titles: Optional[List[List[int]]]):
        """
        In place operation that truncates batch in order to get valid len of input sequence.
        The truncation is done in a way when we presume that each part of input (passage, question, [title]) can
        acquire at least int(max_input_len/number_of_parts) (fair distribution).
        If an input sequence is longer than it could be, we start truncation in order: title, passage than query
        until we get valid size input. We newer truncate each part bellow it's guaranteed len.
        :param questions: The question/query. X times the same so each sample haves its own.
            Control that it actually are same question is not done here.
        :type questions: List[int]
        :param passages: Retrieved passages.
        :type passages: List[List[int]]
        :param titles: Voluntary titles of passages. If none titles should not be in input sequence.
        :type titles: Optional[List[List[int]]]
        """

        fairLimitForSection = int((self._tokenizer.model_max_length - self._numOfSpecTokInSeq) / (2 if titles is None else 3))

        if titles is None:
            titles = [[]] * len(passages)

        # every batch input consists of:
        #   passage | question | title
        #       question is all the same for all samples in a batch
        #       passages and titles can differ in length

        for i, (actPassage, actQuestion, actTitle) in enumerate(zip(passages, questions, titles)):

            seqLen = self._numOfSpecTokInSeq + len(actPassage) + len(actQuestion) + len(actTitle)

            if seqLen > self._tokenizer.model_max_length:
                diff = seqLen - self._tokenizer.model_max_length

                for takeFrom in [titles, passages, questions]:
                    if len(takeFrom[i]) > fairLimitForSection:
                        take = min(diff, len(takeFrom[i]) - fairLimitForSection)
                        takeFrom[i] = takeFrom[i][:-take]
                        diff -= take

                    if diff == 0:
                        break

    def _assembleInputSequences(self, questions: List[List[int]], passages: List[List[int]],
                                titles: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Makes input sequence with format (shown without special tokens):
            PASSAGE QUESTION <TITLE>

        The title is voluntary.

        :param questions: The batch questions.
            Even we have the same question for all passages, we must provide it multiple times. Because after truncate
            their length can vary.
        :type questions: List[List[int]]
        :param passages: All passages in batch.
        :type passages: List[List[int]]
        :param titles: Corresponding titles to passages.
        :type titles: Optional[List[List[int]]]
        :return: Concatenated input sequences and the token type ids separating the two main parts of input.
        :rtype: Tuple[List[List[int]], List[List[int]]]
        """

        res = []
        tokenTypes = []

        for i, p in enumerate(passages):
            questionTitle = questions[i] if titles is None else questions[i] + [self._tokenizer.sep_token_id] + titles[i]

            seq = self._tokenizer.build_inputs_with_special_tokens(p, questionTitle)
            actTokTypes = self._tokenizer.create_token_type_ids_from_sequences(p, questionTitle)
            res.append(seq)
            tokenTypes.append(actTokTypes)

        return res, tokenTypes
