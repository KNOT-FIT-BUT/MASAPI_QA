import torch

from .base import BaseExtractiveReaderFramework


class R2D2ExtractiveReaderFramework(BaseExtractiveReaderFramework):

    def __init__(self, model, tokenizer, config):
        super().__init__(model, tokenizer, config)

    @torch.no_grad()
    def predict(self, question, passages, titles, config):

        batch = self._tokenizer.tokenize(question, passages, titles)
        answers, scores, passage_ids, character_offsets = self._answer_extract(passages=passages,
                                                                               max_span_len=config["reader_max_tokens_for_answer"],
                                                                               top_k=config["reader_top_k_answers"],
                                                                               **batch)

        return {
            "answers": answers,
            "passage_indices": passage_ids,
            "char_offsets": character_offsets,
            "scores": scores
        }

    def _answer_extract(self, passages, inputSequences, inputSequencesAttentionMask,
                        passageMask, longestPassage, tokenType, tokensOffsetMap,
                        max_span_len, top_k):

        startScores, endScores, jointScore, selectionScore = self.model(inputSequences=inputSequences,
                                                                        inputSequencesAttentionMask=inputSequencesAttentionMask,
                                                                        passageMask=passageMask,
                                                                        longestPassage=longestPassage,
                                                                        tokenType=tokenType)

        logProbs = self.model.scores2logSpanProb(startScores, endScores, jointScore, selectionScore)
        sortedLogProbs, sortedLogProbsInd = torch.sort(logProbs.flatten(), descending=True)

        answers, scores, passageIds, characterOffsets = [], [], [], []

        for i, (predictLogProb, predictedOffset) in enumerate(zip(sortedLogProbs.tolist(), sortedLogProbsInd.tolist())):
            predictedPassageOffset = predictedOffset // (logProbs.shape[1] ** 2)

            spanStartOffset = predictedOffset % (logProbs.shape[1] ** 2)
            spanEndOffset = spanStartOffset
            spanStartOffset //= logProbs.shape[1]
            spanEndOffset %= logProbs.shape[1]

            start = tokensOffsetMap[predictedPassageOffset][spanStartOffset][0] - 1
            end = tokensOffsetMap[predictedPassageOffset][spanEndOffset][1] - 1
            span = passages[predictedPassageOffset][start:end]

            if max_span_len is None or len(span.split(" ")) <= max_span_len or top_k - len(answers) >= len(
                    sortedLogProbs) - i:
                answers.append(span)
                scores.append(predictLogProb)
                passageIds.append(predictedPassageOffset)
                characterOffsets.append((start, end))

            if len(answers) == top_k:  # complete
                break

        return answers, scores, passageIds, characterOffsets
