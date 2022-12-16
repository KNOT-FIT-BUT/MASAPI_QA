import copy
import torch
import torch.nn.functional as F

from scalingqa.generative_reader.dataset.fid_generative_reader_dataset import FusionInDecoderDataset
from torch.nn import DataParallel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput

from .base import BaseAbstractiveReaderFramework


def get_model(m):
    if type(m) == DataParallel:
        return m.module
    return m


class T5FusionInDecoderFramework(BaseAbstractiveReaderFramework):

    def __init__(self, model, tokenizer, config):
        super().__init__(model, tokenizer, config)

        self._model.resize_token_embeddings(len(self._tokenizer))

    @torch.no_grad()
    def predict(self, question, passages, titles, config):
        """
        Runs generative prediction using greedy search
        """
        self._model.eval()
        b = self._tokenizer.tokenize(question, passages, titles)

        include_passage_masks = self._config["fusion_strategy"] == "passages"

        # FiD operates only in batch size 1
        b["doc_mask"] = b["doc_mask"] if include_passage_masks else None
        concatenated_encoder_output, concatenated_encoder_attention = self._model(input_ids=b["src"],
                                                                                  attention_mask=b["src_mask"],
                                                                                  encode_only=True)
        concatenated_encoder_output_copy = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=copy.deepcopy(concatenated_encoder_output['last_hidden_state']))
        concatenated_encoder_attention_copy = copy.deepcopy(concatenated_encoder_attention)

        tensorised_answers = get_model(self._model).generate(input_ids=concatenated_encoder_attention,
                                                             # num_beams=5,
                                                             # num_return_sequences=5,
                                                             attention_mask=concatenated_encoder_attention,
                                                             encoder_outputs=concatenated_encoder_output,
                                                             decoder_start_token_id=b["target"][0])

        score = 0.
        tensorized_answer = tensorised_answers[0]

        tensorized_answer_mask = torch.ones(tensorized_answer.size(), device=self._tokenizer._device)
        # the decoder inference needs to be run twice for 4.0.x transformers :(
        outputs: Seq2SeqLMOutput = self._model(input_ids=None,
                                               attention_mask=concatenated_encoder_attention_copy,
                                               encoder_outputs=concatenated_encoder_output_copy,
                                               passage_mask=b["doc_mask"],
                                               decoder_input_ids=tensorized_answer.unsqueeze(0)[:, :-1],
                                               decoder_attention_mask=tensorized_answer_mask.unsqueeze(0)[:, :-1])
        lm_logits = outputs.logits

        labels = tensorized_answer[1:]
        logprobs = - F.cross_entropy(lm_logits.view(-1, get_model(self._model).config.vocab_size), labels,
                                     reduction='none')
        logprobs[labels == self._tokenizer._tokenizer.pad_token_id] = 0.
        score = logprobs.sum().item()

        predicted_answer = self._tokenizer.decode(tensorized_answer, skip_special_tokens=True)

        if predicted_answer == "":
            predicted_answer = "I don't know the answer. :("

        return {
            "answers": [predicted_answer],
            "scores": [score]
        }

    @torch.no_grad()
    def rerank(self, question, answers, passages, titles, config):

        self._model.eval()
        b = self._tokenizer.tokenize(question, passages, titles)
        include_passage_masks = self._config["fusion_strategy"] == "passages"

        # encode passages
        result = self._model(input_ids=b["src"],
                             attention_mask=b["src_mask"],
                             encode_only=True)
        concatenated_encoder_output, concatenated_encoder_attention = result
        # tokenize & numericalize answers from extractive reader
        tokenized_answers = FusionInDecoderDataset.assemble_target_sequences(
            answers=answers,
            tokenizer=self._tokenizer._tokenizer)
        answer_masks = [[1] * len(a) for a in tokenized_answers]

        # rather do this in for cycle, to not further increase memory complexity
        scores = []
        for ans, mask in zip(tokenized_answers, answer_masks):
            tensorized_answer = torch.LongTensor(ans).to(self.tokenizer._device).unsqueeze(0)
            tensorized_answer_mask = torch.LongTensor(mask).to(self.tokenizer._device).unsqueeze(0)

            b["doc_mask"] = b["doc_mask"] if include_passage_masks else None

            concatenated_encoder_output_copy = BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=copy.deepcopy(concatenated_encoder_output['last_hidden_state']))
            concatenated_encoder_attention_copy = copy.deepcopy(concatenated_encoder_attention)

            lm_logits = self._model(input_ids=None, attention_mask=concatenated_encoder_attention_copy,
                                    passage_mask=b["doc_mask"], encoder_outputs=concatenated_encoder_output_copy,
                                    decoder_input_ids=tensorized_answer[:, :-1].contiguous(),
                                    decoder_attention_mask=tensorized_answer_mask[:, :-1].contiguous())[0]

            labels = tensorized_answer[:, 1:].reshape(-1)
            logprobs = - F.cross_entropy(lm_logits.view(-1, get_model(self._model).config.vocab_size), labels,
                                         reduction='none')
            logprobs[labels == self._tokenizer.pad_token_id] = 0.
            scores.append(logprobs.sum().item())

        return {
            "reranked_scores": scores
        }
