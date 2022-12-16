import torch
from scalingqa.extractivereader.datasets.pass_database import PassDatabase
from torch import Tensor

from .base import BaseRetrieverFramework
from ..models import R2D2Encoder
from ..tokenizers import R2D2EncoderTokenizer


class R2D2EncoderFramework(BaseRetrieverFramework):

    def __init__(self, model: R2D2Encoder, tokenizer: R2D2EncoderTokenizer, index: Tensor, database_path: str):
        self.model = model
        self.tokenizer = tokenizer
        self.passage_embeddings = index
        self.database = PassDatabase(database_path)
        self.model.eval()

    def predict(self, question: str, config: dict) -> dict:
        features = self.tokenizer.tokenize(question)

        encoded_queries = self.model.encode(features["input_ids"],
                                            features["token_type_ids"],
                                            features["attention_mask"])

        encoded_queries = encoded_queries.to(self.passage_embeddings.device)

        scores = self.model.get_scores(q=encoded_queries,
                                       embeddings=self.passage_embeddings,
                                       targets=None,
                                       return_only_scores=True)

        # 1 x K
        top_predicted_scores, top_predicted_indices = torch.topk(scores, dim=1, k=config["top_k"])

        passages = []
        titles = []

        with self.database:
            for idx in top_predicted_indices[0].tolist():
                _, title, context = self.database[idx]
                titles.append(title)
                passages.append(context)

        return {
            "question": features["raw_question"],
            "indices": top_predicted_indices[0].tolist(),
            "scores": top_predicted_scores[0].tolist(),
            "passages": passages,
            "titles": titles
        }

    def to(self, device):
        self.model.to(device)
        self.tokenizer.to(device)

