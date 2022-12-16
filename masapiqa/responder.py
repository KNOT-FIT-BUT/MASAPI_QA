import copy
import logging
import time
from typing import Optional

import numpy as np

from .type.runtime_configuration import RuntimeConfiguration
from .type.startup_configuration import StartupConfiguration


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def log_softmax(x):
    return np.log(softmax(x))


def is_module_active(module: str, config: dict) -> bool:
    return module in config and config[module].get("model", None) is not None


def is_passage_reranking_active(config: dict) -> bool:
    return is_module_active("passage_reranker", config)


def is_answer_reranking_active(config: dict) -> bool:
    return is_extractive_reader_active(config) and config["extractive_reader"].get("generative_reranking", False)


def is_extractive_reader_active(config: dict) -> bool:
    return is_module_active("extractive_reader", config)


def is_abstractive_reader_active(config: dict) -> bool:
    return is_module_active("abstractive_reader", config)


def is_score_aggregation_active(config: dict) -> bool:
    return is_extractive_reader_active(config) and config.get("score_aggregation", False)


class OpenQAResponder(object):

    def __init__(self, config: RuntimeConfiguration):
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.runtime_config = config

    def predict(self, question: str, config: dict) -> dict:
        self.logger.debug("Start predict...")
        time_start = time.time()

        response = {
            "question": question
        }

        passages = self.retrieval(question, config)

        if is_passage_reranking_active(config):
            passages = self.passage_reranking(question,
                                              passages,
                                              config["passage_reranker"])

        response["ranker"] = passages

        if is_extractive_reader_active(config):
            response["extractive_reader"] = self.read(question,
                                                      passages,
                                                      config["extractive_reader"],
                                                      config["extractive_reader"]["top_passages"])

        if is_abstractive_reader_active(config):
            response["abstractive_reader"] = self.read(question,
                                                       passages,
                                                       config["abstractive_reader"],
                                                       config["abstractive_reader"]["top_passages"])

        if is_answer_reranking_active(config):
            if not is_extractive_reader_active(config) and is_abstractive_reader_active(config):
                raise Exception(
                    "If the answer reranking is activated, the abstractive and extractive reader must also be activated.")

            reranked_answer_scores = self.answer_reranking(question,
                                                           response["extractive_reader"],
                                                           passages,
                                                           config["abstractive_reader"])
            response["extractive_reader"] = reranked_answer_scores

        if is_score_aggregation_active(config):
            aggregated_scores = self.score_aggregation(response, config)
            response["extractive_reader"]["aggregated_scores"] = aggregated_scores

        time_end = time.time()

        self.logger.debug("Prediction finished after %s seconds.", time_end - time_start)

        return response

    def retrieval(self, question: str, config: dict) -> dict:
        self.logger.debug("Retrieves relevant passages to question '%s'.", question)
        time_start = time.time()

        ranker_config = config["retriever"]

        ranker = self.runtime_config.rankers[ranker_config["model"]]
        features = ranker.predict(question, ranker_config)

        time_end = time.time()
        self.logger.debug("Retriever finish after %s s.", time_end - time_start)

        return {
            "paragraphs": features["passages"],
            "titles": features["titles"],
            "ids": features["indices"],
            "scores": log_softmax(features["scores"]).tolist()
        }

    def passage_reranking(self, question: str, passages: dict, config: dict) -> dict:
        self.logger.debug("Reranks passages to question '%s'.", question)
        time_start = time.time()

        reranker = self.runtime_config.passage_rerankers[config["model"]]
        features = reranker.predict(question=question,
                                    passages=passages["paragraphs"],
                                    titles=passages["titles"],
                                    config=config)

        time_end = time.time()
        self.logger.debug("Passage reranking finish after %s s.", time_end - time_start)

        return {
            "paragraphs": passages["paragraphs"],
            "titles": passages["titles"],
            "ids": passages["ids"],
            "scores": passages["scores"],
            "reranked_scores": log_softmax(features["reranked_scores"]).tolist()
        }

    def read(self, question: str, passages: dict, reader_config: dict, top_k_passages: Optional[int]) -> dict:
        self.logger.debug("Reads %s passages.", len(passages["paragraphs"]))
        time_start = time.time()

        reader = self.runtime_config.readers[reader_config["model"]]

        passages = copy.deepcopy(passages)
        true_indices = list(range(len(passages["ids"])))

        if "reranked_scores" in reader_config:
            true_indices = sorted(true_indices, key=lambda idx: passages["reranked_scores"][idx], reverse=True)

            if top_k_passages is not None:
                true_indices = true_indices[:top_k_passages]

            passages["paragraphs"] = [passages["passages"][idx] for idx in true_indices]
            passages["titles"] = [passages["titles"][idx] for idx in true_indices]
            passages["ids"] = [passages["indices"][idx] for idx in true_indices]
            passages["scores"] = [passages["scores"][idx] for idx in true_indices]
            passages["reranked_scores"] = [passages["reranked_scores"][idx] for idx in true_indices]

        elif top_k_passages is not None:
            passages["paragraphs"] = passages["paragraphs"][:top_k_passages]
            passages["titles"] = passages["titles"][:top_k_passages]
            passages["ids"] = passages["ids"][:top_k_passages]
            passages["scores"] = passages["scores"][:top_k_passages]

        if top_k_passages is not None:
            self.logger.debug(f"Only uses top {len(passages['paragraphs'])} passages for reader: {reader_config['model']}.")

        if reader.is_extractive:
            key = "extractive_reader"
        elif reader.is_abstractive:
            key = "abstractive_reader"
        else:
            raise Exception("Unknown reader topology of '{}' model".format(
                reader_config["model"])
            )

        features = reader.predict(question, passages["paragraphs"],
                                  passages["titles"], reader_config)

        if "passage_indices" in features:
            features["passage_indices"] = [true_indices[idx] for idx in features["passage_indices"]]

        time_end = time.time()
        self.logger.debug("The %s finish after %i s.", key, time_end - time_start)
        return features

    def answer_reranking(self, question: str, spans: dict, passages: dict,
                         reader_config: dict) -> dict:
        self.logger.debug("Rerank %s passages.", len(spans["answers"]))
        time_start = time.time()

        reader = self.runtime_config.readers[reader_config["model"]]
        passages = copy.deepcopy(passages)
        spans = copy.deepcopy(spans)

        if not reader.is_abstractive:
            raise Exception("Only the abstractive reader can rerank the answers!")

        features = reader.rerank(question, spans["answers"], passages["paragraphs"], passages["titles"], reader_config)
        time_end = time.time()
        self.logger.debug("The answer reranking finish after %i s.", time_end - time_start)

        spans["reranked_scores"] = features["reranked_scores"]

        return spans

    def score_aggregation(self, response: dict, config: dict):
        ext_reader_scores = response["extractive_reader"]["scores"]

        if is_answer_reranking_active(config):
            reranked_reader_scores = response["extractive_reader"]["reranked_scores"]
            abs_reader_model = config["abstractive_reader"]["model"]
        else:
            reranked_reader_scores = [0.0] * len(ext_reader_scores)
            abs_reader_model = None

        passage_indeces = response["extractive_reader"]["passage_indices"]
        retriever_scores = response["ranker"]["scores"]
        retriever_scores = np.array([retriever_scores[i] for i in passage_indeces])
        # retriever_scores = log_softmax(retriever_scores).tolist()
        retriever_model = config["retriever"]["model"]

        if is_passage_reranking_active(config):
            reranker_scores = response["ranker"]["scores"]
            reranker_scores = np.array([reranker_scores[i] for i in passage_indeces])
            # reranker_scores = log_softmax(reranker_scores).tolist()
            reranker_model = config["passage_reranker"]["model"]
        else:
            reranker_scores = [0.0] * len(ext_reader_scores)
            reranker_model = None

        params = self.runtime_config.aggregation_params.get((retriever_model,
                                                             reranker_model,
                                                             config["extractive_reader"]["model"],
                                                             abs_reader_model))

        aggregated_scores = [
            params["w1"] * r + params["w2"] * rr + params["w3"] * a + params["w4"] * ra + params["bias"]
            for r, rr, a, ra in zip(retriever_scores,
                                    reranker_scores,
                                    ext_reader_scores,
                                    reranked_reader_scores)]

        return aggregated_scores

    def get_supported_models(self) -> dict:
        return {
            "retriever": list(self.runtime_config.rankers),
            "passage_reranker": list(self.runtime_config.passage_rerankers),
            "extractive_reader": [key for key, reader in self.runtime_config.readers.items() if reader.is_extractive],
            "abstractive_reader": [key for key, reader in self.runtime_config.readers.items() if reader.is_abstractive]
        }


TEST_CONFIG_01 = {
    "use_gpu": True,
    "checkpoint_dir": "/home/idocekal/onlineQA/data/checkpoint",
    "database_dir": "/home/idocekal/onlineQA/data/index",
    "index_dir": "/home/idocekal/onlineQA/data/index",
    "cache_dir": "/home/idocekal/onlineQA/data/cache",
    "retriever_models": [
        {
            "label": "dpr",
            "model": "LRMEncoder",
            "tokenizer": "LRMEncoderTokenizer",
            "framework": "LRMEncoderFramework",
            "checkpoint": "dpr_official_questionencoder_nq.pt",
            "database": "wiki2018_dpr_blocks_nq_open_pruned_P1700000.db",
            "index": "DPR_nqsingle_offic_electrapruner_nqopen_1700000_HALF.h5"
        }
    ],
    "passage_reranker_models": [
        {
            "label": "roberta",
            "model": "PassageReranker",
            "tokenizer": "PassageRerankerTokenizer",
            "framework": "PassageRerankerFramework",
            "checkpoint": "reranker_roberta-base_2021-02-25-17-27_athena19_HIT@25_0.8299645997487725_fp16.ckpt"
        }
    ],
    "reader_models": [
        {
            "label": "R2D2 reader",
            "model": "R2D2ExtractiveReader",
            "tokenizer": "R2D2ExtractiveReaderTokenizer",
            "framework": "R2D2ExtractiveReaderFramework",
            "checkpoint": "EXT_READER_ELECTRA_LARGE_B128_049_J24_HALF.pt"
        }
    ],
    "aggregation": {
        "params": []
    }
}

PREDICT_TEST_CONFIG_01 = {
    "question": "the last time la dodgers won the world series",
    "configuration": {
        "retriever": {
            "model": "dpr",
            "top_k": 100
        },
        "passage_reranker": {
            "model": "roberta"
        },
        "extractive_reader": {
            "model": "R2D2 reader",
            "reader_top_k_answers": 5,
            "reader_max_tokens_for_answer": 5,
            "generative_reranking": False
        }
    }
}

if __name__ == "__main__":
    startup_config = StartupConfiguration(TEST_CONFIG_01)
    runtime_config = RuntimeConfiguration(startup_config)

    responder = OpenQAResponder(runtime_config)
    print(responder.predict(PREDICT_TEST_CONFIG_01["question"], PREDICT_TEST_CONFIG_01["configuration"]))

