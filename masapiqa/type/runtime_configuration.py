"""
"""
import os
import h5py
import torch

from .startup_configuration import StartupConfiguration
from ..checkpoint import Checkpoint
from ..utility import lazy_unzip


RANKER_MODULE = __import__("masapiqa.module.retriever", globals(), locals(), ["models", "frameworks", "tokenizers"], 0)
PASSAGE_RERANKER_MODULE = __import__("masapiqa.module.reranker", globals(), locals(), ["models", "frameworks", "tokenizers"], 0)
READER_MODULE = __import__("masapiqa.module.reader", globals(), locals(), ["models", "frameworks", "tokenizers"], 0)


class RuntimeConfiguration(object):

    def __init__(self, config: StartupConfiguration):
        self.startup_config = config

        self.rankers = {}
        self.passage_rerankers = {}
        self.readers = {}

        self._half = False
        self._device = torch.device("cuda:0") if self.startup_config["use_gpu"] else torch.device("cpu")

        self._load_modules()

    @property
    def half(self):
        return self._half

    def _load_modules(self):
        self._load_aggregation_params()
        self._load_rankers()
        self._load_passage_rankers()
        self._load_readers()

    def _load_rankers(self):
        self.rankers = {}

        for cfg in self.startup_config["retriever_models"]:
            key = cfg["label"]

            if "checkpoint" in cfg:
                ckpt_config = Checkpoint.load_config(cfg["checkpoint"])
                ckpt_config["cache_dir"] = self.startup_config["cache_dir"]

            attributes = {"database_path": cfg["database"]}
            if "model" in cfg:
                model_cls = getattr(RANKER_MODULE.models, cfg["model"])
                attributes["model"] = model_cls(ckpt_config)
                Checkpoint.load_model(attributes["model"], cfg["checkpoint"])

            if "tokenizer" in cfg:
                tokenizer_cls = getattr(RANKER_MODULE.tokenizers, cfg["tokenizer"])
                attributes["tokenizer"] = tokenizer_cls(ckpt_config)

            if "index" in cfg:
                index_path = cfg["index"]
                attributes["index"] = self._get_index(index_path)

            framework_cls = getattr(RANKER_MODULE.frameworks, cfg["framework"])
            self.rankers[key] = framework_cls(**attributes)
            self.rankers[key].to(self._device)

            if self.half:
                self.rankers[key].half()

    def _load_passage_rankers(self):
        self.passage_rerankers = {}
        for cfg in self.startup_config["passage_reranker_models"]:
            key = cfg["label"]
            self.passage_rerankers[key] = self._build_module(PASSAGE_RERANKER_MODULE, cfg)

    def _load_readers(self):
        self.readers = {}
        for cfg in self.startup_config["reader_models"]:
            key = cfg["label"]
            self.readers[key] = self._build_module(READER_MODULE, cfg)

    def _build_module(self, module, config):
        ckpt_config = Checkpoint.load_config(config["checkpoint"])
        ckpt_config["cache_dir"] = self.startup_config["cache_dir"]

        model_cls = getattr(module.models, config["model"])
        tokenizer_cls = getattr(module.tokenizers, config["tokenizer"])
        framework_cls = getattr(module.frameworks, config["framework"])

        model = model_cls(ckpt_config)
        tokenizer = tokenizer_cls(ckpt_config)
        Checkpoint.load_model(model, config["checkpoint"])
        framework = framework_cls(model, tokenizer, ckpt_config)

        framework.to(self._device)

        if self.half:
            framework.half()

        return framework

    def _load_aggregation_params(self):
        self.aggregation_params = {}

        for r, rr, ext, gen, params in self.startup_config["aggregation"]["params"]:
            self.aggregation_params[(r, rr, ext, gen)] = params

    def _get_index(self, passage_embeddings_path):
        if passage_embeddings_path.endswith(".zip"):
            lazy_unzip(passage_embeddings_path)
            passage_embeddings_path = passage_embeddings_path[:-len(".zip")]

        h5p_tensor = h5py.File(passage_embeddings_path, 'r')['data'][()]
        passage_embeddings = torch.FloatTensor(h5p_tensor)
        del h5p_tensor
        return passage_embeddings
