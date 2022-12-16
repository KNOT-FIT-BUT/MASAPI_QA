

from transformers import AutoConfig, AutoTokenizer, AutoModel

from scalingqa.reranker.models import BaselineReranker


class PassageRerankerWrapper(BaselineReranker):

    def __init__(self, config):
        model_config = AutoConfig.from_pretrained(config["encoder"], 
                                                  cache_dir=config["cache_dir"])
        encoder = AutoModel.from_config(model_config)

        super().__init__(model_config, encoder)