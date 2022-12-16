import jsonschema

DEFAULT_MODULE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "model": {"type": "string"},
        "tokenizer": {"type": "string"},
        "framework": {"type": "string"},
        "checkpoint": {"type": "string"},
    },
    "required": ["label", "model", "tokenizer", "framework", "checkpoint"]
}

RANKER_MODULE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "model": {"type": "string"},
        "tokenizer": {"type": "string"},
        "framework": {"type": "string"},
        "checkpoint": {"type": "string"},
        "database": {"type": "string"},
        "index": {"type": "string"},
    },
    "required": ["label", "framework", "database"]
}

PASSAGE_RERANKER_SCHEMA = DEFAULT_MODULE_SCHEMA
READER_SCHEMA = DEFAULT_MODULE_SCHEMA

STARTUP_CONFIGURATION_SCHEMA = {
    "type": "object",
    "properties": {
        "use_gpu": {"type": "boolean"},
        "cache_dir": {"type": "string"},
        "retriever_models": {
            "type": "array",
            "items": RANKER_MODULE_SCHEMA
        },
        "passage_reranker_models": {
            "type": "array",
            "items": PASSAGE_RERANKER_SCHEMA
        },
        "reader_models": {
            "type": "array",
            "items": READER_SCHEMA
        },
        "aggregation": {
            "type": "object"
        }
    },
    "required": ["use_gpu", "cache_dir", "retriever_models", "passage_reranker_models", "reader_models",
                 "aggregation"]
}


class StartupConfiguration(object):

    def __init__(self, config: dict):
        self._config = config
        self.validate(self._config)

    @property
    def schema(self):
        return STARTUP_CONFIGURATION_SCHEMA

    def validate(self, config: dict):
        jsonschema.validate(instance=config, schema=self.schema)

    def __getitem__(self, key):
        return self._config.get(key)
