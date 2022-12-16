# -*- coding: UTF-8 -*-
"""
Created on 07.10.22
Configuration loader.

:author:     Martin Dočekal
"""
from pathlib import Path
from typing import Dict, Sequence, Callable

from windpyutils.config import Config


class MASAPIQAConfig(Config):
    """
    Configuration for MASAPI_QA pipeline.
    """

    def validate(self, config: Dict):
        """
        This validation method is here mainly to ensure proper translation of relative paths.

        :param config: user config
        """

        if "use_gpu" not in config or not isinstance(config["use_gpu"], bool):
            raise ValueError("You must provide valid [use_gpu].")

        if "cache_dir" not in config or not isinstance(config["cache_dir"], str):
            raise ValueError("You must provide valid [cache_dir].")
        config["cache_dir"] = self.translate_file_path(config["cache_dir"])

        for section in ["retriever_models", "passage_reranker_models", "reader_models", "aggregation",
                        "open_domain_config"]:
            if section not in config:
                raise ValueError(f"The [{section}] section is missing.")

        self.validate_module_sequence(config["retriever_models"],
                                      [self.validate_r2d2_retriever, self.validate_pyserini_retriever],
                                      "[retriever_models]")
        self.validate_module_sequence(config["passage_reranker_models"],
                                      [self.validate_model],
                                      "[passage_reranker_models]")
        self.validate_module_sequence(config["reader_models"],
                                      [self.validate_model],
                                      "[reader_models]")

        self.validate_open_domain(config["open_domain_config"], "[open_domain_config]")
        self.validate_on_demand(config["on_demand"], "[on_demand]")

    def validate_open_domain(self, part_of_config: Dict, prefix: str):
        for section in ["retriever", "passage_reranker", "extractive_reader"]:
            if section not in part_of_config:
                raise ValueError(f"The {prefix}[{section}] section is missing.")

        if not isinstance(part_of_config["retriever"]["model"], str):
            raise ValueError(f"The {prefix}[retriever][model] should be string.")
        if not isinstance(part_of_config["retriever"]["top_k"], int) and part_of_config["retriever"]["top_k"] <= 0:
            raise ValueError(f"The {prefix}[retriever][top_k] should be positive integer.")

        if not isinstance(part_of_config["passage_reranker"]["model"], str):
            raise ValueError(f"The {prefix}[passage_reranker][model] should be string.")

        if "extractive_reader" not in part_of_config and "abstractive_reader" not in part_of_config:
            raise ValueError(f"Provide configuration for an extractive or abstractive reader.")

        if "extractive_reader" in part_of_config:
            if not isinstance(part_of_config["extractive_reader"]["model"], str):
                raise ValueError(f"The {prefix}[extractive_reader][model] should be string.")
            if not isinstance(part_of_config["extractive_reader"]["reader_top_k_answers"], int) and \
                    part_of_config["extractive_reader"]["reader_top_k_answers"] <= 0:
                raise ValueError(f"The {prefix}[extractive_reader][reader_top_k_answers] should be positive integer.")
            if not isinstance(part_of_config["extractive_reader"]["reader_max_tokens_for_answer"], int) and \
                    part_of_config["extractive_reader"]["reader_max_tokens_for_answer"] <= 0:
                raise ValueError(
                    f"The {prefix}[extractive_reader][reader_max_tokens_for_answer] should be positive integer.")
            if not isinstance(part_of_config["extractive_reader"]["generative_reranking"], bool):
                raise ValueError(f"The {prefix}[extractive_reader][generative_reranking] should be boolean.")

            # voluntary
            if "top_passages" in part_of_config["extractive_reader"]:
                if not isinstance(part_of_config["extractive_reader"]["top_passages"], int):
                    raise ValueError(f"The voluntary {prefix}[extractive_reader][top_passages] should be int.")
            else:
                part_of_config["extractive_reader"]["top_passages"] = None

        if "abstractive_reader" in part_of_config:
            if not isinstance(part_of_config["abstractive_reader"]["model"], str):
                raise ValueError(f"The {prefix}[abstractive_reader][model] should be string.")

            # voluntary
            if "top_passages" in part_of_config["abstractive_reader"]:
                if not isinstance(part_of_config["abstractive_reader"]["top_passages"], int):
                    raise ValueError(f"The voluntary {prefix}[abstractive_reader][top_passages] should be int.")
            else:
                part_of_config["abstractive_reader"]["top_passages"] = None

    def validate_on_demand(self, part_of_config: Dict, prefix: str):
        self.validate_open_domain(part_of_config, prefix)
        if not isinstance(part_of_config["retriever"]["batch"], int) and part_of_config["retriever"]["batch"] <= 0:
            raise ValueError(f"The {prefix}[retriever][batch] should be positive integer.")
        if not isinstance(part_of_config["retriever"]["threads"], int) and \
                (part_of_config["retriever"]["threads"] <= 0 and part_of_config["retriever"]["threads"] != -1):
            raise ValueError(f"The {prefix}[retriever][threads] should be positive integer or -1.")

    def validate_module_sequence(self, seq: Sequence[Dict], validators: Sequence[Callable[[Dict, str], None]],
                                 prefix: str):

        for i, module in enumerate(seq):
            last_exception = None
            for validator in validators:
                try:
                    validator(module, prefix + f"[{i}]")
                    break
                except Exception as e:
                    last_exception = e
            else:
                raise ValueError(f"Unknown module configuration for {prefix}[{i}]\n" + str(last_exception))

    def validate_r2d2_retriever(self, config: Dict, prefix: str):
        if "label" not in config or not isinstance(config["label"], str):
            raise ValueError(f"You must provide valid {prefix}[label].")

        if "model" not in config or not isinstance(config["model"], str):
            raise ValueError(f"You must provide valid {prefix}[model].")

        if "tokenizer" not in config or not isinstance(config["tokenizer"], str):
            raise ValueError(f"You must provide valid {prefix}[tokenizer].")

        if "framework" not in config or not isinstance(config["framework"], str):
            raise ValueError(f"You must provide valid {prefix}[framework].")

        if "checkpoint" not in config or not isinstance(config["checkpoint"], str):
            raise ValueError(f"You must provide valid {prefix}[checkpoint].")
        config["checkpoint"] = self.translate_file_path(config["checkpoint"])

        if "database" not in config or not isinstance(config["database"], str):
            raise ValueError(f"You must provide valid {prefix}[database].")
        config["database"] = self.translate_file_path(config["database"])

        if "index" not in config or not isinstance(config["index"], str):
            raise ValueError(f"You must provide valid {prefix}[index].")
        config["index"] = self.translate_file_path(config["index"])

    def validate_pyserini_retriever(self, config: Dict, prefix: str):
        if "label" not in config or not isinstance(config["label"], str):
            raise ValueError(f"You must provide valid {prefix}[label].")

        if "framework" not in config or not isinstance(config["framework"], str):
            raise ValueError(f"You must provide valid {prefix}[framework].")

        if "database" not in config or not isinstance(config["database"], str):
            raise ValueError(f"You must provide valid {prefix}[database].")
        config["database"] = self.translate_file_path(config["database"])

    def validate_model(self, config: Dict, prefix: str):
        if "label" not in config or not isinstance(config["label"], str):
            raise ValueError(f"You must provide valid {prefix}[label].")

        if "model" not in config or not isinstance(config["model"], str):
            raise ValueError(f"You must provide valid {prefix}[model].")

        if "tokenizer" not in config or not isinstance(config["tokenizer"], str):
            raise ValueError(f"You must provide valid {prefix}[tokenizer].")

        if "checkpoint" not in config or not isinstance(config["checkpoint"], str):
            raise ValueError(f"You must provide valid {prefix}[checkpoint].")
        config["checkpoint"] = self.translate_file_path(config["checkpoint"])


