#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Created on 16.01.24
Script for running API.


:author:     Martin Dočekal
"""
import json
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI
from contextlib import asynccontextmanager

from pydantic import BaseModel

from masapiqa.checkpoint import Checkpoint
from masapiqa.module.reranker.frameworks import PassageRerankerFramework
from masapiqa.module.reranker.models.reranker import PassageRerankerWrapper
from masapiqa.module.reranker.tokenizers import PassageRerankerTokenizer

reranker = None
config = None

SCRIPT_PATH_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for starting and stopping the model.
    """

    # this code is done on startup
    global reranker
    global config
    with open(SCRIPT_PATH_DIR / "config.json", "rt") as f:
        config = json.load(f)
    ckpt_config = Checkpoint.load_config(config["checkpoint"])

    ckpt_config["cache_dir"] = config["cache_dir"]

    model = PassageRerankerWrapper(ckpt_config)
    Checkpoint.load_model(model, config["checkpoint"])

    reranker = PassageRerankerFramework(
        model=model,
        tokenizer=PassageRerankerTokenizer(ckpt_config),
        config=ckpt_config
    )

    yield
    # this code is done on shutdown


app = FastAPI(lifespan=lifespan)


class RerankRequest(BaseModel):
    """
    Request for reranking.
    """
    question: str  # This is the question for which we want to rerank passages.
    passages: List[str]  # List of passages that should be reranked.
    titles: List[str]  # List of titles for passages.


@app.get("/rerank")
async def rerank(request: RerankRequest) -> List[float]:
    """
    Reranks passages according to the question.

    :param request: Request for reranking.
    :return: List of scores for each passage.
    """

    return reranker.predict(request.question, request.passages, request.titles, config)["reranked_scores"]
