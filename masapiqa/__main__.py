# -*- coding: UTF-8 -*-
""""
Created on 12.08.22

:author:     Martin Dočekal
"""
import argparse
import json
import logging
import multiprocessing
import os
import shutil
import sys
import time
from contextlib import nullcontext
from math import ceil
from pathlib import Path
from subprocess import Popen
from typing import Generator, Tuple, Sequence, Dict, TextIO, Optional

import numpy as np
from datasets import load_dataset
from pyserini.eval.evaluate_dpr_retrieval import SimpleTokenizer, Tokenizer
from pyserini.search import LuceneSearcher, FaissSearcher, DprQueryEncoder
from tqdm import tqdm
from windpyutils.args import ExceptionsArgumentParser, ArgumentParserError

from masapiqa.config import MASAPIQAConfig
from masapiqa.database import Database
from masapiqa.evaluate import evaluate_predictions
from masapiqa.responder import OpenQAResponder
from masapiqa.type.runtime_configuration import RuntimeConfiguration
from masapiqa.type.startup_configuration import StartupConfiguration

DEFAULT_CONFIG_PATH = os.path.dirname(os.path.realpath(__file__)) + "/data/default_config.py"


class ArgumentsManager(object):
    """
    Parsers arguments for script.
    """

    @classmethod
    def parse_args(cls):
        """
        Performs arguments parsing.

        :param cls: arguments class
        :returns: Parsed arguments.
        """

        parser = ExceptionsArgumentParser(description="QA in MASAPI project.")

        subparsers = parser.add_subparsers()

        ask_parser = subparsers.add_parser("ask", help="Allows to ask a question.")
        ask_parser.add_argument("question", help="Question you want to ask.", type=str)
        ask_parser.add_argument("-c", "--config", help="Path to configuration that should be used, "
                                                       "else uses default one.", type=str)
        ask_parser.add_argument("-t", "--text", help="Path to text file or a text itself that should be used for "
                                                     "answering question. If omitted the open domain variant is used.",
                                type=str)
        ask_parser.add_argument("--no_title", help="By default it is expected that first line is a title. "
                                                   "If you use this flag the title is not extracted and is assumed to "
                                                   "be empty string. "
                                                   "This flag make sense only when combined with --text.",
                                action="store_true")
        ask_parser.set_defaults(func=ask)

        huggingface_create_db_parser = subparsers.add_parser("huggingface_create_db",
                                                             help="Creates database with content to ask questions on from hugginface dataset.")
        huggingface_create_db_parser.add_argument("result", help="path to folder where results will be saved", type=str)
        huggingface_create_db_parser.add_argument("path",
                                                  help="Path or name of the huggingface dataset.",
                                                  type=str)
        huggingface_create_db_parser.add_argument("split",
                                                  help="Which split of the data to load.",
                                                  type=str)
        huggingface_create_db_parser.add_argument("id_field",
                                                  help="Name of field with id.",
                                                  type=str)
        huggingface_create_db_parser.add_argument("text_field",
                                                  help="Name of field with text.",
                                                  type=str)
        huggingface_create_db_parser.add_argument("index_type", help="Which index type should be created.",
                                                  choices=['BM25', 'DPR'])
        huggingface_create_db_parser.add_argument("--title_field",
                                                  help="Dataset field that could be used as title. "
                                                       "If not set the title will be empty string.",
                                                  type=str, default=None)
        huggingface_create_db_parser.add_argument("-n", "--name",
                                                  help="Defining the name of the dataset configuration.",
                                                  type=str, default=None)
        huggingface_create_db_parser.add_argument("--threads", help="Number of threads used for index creation when "
                                                                    "BM25 is used.",
                                                  default=-1, type=int)
        huggingface_create_db_parser.add_argument("--batch", help="GPU batch size when dense index is used.",
                                                  default=8, type=int)
        huggingface_create_db_parser.add_argument("-c", "--config", help="Path to configuration that should be used, "
                                                       "else uses default one.", type=str)
        huggingface_create_db_parser.set_defaults(func=huggingface_create_db)

        single_query_retrieval_parser = subparsers.add_parser("single_query_retrieval",
                                                              help="Performs single query search on database of your choice.")
        single_query_retrieval_parser.add_argument("database", help="path to folder where database is saved", type=str)
        single_query_retrieval_parser.add_argument("query", help="your query", type=str)
        single_query_retrieval_parser.add_argument("-k", help="top k results will be selected", type=int, default=3)
        single_query_retrieval_parser.set_defaults(func=single_query_retrieval)

        evaluate_parser = subparsers.add_parser("evaluate", help="Evaluates system on provided dataset.")
        evaluate_parser.add_argument("dataset", help="Path to dataset for evaluation. It is expexted that it is a jsonl file with question and answer field", type=str)
        evaluate_parser.add_argument("-r", "--results", help="path where results should be saved", type=str)
        evaluate_parser.add_argument("-t", "--text", help="Path to text file or a text itself that should be used for "
                                                     "answering questions. If omitted the open domain variant is used.", type=str)
        evaluate_parser.add_argument("-c", "--config", help="Path to configuration that should be used, "
                                                       "else uses default one.", type=str)
        evaluate_parser.set_defaults(func=evaluate)

        if len(sys.argv) < 2:
            parser.print_help()
            return None
        try:
            parsed = parser.parse_args()

        except ArgumentParserError as e:
            parser.print_help()
            print("\n" + str(e), file=sys.stdout, flush=True)
            return None

        return parsed


def split_doc(doc: str, segment_size: int = 100, tokenizer: Tokenizer = SimpleTokenizer()) -> Generator[
    Tuple[str, int], None, None]:
    """
    Split document using heuristic where, following the approach of DPR, passages are split into 100 word windows.

    :param doc: content that should be splited
    :param segment_size: max size of segment in number of tokens
    :param tokenizer: specifies tokenizer that should be used
    :return: generator of splits
    """

    tokenized = tokenizer.tokenize(doc)
    words = tokenized.data
    start_split_offset = 0

    for i in range(ceil(len(words) / segment_size)):
        end_word_index = (i + 1) * segment_size
        if end_word_index < len(words):
            end_split_offset = words[end_word_index][-1][-1]
        else:
            end_split_offset = len(doc)
        text = doc[start_split_offset: end_split_offset].strip()
        yield text, start_split_offset
        start_split_offset = end_split_offset


def convert_dataset_2_db(dataset: Sequence[Dict], id_field: str, text_field: str, title_field: Optional[str],
                         out: TextIO, segment_size: int = 100) -> int:
    """
    Splits dataset into segments of given size  and prints it in jsonl format to the out.

    :param dataset: sequence of samples represented by dictionary
    :param id_field: name of field with sample ids
    :param text_field: name of field with text content
    :param title_field: field that contains title, if None empty string is used
    :param out: where the result should be printed
    :param segment_size: maximal size of a segment
    :return: number of records in database
    """
    global_id = 0
    for doc in tqdm(dataset, "Converting documents"):
        title = "" if title_field is None else doc[title_field]
        for i, (paragraph, start_split_offset) in enumerate(split_doc(doc[text_field], segment_size=segment_size)):
            print(json.dumps({
                "id": global_id,
                'split_index': i,
                "dataset_sample_id": doc[id_field],
                'split_start_char_offset': start_split_offset,
                "title": title,
                "contents": paragraph,
            }), file=out)
            global_id += 1
    return global_id


def convert_dataset_2_index(dataset: Sequence[Dict], id_field: str, text_field: str, title_field: Optional[str],
                            res_dir_path: str, index_type: str, batch: int = 8, threads: int = -1,
                            segment_size: int = 100) -> bool:
    """
    Converts dataset into retrieval index.

    :param dataset: sequence of samples represented by dictionary
    :param id_field: name of field with sample ids
    :param text_field: name of field with text content
    :param title_field: field that contains title, if None empty string is used
    :param res_dir_path: path to directory where the results will be saved
    :param out: where the result should be printed
    :param index_type: BM25 or DPR
    :param batch: GPU batch size when dense index is used.
    :param threads: Number of threads used for index creation when BM25 is used (-1 means use all).
    :param segment_size: maximal size of a segment
    :return:  True the index was established.
    False only database was created as there is not enough records to build index.
    """
    if threads == -1:
        threads = multiprocessing.cpu_count()

    res_dir_path = Path(res_dir_path)
    res_dir_path.mkdir(parents=True, exist_ok=True)

    index_path = res_dir_path.joinpath('index')
    config_path = res_dir_path.joinpath("config.json")

    # remove the old ones if exist
    if config_path.exists():
        config_path.unlink()
    if index_path.exists():
        shutil.rmtree(index_path)

    with open(res_dir_path.joinpath("database.jsonl"), "w") as f:
        num_of_records = convert_dataset_2_db(dataset, id_field, text_field, title_field, f, segment_size)

    config = {
        "segment_size": segment_size,
        "index_type": None,  # None signalizes not established index
    }

    if num_of_records > 1:
        config["index_type"] = index_type
        if index_type == "BM25":
            p = Popen(
                "python"
                " -m pyserini.index.lucene"
                " --collection JsonCollection"
                f" --input {res_dir_path}"
                f" --index {index_path}"
                " --generator DefaultLuceneDocumentGenerator"
                f" --threads {threads}"
                " --storePositions"
                " --storeDocvectors", shell=True)
        elif index_type == "DPR":
            p = Popen(
                "python"
                " -m pyserini.encode"
                " input"
                f" --corpus {res_dir_path}"
                " --fields text"
                " --shard-id 0"
                " --shard-num 1"
                " output"
                f" --embeddings  {index_path}"
                " --to-faiss"
                " encoder"
                " --encoder facebook/dpr-ctx_encoder-multiset-base"
                " --encoder-class dpr"
                " --fields text"
                f" --batch {batch}"
                f" --fp16", shell=True)
            config["used_encoder"] = "facebook/dpr-ctx_encoder-multiset-base"
            config["suggested_query_encoder"] = "facebook/dpr-question_encoder-multiset-base"
        else:
            raise RuntimeError("unknown index")

        p.wait()
    with open(config_path, "w") as f:
        json.dump(config, f)

    return num_of_records > 1


def huggingface_create_db(args: argparse.Namespace):
    """
    Creates database that could be used for passage retrieval.

    :param args: user arguments
    """
    config = MASAPIQAConfig(args.config if args.config else DEFAULT_CONFIG_PATH)  # mainly for path conversion

    dataset = load_dataset(args.path, args.name, split=args.split, cache_dir=config["cache_dir"])

    convert_dataset_2_index(dataset, args.id_field, args.text_field, args.title_field, args.result, args.index_type,
                            args.batch, args.threads, 100)


def single_query_retrieval(args: argparse.Namespace):
    """
    Performs search with single query on selected database.

    :param args: user arguments
    """
    directory_path = Path(args.database)
    config_path = directory_path.joinpath("config.json")
    index_path = str(directory_path.joinpath('index'))

    with open(config_path) as f:
        config = json.load(f)

    if config["index_type"] == "BM25":
        searcher = LuceneSearcher(index_path)
    else:
        q_enc = DprQueryEncoder(config["suggested_query_encoder"])
        searcher = FaissSearcher(index_path, q_enc)

    with Database(str(directory_path.joinpath("database.jsonl"))) as db:

        for res in searcher.search(args.query, args.k):
            print(res.docid)
            print(res.score)
            print(db[int(res.docid)])


def prepare_responder(args: argparse.Namespace) -> Tuple[OpenQAResponder, dict]:
    """
    Prepares responder

    :param args: user arguments
    :return: responder and configuration for prediction
    """

    config = MASAPIQAConfig(args.config if args.config else DEFAULT_CONFIG_PATH)  # mainly for path conversion

    # configs for R2D2 pipeline

    predict_config = config["open_domain_config"]

    if args.text:
        # ask on given data
        if Path(args.text).exists():
            with open(args.text, "r") as f:
                text = f.read()
        else:
            text = args.text

        title = ""
        if not args.no_title:
            title, text = text.split("\n", 1)

        # By default, Pyserini expects the fields in contents are separated by \n (in case of DPR).
        # We have only one field (text), thus let's remove the new lines.
        text = text.replace("\n", " ")

        dataset = [{
            "id": 0,
            "text": text,
            "title": title
        }]

        predict_config = config["on_demand"]

        database_path = os.path.join(config["cache_dir"], "on_demand")
        # add the on demand retriever model
        config["retriever_models"] = [{
            "label": "on_demand",
            "framework": "PyseriniRetrieverFramework",
            "database": database_path
        }]

        index_established = convert_dataset_2_index(dataset, "id", "text", "title", database_path,
                                                    predict_config["retriever"]["model"],
                                                    predict_config["retriever"]["batch"],
                                                    predict_config["retriever"]["threads"], 100)

        if not index_established:
            config["retriever_models"][0]["framework"] = "FirstKFramework"
        # this label will be used by R2D2 pipeline to identify retriever
        predict_config["retriever"]["model"] = "on_demand"

    startup_config = StartupConfiguration(config)
    runtime_config = RuntimeConfiguration(startup_config)
    return OpenQAResponder(runtime_config), predict_config


def ask(args: argparse.Namespace):
    """
    Runs question answering pipeline for given question.

    :param args: user arguments
    """
    responder, predict_config = prepare_responder(args)

    # open domain
    print(responder.predict(args.question, predict_config))


def evaluate(args: argparse.Namespace):
    """
    Evaluates the model on given dataset.

    :param args: user arguments
    """
    responder, predict_config = prepare_responder(args)

    with open(args.dataset, "r") as f, (open(args.results, "w") if args.results else nullcontext()) as res_f:
        answers = []
        ext_predictions = []
        abs_predictions = []

        start_time = time.time()
        for line in f:
            record = json.loads(line)
            answers.append(record["answer"])
            res = responder.predict(record["question"], predict_config)

            if "extractive_reader" in res:
                ext_res = res["extractive_reader"]
                ind = 0
                if "aggregated_scores" in ext_res:
                    ind = np.argmax(ext_res["aggregated_scores"])
                elif "reranked_scores" in ext_res:
                    ind = np.argmax(ext_res["reranked_scores"])

                ext_predictions.append(ext_res["answers"][ind])

            if "abstractive_reader" in res:
                abs_predictions.append(res["abstractive_reader"]["answers"][0])

            if args.results:
                res_record = {
                    "question": record["question"],
                    "answer": record["answer"]
                }
                if ext_predictions:
                    res_record["extractive_prediction"] = ext_predictions[-1]
                if abs_predictions:
                    res_record["abstractive_prediction"] = abs_predictions[-1]

                print(json.dumps(res_record), file=res_f, flush=True)

        end_time = time.time()

        eval_res = {
            "prediction_time [s]": end_time - start_time,
            "prediction_time_per_question [s]": (end_time - start_time) / len(answers),
            "extractive_em": evaluate_predictions(answers, ext_predictions, False)["accuracy"],
            "abstractive_em": evaluate_predictions(answers, abs_predictions, False)["accuracy"],
            "number_of_questions": len(answers)
        }

        print(json.dumps(eval_res))


def main():
    logging.basicConfig(format='%(process)d: %(levelname)s : %(asctime)s : %(message)s', level=logging.DEBUG)
    args = ArgumentsManager.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == '__main__':
    main()
