{
    "use_gpu": True,
    "cache_dir": "../../experiments/cache",
    "retriever_models": [
        {
            "label": "dpr",
            "model": "R2D2Encoder",
            "tokenizer": "R2D2EncoderTokenizer",
            "framework": "R2D2EncoderFramework",
            "checkpoint": "../../experiments/r2d2/dpr_official_questionencoder_nq.pt",
            "database": "../../experiments/r2d2/wiki2018_dpr_blocks_nq_open_pruned_P1700000.db",
            "index": "../../experiments/r2d2/DPR_nqsingle_offic_electrapruner_nqopen_1700000_HALF.h5"
        }
    ],
    "passage_reranker_models": [
        {
            "label": "roberta",
            "model": "PassageReranker",
            "tokenizer": "PassageRerankerTokenizer",
            "framework": "PassageRerankerFramework",
            "checkpoint": "../../experiments/r2d2/reranker_roberta-base_2021-02-25-17-27_athena19_HIT@25_0.8299645997487725_fp16.ckpt"
        }
    ],
    "reader_models": [
        {
            "label": "R2D2 reader",
            "model": "R2D2ExtractiveReader",
            "tokenizer": "R2D2ExtractiveReaderTokenizer",
            "framework": "R2D2ExtractiveReaderFramework",
            "checkpoint": "../../experiments/r2d2/EXT_READER_ELECTRA_LARGE_B128_049_J24_HALF.pt"
        }
    ],
    "aggregation": {
        "params": []
    },
    "open_domain_config": {
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
            "generative_reranking": False,
            "top_passages": 24
        },
        "score_aggregation": False
    },
    "on_demand": {
        "retriever": {
            "model": "BM25",
            "top_k": 100,
            "batch": 8,
            "threads": -1,
        },
        "passage_reranker": {
            "model": "roberta"
        },
        "extractive_reader": {
            "model": "R2D2 reader",
            "reader_top_k_answers": 5,
            "reader_max_tokens_for_answer": 5,
            "generative_reranking": False,
            "top_passages": 24
        },
        "score_aggregation": False
    }
}