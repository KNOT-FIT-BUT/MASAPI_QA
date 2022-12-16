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
        },
        {
            "label": "cnn_bm25",
            "framework": "PyseriniRetrieverFramework",
            "database": "../../experiments/cnn_dailymail",
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
        },
        {
            "label": "R2D2 abs reader",
            "model": "T5FusionInDecoder",
            "tokenizer": "T5FusionInDecoderTokenizer",
            "framework": "T5FusionInDecoderFramework",
            "checkpoint": "../../experiments/r2d2/generative_reader_EM0.4830_S7000_Mt5-large_2021-01-15_05_fp16.pt"
        }
    ],
    "aggregation": {
        "params": [
            ["dpr", None,    "R2D2 reader", None, {"w1": 0.06394632905721664 , "w2": 0.0                , "w3": 0.12902936339378357, "w4": 0.0               , "bias": 0.6814973950386047}],
            ["dpr", None,    "R2D2 reader",    "R2D2 abs reader", {"w1": 0.051815975457429886, "w2": 0.0                , "w3": 0.19255106151103973, "w4": 0.4719049334526062, "bias": -0.23736271262168884}],
            ["dpr",    "roberta",    "R2D2 reader", None, {"w1": 0.02802908793091774 , "w2": 0.11126989126205444, "w3": 0.0582578182220459 , "w4": 0.0               , "bias": 0.5771093368530273}],
            ["dpr",    "roberta",    "R2D2 reader",    "R2D2 abs reader", {"w1": 0.19083140790462494 , "w2": 0.37659892439842224, "w3": 0.2390064001083374 , "w4": 1.0336636304855347, "bias": 0.02763156034052372}]
        ]
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
            "generative_reranking": True,
            "top_passages": 24
        },
        "abstractive_reader": {
            "model": "R2D2 abs reader",
            "top_passages": 25
        },
        "score_aggregation": True
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