# MASAPI_QA
QA for MASAPI project

## Install
Firstly you need to have JAVA installed as the Pyserini package depends on JDK 11. Please make sure to install it and 
set JAVA_HOME to JDK's directory.

You must also install the faiss (faiss~=1.7.2):

    conda install -c conda-forge faiss-gpu

or

    conda install -c pytorch faiss-gpu

Finally, use the standard requirements.txt for installation of other packages:

    pip install -r requirements.txt

Details regarding the installation of PyTorch could be found here https://pytorch.org/get-started/locally/.

### Models and Processed Wikipedia
To download trained models and processed Wikipedia, which is used for open domain question answering, download this zip
file and unzip it to your desired location:

    r2d2.fit.vutbr.cz/checkpoints/nq-open/MASAPI_QA.zip

## Usage
The model is trained to answer factoid questions using data from Wikipedia. Nevertheless, it also allows using your own data. This section presents several use cases for your smooth start with the system. 

### Asking open domain question
To ask open domain questions, use:

	./run.py ask "When Alan Turing was born?"

It will use the pre-created passage database from Wikipedia to answer this question. You can change it in the configuration file. See the [configuration](##Configuration) section.

### Asking question on given text
If you have a text that can be used to answer your question, you can use the ask argument in combination with --text argument:

	./run.py ask "When Alan Turing was born?" -t bibliography_of_alan_turing.txt

or

	./run.py ask "When Alan Turing was born?" -t "... Alan Mathison Turing was born in 1912... "  --no_title

when you want to pass your content directly.
The voluntary <b>--no_title</b> argument says that the first line doesn't contain a title and should be taken as a part of the context. It can also be used with files.


## Configuration
This section describes the configuration of the question answering pipeline that is used when the system is called with the ask argument.

The default configuration can be found at:

	masapiqa/data/default_config.py

Also look at the:
  
    masapiqa/data/example_config.py

Now follows the description of most essential parts of the configuration file:

* <b>use_gpu</b> - allows/disables the usage of GPU
* <b>cache_dir</b> - path to cache directory that is used for storing models and databases
* <b>retriever_models</b> - list of retrieval models that should be loaded
  * There are multiple types of retrievals:
    * R2D2EncoderFramework
      * It is an implementation of DPR retrieval used in [R2D2 system] (https://arxiv.org/abs/2109.03502)
      * Description of parameters:
        * label - is used to identify a given model
        * model - trained encoder to obtain a representation of a question 
        * tokenizer - used for tokenization of inputs
        * framework (R2D2EncoderFramework) - wrapper around retriever that allows to use it as part of the pipeline
        * checkpoint - path to the checkpoint of the trained model
        * database - file containing contexts
        * index - embedded contexts used for searching
    * PyseriniRetrieverFramework
      * Implementation using Pyserini (Lucene)
      * It could be DPR or BM25. It depends on the type of database that is used.
      * Description of parameters:
        * label - is used to identify a given model
        * framework (PyseriniRetrieverFramework) - wrapper around retriever that allows to use it as part of the pipeline
        * database - Path to the folder with the database. In this case, the database also contains the index and determines whether the BM25 or DPR approach should be used.
    * FirstKFramework
      * simple implementation that always takes first k records from given database
      * score of each passage is artificial set to 1
        * label - is used to identify a given model
        * framework (FirstKFramework) - wrapper around retriever that allows to use it as part of the pipeline
        * database - path to the folder with the database
* <b>passage_reranker_models</b> - list of reranker models that should be loaded
  * The fields are similar to those used for R2D2EncoderFramework.
* <b>reader_models</b> - list of reader models that should be loaded
  * The fields are similar to those used for R2D2EncoderFramework.
* <b>aggregation</b> - configurations of aggregation part that is used to aggregate scores from retriever, reranker, extractive reader, and generative reader
  * You can use it to define the weights of each score plus bias:

    [(retriver_label, reranker_label, ext_reader_label, abst_reader_used_for_reranking_label, {"w1": num, "w2": num, "w3": num, "w4": num, "bias": num})]
  
  * If you want to define aggregation for configurations without reranker or abstractive reader use None instead of their labels.
* <b>open_domain_config</b> - configuration for open domain use case
  * Description of fields:
    * retriever
      * model - a label of retriever that should be used (defined in retriever_models part)
      * top_k - how many context should be obtained
    * passage_reranker
      * model - a label of reranker that should be used (defined in passage_reranker_models)
    * extractive_reader
      * model - a label of extractive reader that should be used (defined in reader_models)
      * reader_top_k_answers - how many top answers should be returned
      * reader_max_tokens_for_answer - will consider only answers with given maximal number of tokens
      * generative_reranking - (de)activates answer reranking with an abstractive reader (use only with an abstractive reader)
    * abstractive_reader
      * You can define an abstractive reader that should be used alongside the extractive system.
      * label - a label of reader that should be used (defined in reader_models)
    * score_aggregation - activates/deactivates score aggregation 
* <b>on_demand</b> - configuration for use case when the contexts are provided
  * it is using the very same fields but the retriever part
  * As the index is created on the fly, it doesn't need to provide the label of a retriever with created index. Instead, use the model field to define whether you want to use DPR or BM25. You will define it by providing "DPR" or "BM25" strings.
  * There are also two additional fields batch and threads. The batch defines batch size when the DPR is used, and threads define the number of parallel workers when BM25 is used. 


## How to run Reranker API

In `api` subdirectory, run (for development purposes):

    uvicorn rerank:app --reload 

The API will be available at http://localhost:8000/docs. See also the config.json file for the configuration of the API.

Running the above in given subdirectory needs to have masapiqa package installed or on PYTHONPATH. You can use the

    export PYTHONPATH=path_to_folder_with_masapiqa

to set the PYTHONPATH.