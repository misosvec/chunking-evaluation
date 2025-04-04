# JetBrains Internship Project
## Intelligent Chunking Methods for Code Documentation RAG

### Results
More detailed results and the implementation approach are described in the file [report.pdf](report.pdf). All the computed results from various corpora are located in the [results](results/) folder.

### How to Reproduce
At first, install all required dependencies from the [requirements.txt](requirements.txt) file. The folder `data` contains all the required corpora, and the corresponding questions are located in the [questions_df.csv](data/questions_df.csv) file. The evaluation can be run using the command:

```sh
python3 src/retrieval_evaluation_pipeline.py
```
Before running, you may want to modify the parameters of the `RetrievalEvaluationPipeline` in [retrieval_evaluation_pipeline.py](src/retrieval_evaluation_pipeline.py) file, such as the embedding function, corpus, and labels for the questions:
```python
rep = RetrievalEvaluationPipeline(
    embedding_function,
    corpus_file='data/chatlogs.md',
    questions_label='chatlogs'
)
```