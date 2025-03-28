from typing import List, Tuple
from fixed_token_chunker import FixedTokenChunker
from sentence_transformers import SentenceTransformer, util
from numpy.typing import NDArray
import torch
import pandas as pd
import numpy as np
import json
from chunking.utils import rigorous_document_search

_QUESTIONS_FILE = 'datasets/questions_df.csv'
_CORPUS_FILE = 'datasets/state_of_the_union.md'

class Answer:
    def __init__(self, text:str, start: int, end:int):
        self.text = text
        self.start = start
        self.end = end
    
    def __str__(self):
        return f"ANSWER: {self.text} \n START_INDEX: {self.start} \n END_INDEX: {self.end}"

class Query:
    def __init__(self, question: str, answers: List[Answer]):
        self.question = question
        self.answers = list(answers)

    def __str__(self):
        answers = ""
        for ans in self.answers:
            answers += f"{str(ans)}\n"
        return f"QUESTION: {self.question} \n {answers}"


class RetrievalEvaluationPipeline():

    def __init__(self, embedding_function):
        pass

    def run(self, corpus_file:str, query_file:str):
        corpus = self._read_corpus(corpus_file)
        chunks = self._chunk_corpus(corpus)
        embeddings = self._generate_embeddings(chunks)
        


    def _read_corpus(self, filename:str) -> str:
        with open(filename, 'r') as file:
            text = file.read()
        return text
    
    def _chunk_corpus(self, corpus:str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int,int]]:
        chunker = FixedTokenChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base"
        )
        return [rigorous_document_search(corpus, chunk) for chunk in chunker.split_text(corpus)]

    def _generate_embeddings(self, chunks: List[str]) -> NDArray[np.float64]:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(chunks)
        return embeddings
    
    def _read_queries(self, filename:str) -> List[Query]:
        df = pd.read_csv(filename)
        df = df[df['corpus_id'] == 'state_of_the_union']
        df['references'] = df['references'].apply(json.loads)
        queries: List[Query]= []
        for i in range(len(df)):
            row = df.iloc[i]
            answers: List[Answer] = []
            for ref in row['references']:
                answers.append(Answer(text=ref['content'], start=int(ref['start_index']), end=int(ref['end_index'])))
            queries.append(Query(question=row['question'], answers=answers))
        return queries

    def _embed_queries(self,queries: List[Query]) ->List[Tuple[NDArray, List[Tuple[NDArray, int, int]]]]:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embedded_queries = []
        for q in queries:
            embedded_question = model.encode(q.question)
            embedded_answers = [(model.encode(ans.text), ans.start, ans.end) for ans in q.answers]
            embedded_queries.append((embedded_question, embedded_answers))
        return embedded_queries
    
    def _compute_similarities(self, chunk_embeddings, queries, queries_embeddings, k:int):
        question_embeddings = [q[0] for q in queries_embeddings]
        cosine_similarities = util.cos_sim(question_embeddings, chunk_embeddings)
        # Number of top results to return
        top_k_indices = []
        for i in range(len(queries_embeddings)):
            top_k_indices.append(torch.topk(cosine_similarities[i], k=k))
        return top_k_indices

    
    


if __name__ == '__main__':
    # solve()
    # emb = RetrievalEvaluationPipeline()._generate_embeddings(["This is an example sentence", "Each sentence is converted"])
    # print(f"emb shape is {emb.shape}")
    # print(f"embediddngs are {emb}")
    rep = RetrievalEvaluationPipeline(None)
    corpus = rep._read_corpus(_CORPUS_FILE)
    chunks = rep._chunk_corpus(corpus, 300, 30)
    embeddings = rep._generate_embeddings(chunks)
    queries = rep._read_queries(_QUESTIONS_FILE)
    embedded_queries = rep._embed_queries(queries)
    top_k_indices = rep._compute_similarities(queries=queries, queries_embeddings=embedded_queries, chunk_embeddings=embeddings, k=3)
    print(f"QUESTION: {queries[0].question}")

    for ans in queries[0].answers:
        print(f"ORIGINAL ANSWER: {ans.text} \n ORIGINAL ANSWER START INDEX: {ans.start} \n ORIGINAL ANSWER END INDEX: {ans.end}")
    for id in top_k_indices[0].indices:
        print(f"RETRIEVED ANSWER: {chunks[id]}")
    
    # print(f"embedded queries are {embedded_queries[1]}")
    # print(f"embedded queries are {embedded_queries[1].shape}")




