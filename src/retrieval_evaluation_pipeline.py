from typing import List, Tuple
from chunking.fixed_token_chunker import FixedTokenChunker
from sentence_transformers import SentenceTransformer, util
from numpy.typing import NDArray
import torch
import pandas as pd
import numpy as np
import json
from chunking.utils import rigorous_document_search
import chunking.ranges as ranges

class Excerpt:
    def __init__(self, text:str, start: int, end:int):
        self.text = text
        self.start = start
        self.end = end
    
    def __str__(self):
        return f"ANSWER: {self.text} \n START_INDEX: {self.start} \n END_INDEX: {self.end}"

class Query:
    def __init__(self, question: str, answers: List[Excerpt]):
        self.question = question
        self.answers = list(answers)

    def __str__(self):
        answers = ""
        for ans in self.answers:
            answers += f"{str(ans)}\n"
        return f"QUESTION: {self.question} \n {answers}"

Recall = Tuple[float, float]
Precision = Tuple[float, float]

class RetrievalEvaluationPipeline():

    def __init__(self, embedding_function, questions_file = 'data/questions_df.csv', corpus_file = 'data/state_of_the_union.md', questions_label='state_of_the_union'):
        self.questions_file = questions_file
        self.corpus_file = corpus_file
        self.questions_label = questions_label
        self.embedding_function = embedding_function
        self.queries = self._read_queries()
        self.embedded_queries = self._embed_queries(self.queries)

    def run(self, chunk_num: int, chunk_size:int, chunk_overlap:int) -> Tuple[Precision, Recall]:
        corpus_text = self._read_corpus()
        chunks = self._chunk_corpus(corpus_text, chunk_size, chunk_overlap)
        chunk_embeddings = self._generate_embeddings(chunks)

        retrieved_chunks = rep._compute_similarities(chunks=chunks, chunk_embeddings=chunk_embeddings, k=chunk_num)
        recall_scores , precision_scores = self._evaluate(answers=[q.answers for q in self.queries], retrieved_chunks=retrieved_chunks)
        
        recall_mean = np.mean(recall_scores)
        recall_std = np.std(recall_scores)

        precision_mean = np.mean(precision_scores)
        precision_std = np.std(precision_scores)

        return ((precision_mean, precision_std), (recall_mean, recall_std))

    def _read_corpus(self) -> str:
        with open(self.corpus_file, 'r') as file:
            text = file.read()
        return text
    
    def _chunk_corpus(self, corpus:str, chunk_size: int, chunk_overlap: int) -> List[Excerpt]:
        chunker = FixedTokenChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base"
        )

        res = [rigorous_document_search(corpus, chunk) for chunk in chunker.split_text(corpus)]
        return [Excerpt(text=r[0], start=r[1], end=r[2]) for r in res]

    def _generate_embeddings(self, chunks: List[Excerpt]) -> NDArray[np.float64]:    
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_function(texts)    
        return embeddings
    
    def _read_queries(self) -> List[Query]:
        df = pd.read_csv(self.questions_file)
        df = df[df['corpus_id'] == self.questions_label]
        df['references'] = df['references'].apply(json.loads)
        queries: List[Query]= []
        for i in range(len(df)):
            row = df.iloc[i]
            answers: List[Excerpt] = []
            for ref in row['references']:
                answers.append(Excerpt(text=ref['content'], start=int(ref['start_index']), end=int(ref['end_index'])))
            queries.append(Query(question=row['question'], answers=answers))
        return queries

    def _embed_queries(self,queries: List[Query]) ->List[Tuple[NDArray, List[Tuple[NDArray, int, int]]]]:
        embedded_queries = []
        for q in queries:
            embedded_question = self.embedding_function(q.question)
            embedded_answers = [(self.embedding_function(ans.text), ans.start, ans.end) for ans in q.answers]
            embedded_queries.append((embedded_question, embedded_answers))
        return embedded_queries
    
    def _compute_similarities(self, chunks: List[Excerpt], chunk_embeddings: List[NDArray], k: int) -> List[List[Excerpt]]:
        question_embeddings = [q[0] for q in self.embedded_queries]
        cosine_similarities = util.cos_sim(torch.tensor(np.array(question_embeddings)), torch.tensor(chunk_embeddings))
        
        results = []
        for i in range(len(self.embedded_queries)):
            scores, indices = torch.topk(cosine_similarities[i], k=k)
            top_chunks = []
            for idx, score in zip(indices, scores):
                chunk_idx = idx.item()
                chunk = chunks[chunk_idx]
                top_chunks.append(chunk)

            results.append(top_chunks)
        
        return results
    
    def _evaluate(self, answers: List[List[Excerpt]], retrieved_chunks: List[List[Excerpt]]) -> Tuple[List[float], List[float]]:
        recall_scores = []
        precision_scores = []
        for i, query_answers in enumerate(answers):
            chunks = retrieved_chunks[i]
            numerator_sets = []
            for ans in query_answers:
                for chunk in chunks:
                    intersection = ranges.intersect_two_ranges((ans.start, ans.end), (chunk.start, chunk.end))
                    if intersection is not None:
                        numerator_sets = ranges.union_ranges([intersection] + numerator_sets)
            
            numerator_value = ranges.sum_of_ranges(numerator_sets) if numerator_sets else 0

            answer_ranges = [(ans.start, ans.end) for ans in query_answers]
            answer_union = ranges.union_ranges(answer_ranges)
            recall_denominator = ranges.sum_of_ranges(answer_union)

            chunk_ranges = [(chunk.start, chunk.end) for chunk in chunks]
            chunk_union = ranges.union_ranges(chunk_ranges)
            precision_denominator = ranges.sum_of_ranges(chunk_union)

            recall_score = numerator_value / recall_denominator 
            precision_score = numerator_value / precision_denominator

            recall_scores.append(recall_score)
            precision_scores.append(precision_score)

        return recall_scores, precision_scores

            
if __name__ == '__main__':

    def embedding_function(input):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model.encode(input)
    
    rep = RetrievalEvaluationPipeline(
        embedding_function,
        corpus_file='data/wikitexts.md',
        questions_label='wikitexts'
    )

    results = []

    for chunk_size in [200,400,600]:
        for overlap in [0,50,100,150]:
            for num_chunks in [2, 4, 6, 10]:
                print(f"chunk_size={chunk_size} overlap={overlap} num_chunks={num_chunks}")
                precision_data, recall_data = rep.run(num_chunks, chunk_size, overlap)
                recall_mean, recall_std = recall_data[0]*100, recall_data[1]*100
                precision_mean, precision_std = precision_data[0]*100, precision_data[1]*100
                results.append({
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "num_chunks": num_chunks,
                    "precision_mean": f"{precision_mean:.1f}",
                    "precision_std": f"{precision_std:.1f}",
                    "recall_mean": f"{recall_mean:.1f}",
                    "recall_std": f"{recall_std:.1f}"
                })

    df = pd.DataFrame(results)
    df.to_csv("results/results-wikitexts.csv", index=False)