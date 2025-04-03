from typing import List, Tuple
from fixed_token_chunker import FixedTokenChunker
from sentence_transformers import SentenceTransformer, util
from numpy.typing import NDArray
import torch
import pandas as pd
import numpy as np
import json
from chunking.utils import rigorous_document_search
import metrics


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

class RetrievalEvaluationPipeline():

    def __init__(self, embedding_function, questions_file = 'datasets/questions_df.csv', corpus_file = 'datasets/state_of_the_union.md'):
        self.questions_file = questions_file
        self.corpus_file = corpus_file

        self.queries = self._read_queries()
        self.embedded_queries = self._embed_queries(self.queries)

    def run(self, chunk_num: int, chunk_size:int, chunk_overlap:int):
        corpus_text = self._read_corpus()
        chunks = self._chunk_corpus(corpus_text, chunk_size, chunk_overlap)
        chunk_embeddings = self._generate_embeddings(chunks)
             
        retrieved_chunks = rep._compute_similarities(chunks=chunks, chunk_embeddings=chunk_embeddings, k=chunk_num)
        recall_scores , precision_scores = self._evaluate(answers=[q.answers for q in self.queries], retrieved_chunks=retrieved_chunks)
        
        recall_mean = np.mean(recall_scores)
        recall_std = np.std(recall_scores)

        precision_mean = np.mean(precision_scores)
        precision_std = np.std(precision_scores)

        print("Recall scores: ", recall_scores)
        print("Precision scores: ", precision_scores)
        print("Recall Mean: ", recall_mean)
        print("Recall Std Mean: ", recall_std)
        print("Precision Mean: ", precision_mean)
        print("Precision Std: ", precision_std)
        
    

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
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        texts = [chunk.text for chunk in chunks]
        embeddings = model.encode(texts)    
        return embeddings
    
    def _read_queries(self) -> List[Query]:
        df = pd.read_csv(self.questions_file)
        df = df[df['corpus_id'] == 'state_of_the_union']
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
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embedded_queries = []
        for q in queries:
            embedded_question = model.encode(q.question)
            embedded_answers = [(model.encode(ans.text), ans.start, ans.end) for ans in q.answers]
            embedded_queries.append((embedded_question, embedded_answers))
        return embedded_queries
    
    def _compute_similarities(self, chunks: List[Excerpt], chunk_embeddings: List[NDArray], k: int):
        question_embeddings = [q[0] for q in self.embedded_queries]
        cosine_similarities = util.cos_sim(question_embeddings, chunk_embeddings)
        
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
    
    def _evaluate(self, answers: List[List[Excerpt]], retrieved_chunks: List[List[Excerpt]]):
        recall_scores = []
        precision_scores = []
        for i, query_answers in enumerate(answers):
            chunks = retrieved_chunks[i]
            unused_highlights = [(ans.start, ans.end) for ans in query_answers]
            numerator_sets = []
            denominator_chunks_sets = []
            for ans in query_answers:
                for chunk in chunks:
                    # Calculate intersection between chunk and reference
                    intersection = metrics.intersect_two_ranges((ans.start, ans.end), (chunk.start, chunk.end))
                    
                    if intersection is not None:
                        # Remove intersection from unused highlights
                        unused_highlights = metrics.difference(unused_highlights, intersection)

                        # Add intersection to numerator sets
                        numerator_sets = metrics.union_ranges([intersection] + numerator_sets)
                        
                        # Add chunk to denominator sets
                        denominator_chunks_sets = metrics.union_ranges([(chunk.start, chunk.end)] + denominator_chunks_sets)
            
            if numerator_sets:
                numerator_value = metrics.sum_of_ranges(numerator_sets)
            else:
                numerator_value = 0

            recall_denominator = metrics.sum_of_ranges([(ans.start,ans.end) for ans in query_answers])
            precision_denominator = metrics.sum_of_ranges([(chunk.start, chunk.end) for chunk in chunks])

            recall_score = numerator_value / recall_denominator
            recall_scores.append(recall_score)

            precision_score = numerator_value / precision_denominator
            precision_scores.append(precision_score)

        return recall_scores, precision_scores

            
if __name__ == '__main__':
    rep = RetrievalEvaluationPipeline(None)
    rep.run(5, 200, 50)




