from typing import List
from fixed_token_chunker import FixedTokenChunker
# from sentence_transformers import SentenceTransformer
import pandas as pd
import json

QUESTIONS_DATASET = 'datasets/questions_df.csv'

def solve():
    corpus = read_corpus()
    chunks = chunk_corpus(corpus)

    print(chunker(text))

def chunk_corpus(text:str):
    chunker = FixedTokenChunker()

    # Or customize the chunker
    chunker = FixedTokenChunker(
        chunk_size=4000,  # Maximum number of tokens per chunk
        chunk_overlap=200,  # Number of tokens to overlap between chunks
        encoding_name="cl100k_base"  # Specify the encoding
    )

    return chunker.split_text(text)

def gen_embeddings(chunks: List[str]):
    sentences = ["This is an example sentence", "Each sentence is converted"]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    print(embeddings)

def read_corpus(filename:str):
    with open(filename, 'r') as file:
        text = file.read()

    return text

def embed_query(filename:str):
    print("EMBED QUERY CALLED")
    df = pd.read_csv(filename)
    print(f"entire corpus length {len(df)}")
    df = df[df['corpus_id'] == 'state_of_the_union']
    print(f"filtered corpus length {len(df)}")
    df['references'] = df['references'].apply(json.loads)
    df = df.drop(columns=['corpus_id'], axis=1)
    print(df.iloc[0])
    ref = df.iloc[0]['references']
    print(f"ref is {ref[0]['content']}")

if __name__ == '__main__':
    # solve()
    embed_query(QUESTIONS_DATASET)