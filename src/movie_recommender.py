# All models and API calling functions
# from google.colab import userdata
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import pandas as pd
from pinecone.grpc import PineconeGRPC as Pinecone
# from sentence_transformers import SentenceTransformer, models
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

## 1. LLM from dummy description generation
genai.configure(api_key=GEMINI_API_KEY)
model_name_for_dummy_desc = "gemini-2.0-flash-exp"
model_for_dummy_desc = genai.GenerativeModel(model_name_for_dummy_desc)

def generate_dummy_desc(user_input: str) -> str:
    """Returns the dummy description based on the user_input."""

    prompt_for_dummy_desc = f"""Using the following text generate a dummy movie overview. This is used 
    for semantic search. The response should be around 100 words. The output should be in the following 
    format 'Genres: [genres seperated by commas]\nOverview: [overview of the movie]'

    TEXT: {user_input}"""
    response = model_for_dummy_desc.generate_content(prompt_for_dummy_desc)
    return response.text


## 2. Embedding model
# model_name_for_embedding = "intfloat/multilingual-e5-large"
# word_embedding_model = models.Transformer(model_name_for_embedding)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
#                                pooling_mode='mean', # 'cls', 'max'
#                             #    pooling_mode_mean_tokens=True,
#                                )
# model_for_embedding = SentenceTransformer(modules=[word_embedding_model, 
#                                                    pooling_model, 
#                                                    models.Normalize()])

# def generate_embeddings(text_list: list[str], show_progress_bar=False) -> np.ndarray:
#     """Returns the embeddings for the given list of texts."""
#     embeddings = model_for_embedding.encode(text_list, 
#                                             show_progress_bar=show_progress_bar)
#     return embeddings

def generate_embeddings_pinecone(text_list: list[str], show_progress_bar=False) -> np.ndarray:
    """Returns the embeddings for the given list of texts."""
    embeddings = pc.inference.embed(
        "multilingual-e5-large",
        inputs=text_list,
        parameters={
            "input_type": "passage"
        }
    )
    return [e['values'] for e in embeddings]

## 3. Querying the vector database
index_name = "example-index"
pinecone_host = "https://example-index-4sgrwfc.svc.aped-4627-b74a.pinecone.io"
pc = Pinecone(PINECONE_API_KEY)
index = pc.Index(host=pinecone_host)
namespace = "ns1"
def similarity_search(query_vector, index, namespace, top_k=3):
    search_result = index.query(
        namespace=namespace,
        vector=query_vector,
        top_k=top_k,
        include_values=True,
    )
    search_result = search_result.to_dict()
    return [(search_result['matches'][i]['id'], search_result['matches'][i]['score'])
            for i in range(len(search_result['matches']))]

## 4. Main function
def movie_recommender(user_input: str, top_k=3) -> list[str]:
    """Returns the list of recommended movies based on the user_input."""
    dummy_desc = generate_dummy_desc(user_input)
    print(dummy_desc)
    embeddings = generate_embeddings_pinecone([dummy_desc])
    search_result = similarity_search(embeddings[0], index, namespace, top_k)
    return search_result
    # movie_ids = [search_result[i][0] for i in range(len(search_result))]
    # return movies_df.loc[movie_ids, 'movie_id'].tolist()


## Inference Pipeline (Assuming the vector database is setup and populated with embedding vectors)
movies_df = pd.read_csv("movies_df.csv", index_col=0)
# 1. User input
user_input = "recommend some war movies"

result = movie_recommender(user_input, top_k=10)
# print(result)
recommendations = movies_df.loc[[r[0] for r in result], :].to_dict(orient='records')
from pprint import pprint
for rec, r in zip(recommendations, result):
    pprint(rec['movie_name'])
    pprint(rec['text'])
    pprint(r[1])
    print('\n')