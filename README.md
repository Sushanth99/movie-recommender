# Movie recommender using XLM-RoBERTa and Gemini

## Introduction
This is a text-based movie recommender. The users are required to convey their interests in a textual format and the system returns movie recommendations along with their descriptions/overviews. The goal is to build a simple coldstart recommendation system which aligns with the user's interests at query time.

The text can be of variable length and no particular format is imposed on the user. Some example user inputs can be: `"action, drama, war"`, `"Recommmend me a comedy movie to relax"`.

The system returns a list of movie suggestions along with their descriptions and genres in the decreasing order of predicted similarity.

## Architecture
The recommender system has three main components: (1) An embedding model (2) A vector database and (3) A generative model.

**Embedding model**: The model is used to encode the descriptions of movies to vector embeddings. Any BERT or RoBERTa like model can be used. The system uses `intfloat/multilingual-e5-large`(HuggingFace model name), which is a 560M parameter model based on XLM-RoBERTa architecture.

**Vector database**: The generated vector embeddings are stored in a vector database for effective semantic search and retrieval. Pinecone's cloud based vector database was used for this purpose with consine similarity as the similarity metric. 

**Generative model**: The text from the user is used to generate a dummy movie description by prompting an LLM. The embedding model encodes this description and is used for similarity search in the vector database. The system uses `Gemini 2.0 Flash Experimental` model for this purpose.

Note: The choices for the embedding and generative models are solely based on accessibility (APIs). You can switch to a different generative model. However, if you wish to use a different embedding model then the entire dataset needs to be encoded and stored in a vector database again.