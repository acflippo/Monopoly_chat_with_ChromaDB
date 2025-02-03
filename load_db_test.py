import chromadb
import ollama

# Define the embedding model to use
embedding_model = 'mxbai-embed-large'

# Define the location for the ChromaDB 
persist_directory = "./monopoly_chroma_db"  # Directory where data will be saved
client = chromadb.PersistentClient(path=persist_directory)

# Load a collection
collection_name = "documents_collection"
collection = client.get_or_create_collection(name=collection_name)

query_text = "How much money does each player start with?"
query_embedding = ollama.embeddings(model=embedding_model, prompt=query_text)['embedding']

# Querying ChromaDB using the same embedding function that was used to add documents
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print("Query Results:", results['documents'][0][0])
