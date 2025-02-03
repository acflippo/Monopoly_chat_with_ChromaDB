import chromadb
import ollama

# Set this flag to True to create the ChromaDB
CREATE_DB_FLAG = True

# Define the embedding model to use
embedding_model = 'mxbai-embed-large'

# Save the ChromaDB onto the disk
persist_directory = "./monopoly_chroma_db"  # Directory where data will be saved
client = chromadb.PersistentClient(path=persist_directory)

# Create or load a collection
collection_name = "documents_collection"
collection = client.get_or_create_collection(name=collection_name)

if CREATE_DB_FLAG:
    # Load the text file
    text_file_path = "annie_monopoly_vault.txt"  # Replace with your text file path
    with open(text_file_path, "r") as file:
        documents = file.readlines()

    # Add documents to the collection
    for idx, doc_text in enumerate(documents):
        doc_id = str(idx + 1)  # Generate a unique ID for each document

        # Create embedding for each row in documents
        embedding_text = ollama.embeddings(model=embedding_model, prompt=doc_text)['embedding']

        # print(doc_id)
        # Add the document to the collection
        collection.add(
            documents=[doc_text.strip()],  # Remove leading/trailing whitespace
            ids=[doc_id],
            embeddings=[embedding_text]  # Uncomment if using embeddings
        )

    print(f"Documents have been saved to the ChromaDB collection '{collection_name}'.")
else:
    print("Nothing to do here - no need to create ChromaDB.")