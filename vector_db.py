import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings
import pandas as pd 
from data_preprocessing import clean_text

# Create client (Database)
CHROMA_DATA_PATH = "vector_db/"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Create embedding function from sentence-transformer
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# Create collection (Table)
## See more configuration on https://docs.trychroma.com/docs/collections/configure
COLLECTION_NAME = "tours_collection"

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func, # embbeding function from sentence-transformer
    metadata={"hnsw:space": "cosine"}, # use cosine similarity to find closeness result. You can choose another include l2, ip.
)

# Load data
df = pd.read_csv('tours_merged_cleaned2.csv')

# Preprocess data
df['description'] = df['description'].apply(clean_text)

# Prepare data to add to collection
ids = []
documents = []
metadatas = []

for index, row in df.iterrows():
    ids.append(str(index))

    documents.append(str(row["description"]))

    metadatas.append({
        "program_tour": str(row["program_tour"]),
        "url": str(row["url"]),
        "price": str(row["price"]),
        "region": str(row["region"])
    })


# Add data to collection
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print("Data added to ChromaDB successfully.")
print("Total records:", collection.count())