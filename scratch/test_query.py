import chromadb

client = chromadb.PersistentClient(path="./my_local_database")
collection = client.get_collection("langchain")
print("Collection metadata:", collection.metadata)
print("Collection embedding function:", collection._embedding_function)
