import chromadb
client = chromadb.PersistentClient(path="./my_local_database")
collection = client.get_or_create_collection("my_collection")