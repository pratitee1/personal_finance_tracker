from chromadb import PersistentClient
from chromadb.config import Settings

client = PersistentClient(path="./chroma_db")
try:
    client.delete_collection("receipts")
    print("Dropped 'receipts' collection.")
except Exception:
    pass
client.get_or_create_collection(name="receipts")
print("Created fresh 'receipts' collection.")