import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from datetime import date, datetime, time

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
        self.client = chromadb.PersistentClient(path=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
        self.collection = self.client.get_or_create_collection(name="receipts")

    def embed_receipt(self, receipt):
        summary = (
            f"Store: {receipt.store_name}; "
            f"Store Address: {receipt.store_address or 'N/A'}; "
            f"Store Number: {receipt.store_number or 'N/A'}; "
            f"Date: {receipt.date.isoformat()}; "
            f"Payment: {receipt.payment_method}; "
            f"Total (incl. taxes): {receipt.total}; Taxes: {receipt.taxes}"
        )
        texts = [summary]
        ids = [f"r:{receipt.id}:summary"]
        ts = datetime.combine(receipt.date, time()).timestamp()
        metadatas = [{
            "user_id": receipt.user_id,
            "receipt_id": receipt.id,
            "date_iso": receipt.date.isoformat(),
            "date_ts": ts,
            "type": "summary"
        }]
        for i, li in enumerate(receipt.line_items):
            texts.append(
                f"Item name: {li.name}; "
                f"Quantity: {li.quantity}; "
                f"Unit Price: {li.price_per_unit}; "
                f"Total Price: {li.total_price}; "
                f"Category: {li.category}"
            )
            ids.append(f"r:{receipt.id}:item:{i}")
            metadatas.append({
                "user_id": receipt.user_id,
                "receipt_id": receipt.id,
                "date_iso": receipt.date.isoformat(),
                "date_ts": ts,
                "type": "line_item",
                "category": li.category,
                "item_name": li.name
            })
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )