import os
from typing import Tuple, List
from datetime import date, datetime, time
from openai import OpenAI
from api.services.embedding_service import EmbeddingService

class RAGService:
    def __init__(self):
        self.emb_svc = EmbeddingService()
        self.client   = OpenAI(
                base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                api_key  = os.getenv("GROQ_API_KEY"),
            )
    def answer_question(self, user_id: int, question: str,
            start_date: date | None = None, end_date: date | None = None, 
            top_k: int = 10) -> Tuple[str, List[str]]:
        filters = [{"user_id": user_id}]
        if start_date and end_date:
            lo = datetime.combine(start_date, time()).timestamp()
            hi = datetime.combine(end_date, time()).timestamp()
            filters.append({"date_ts": {"$gte": lo}})
            filters.append({"date_ts": {"$lte": hi}})
        q_emb = self.emb_svc.model.encode([question], show_progress_bar=False).tolist()[0]
        results = self.emb_svc.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where={"$and": filters} if len(filters) > 1 else filters[0]
        )
        docs = results.get("documents", [])[0]
        if not docs:
            return "I don’t know.", []
        context = "\n\n".join(docs)
        prompt  = (
            "Use the following receipt context to answer the question. "
            "If the answer is not in the context, reply “I don’t know.”\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        resp = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip()
        return answer, docs
