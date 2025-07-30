from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import upload_receipt
from api.routes import rag_qa

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"{request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Status code: {response.status_code}")
    return response

app.include_router(upload_receipt.router, prefix="/upload")
app.include_router(rag_qa.router, prefix="/rag")


