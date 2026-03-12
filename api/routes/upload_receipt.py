from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from loguru import logger
from ingestion.ocr.easyocr_wrapper import extract_lines
from ingestion.ocr.llm_classifier_wrapper import classify_receipt
import tempfile
import os, json
from dotenv import load_dotenv
load_dotenv()
from api.dependencies.auth import get_current_user
from api.services.receipt_service import persist_receipt
from api.services.embedding_service import EmbeddingService
from db.models.user import User
from pathlib import Path

router = APIRouter()

@router.post("/receipt")
async def upload_receipt(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    logger.info(f"Received receipt: {file.filename}")
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".tiff")):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        lines = extract_lines(tmp_path)
        logger.debug(f"OCR extracted {len(lines)} lines")
    finally:
        os.remove(tmp_path)
    json_data = classify_receipt(lines)
    logger.info(f"Classified receipt: {json_data}")
    receipt = None
    if os.getenv("MODEL_VALIDATION") != "1":
        receipt = persist_receipt(json_data, lines, user_id=current_user.id)
        logger.info(f"Persisted receipt with ID: {receipt.id}")
        try:
            emb_svc = EmbeddingService()
            emb_svc.embed_receipt(receipt)
            logger.info(f"Embedded receipt {receipt.id} into vector DB")
        except Exception as e:
            logger.error(f"Failed to embed receipt {receipt.id}: {e}")
    else:
        logger.info("Model validation mode is enabled, skipping persistence and embedding")
        pred_receipt_dir = Path("validation/receipt_val_data/predictions_receipts")
        pred_receipt_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file.filename).stem
        pred_file = pred_receipt_dir / f"{stem}.json"
        full_receipt = json_data.model_dump()
        with pred_file.open("w", encoding="utf-8") as fp:
            json.dump(full_receipt, fp, indent=2, ensure_ascii=False)
    return {
        "filename": file.filename,
        "receipt_id": receipt.id if receipt else None,
        "json_data": json_data,
    }
