from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger
from ingestion.ocr.easyocr_wrapper import extract_lines
from ingestion.ocr.llm_classifier_wrapper import classify_receipt
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()
from api.services.receipt_service import persist_receipt

router = APIRouter()

@router.post("/receipt")
async def upload_receipt(file: UploadFile = File(...)):
    """Run OCR and return lines."""
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
    receipt = persist_receipt(json_data, lines, user_id=1)
    logger.info(f"Persisted receipt with ID: {receipt.id}")
    return {"filename": file.filename, "receipt_id": receipt.id, "json_data": json_data}