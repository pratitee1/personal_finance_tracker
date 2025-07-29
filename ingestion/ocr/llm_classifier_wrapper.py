import os
from typing import List, Optional, Literal
from pydantic import ValidationError, BaseModel, Field, conint, confloat
from fastapi import HTTPException
from openai import OpenAI

#Define required JSON schema
class LineItem(BaseModel):
    name: str
    quantity: Optional[conint(ge=0)] = None
    price_per_unit: Optional[confloat(ge=0.0)] = None
    total_price: Optional[confloat(ge=0.0)]    = None
    category: Literal["essential groceries", "feel good groceries", "clothing", 
                      "electronics", "furniture", "other"]
    confidence_score: int

class ReceiptSummary(BaseModel):
    store_name: str
    store_address: str
    store_number: Optional[str] = None
    items: List[LineItem]
    taxes: float
    total: float
    # date: Optional[str] = Field(
    #     None,
    #     pattern=r"^\d{4}-\d{2}-\d{2}$",
    #     description="YYYY-MM-DD"
    # )
    date: Optional[str] = None
    payment_method: Literal["cash","credit card","debit card","check","other"]
    confidence_score_ocr: int

#OCR lines to json conversion with classification of receipts and confidence scores
def classify_receipt(lines: List[str]) -> ReceiptSummary:
    receipt_lines = "\n".join(lines)
    client = OpenAI(
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key  = os.getenv("GROQ_API_KEY"),
    )
    SYSTEM_PROMPT = (
        "You are a helpful information extractor. You read receipt_lines extracted from OCR "
        "and output a single JSON object that matches the ReceiptSummary schema exactly."
        "Items with very long names may wrap to the next line. "
        "For each grocery item also assign one of: essential groceries, feel good groceries, "
        "clothing, electronics, furniture, or other. If you cannot determine the category,"
        " use 'other'. Include a confidence_score out of 100 for your extraction of each item."
        "At the end include a confidence_score_ocr out of 100 for your confidence in the correctness"
        "of OCR extracted input."
    )
    try:
        response = client.beta.chat.completions.parse(
            model="moonshotai/kimi-k2-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": receipt_lines},
            ],
            response_format=ReceiptSummary,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
    return response.choices[0].message.parsed
