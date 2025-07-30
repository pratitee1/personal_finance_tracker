from db.setup import SessionLocal
from db.models.receipt import Receipt
from db.models.line_item import LineItem
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy.orm import joinedload
def persist_receipt(json_data, lines, user_id=1):
    db = SessionLocal()
    try:
        receipt = Receipt(
            user_id=user_id,
            store_name=json_data.store_name,
            store_address=json_data.store_address,
            store_number=json_data.store_number,
            total=json_data.total,
            taxes=json_data.taxes,
            payment_method=json_data.payment_method,
            confidence_score_ocr=json_data.confidence_score_ocr,
            date=datetime.strptime(json_data.date, "%d-%m-%Y").date() if json_data.date else None,
            raw_text="\n".join(lines),
        )
        db.add(receipt)
        db.flush()
        for item in json_data.items:
            line_item = LineItem(
                name=item.name,
                quantity=item.quantity,
                price_per_unit=item.price_per_unit,
                total_price=item.total_price,
                confidence_score=item.confidence_score,
                category=item.category,
                receipt=receipt,
            )
            db.add(line_item)
        db.commit()
        receipt = db.query(Receipt).options(joinedload(Receipt.line_items)).filter(Receipt.id == receipt.id).one()
        return receipt
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
