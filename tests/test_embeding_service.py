from types import SimpleNamespace
from datetime import date, datetime, time
from unittest.mock import MagicMock, patch
import pytest
from api.services.embedding_service import EmbeddingService



def _make_receipt(store_name="StoreX", store_address="123 Ave", store_number="555",
                  d=date(2022,1,2), payment_method="Card", total=10.0, taxes=0.5,
                  rid=42, uid=7, line_items=None):
    if line_items is None:
        line_items = []
    return SimpleNamespace(
        store_name=store_name,
        store_address=store_address,
        store_number=store_number,
        date=d,
        payment_method=payment_method,
        total=total,
        taxes=taxes,
        id=rid,
        user_id=uid,
        line_items=line_items
    )

@patch('api.services.embedding_service.chromadb.PersistentClient')
@patch('api.services.embedding_service.SentenceTransformer')
def test_embed_receipt_inserts_expected_records(mock_sentence_transformer, mock_persistent_client):
    # Prepare model mock
    model = MagicMock()
    embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    encode_ret = MagicMock()
    encode_ret.tolist.return_value = embeddings
    model.encode.return_value = encode_ret
    mock_sentence_transformer.return_value = model

    # Prepare collection mock
    collection = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    mock_persistent_client.return_value = client

    # Build receipt with two line items
    li1 = SimpleNamespace(name='Apple', quantity=2, price_per_unit=1.0, total_price=2.0, category='Fruit')
    li2 = SimpleNamespace(name='Milk', quantity=1, price_per_unit=3.0, total_price=3.0, category='Dairy')
    receipt = _make_receipt(line_items=[li1, li2], rid=42, uid=7, d=date(2022, 1, 2))

    svc = EmbeddingService()
    svc.embed_receipt(receipt)

    # Expected texts
    summary = (
        f"Store: {receipt.store_name}; "
        f"Store Address: {receipt.store_address or 'N/A'}; "
        f"Store Number: {receipt.store_number or 'N/A'}; "
        f"Date: {receipt.date.isoformat()}; "
        f"Payment: {receipt.payment_method}; "
        f"Total (incl. taxes): {receipt.total}; Taxes: {receipt.taxes}"
    )
    item1_text = "Item name: Apple; Quantity: 2; Unit Price: 1.0; Total Price: 2.0; Category: Fruit"
    item2_text = "Item name: Milk; Quantity: 1; Unit Price: 3.0; Total Price: 3.0; Category: Dairy"
    expected_texts = [summary, item1_text, item2_text]

    # Verify encode called correctly
    model.encode.assert_called_once_with(expected_texts, show_progress_bar=False)

    # Verify upsert called with expected payload
    collection.upsert.assert_called_once()
    _, kwargs = collection.upsert.call_args
    assert kwargs['ids'] == ['r:42:summary', 'r:42:item:0', 'r:42:item:1']
    assert kwargs['documents'] == expected_texts
    assert kwargs['embeddings'] == embeddings

    # Check metadatas content and date_ts approximately
    metadatas = kwargs['metadatas']
    assert len(metadatas) == 3
    expected_ts = datetime.combine(receipt.date, time()).timestamp()
    assert metadatas[0]['user_id'] == receipt.user_id
    assert metadatas[0]['receipt_id'] == receipt.id
    assert metadatas[0]['date_iso'] == receipt.date.isoformat()
    assert abs(metadatas[0]['date_ts'] - expected_ts) < 1e-6
    assert metadatas[0]['type'] == 'summary'

    # Check a line item metadata
    assert metadatas[1]['type'] == 'line_item'
    assert metadatas[1]['category'] == li1.category
    assert metadatas[1]['item_name'] == li1.name

@patch('api.services.embedding_service.chromadb.PersistentClient')
@patch('api.services.embedding_service.SentenceTransformer')
def test_embed_receipt_with_no_line_items(mock_sentence_transformer, mock_persistent_client):
    # Prepare model mock
    model = MagicMock()
    embeddings = [[0.9, 0.8]]
    encode_ret = MagicMock()
    encode_ret.tolist.return_value = embeddings
    model.encode.return_value = encode_ret
    mock_sentence_transformer.return_value = model

    # Prepare collection mock
    collection = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    mock_persistent_client.return_value = client

    # Build receipt with no line items
    receipt = _make_receipt(line_items=[], rid=100, uid=20, d=date(2023, 3, 15))

    svc = EmbeddingService()
    svc.embed_receipt(receipt)

    # Expected texts: only summary
    summary = (
        f"Store: {receipt.store_name}; "
        f"Store Address: {receipt.store_address or 'N/A'}; "
        f"Store Number: {receipt.store_number or 'N/A'}; "
        f"Date: {receipt.date.isoformat()}; "
        f"Payment: {receipt.payment_method}; "
        f"Total (incl. taxes): {receipt.total}; Taxes: {receipt.taxes}"
    )
    expected_texts = [summary]

    model.encode.assert_called_once_with(expected_texts, show_progress_bar=False)
    collection.upsert.assert_called_once()
    _, kwargs = collection.upsert.call_args
    assert kwargs['ids'] == ['r:100:summary']
    assert kwargs['documents'] == expected_texts
    assert kwargs['embeddings'] == embeddings

    metadatas = kwargs['metadatas']
    assert len(metadatas) == 1
    expected_ts = datetime.combine(receipt.date, time()).timestamp()
    assert abs(metadatas[0]['date_ts'] - expected_ts) < 1e-6
    assert metadatas[0]['type'] == 'summary'