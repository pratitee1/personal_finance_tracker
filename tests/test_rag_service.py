import pytest
from unittest.mock import MagicMock, patch, call
from datetime import date, datetime, time
from types import SimpleNamespace
from api.services.rag_service import RAGService


@patch("api.services.rag_service.OpenAI")
@patch("api.services.rag_service.EmbeddingService")
def test_answer_question_with_results(mock_emb_svc_class, mock_openai_class):
    """Test answer_question with matching documents and date filters."""
    # Setup mocks
    mock_emb_svc = MagicMock()
    mock_emb_svc_class.return_value = mock_emb_svc
    
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
    mock_emb_svc.model = mock_model
    
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [
            [
                "Store: Walmart; Store Address: 123 Main St; Date: 2026-02-15; Total: 45.99; Taxes: 3.50",
                "Item name: Groceries; Quantity: 1; Total Price: 42.49; Category: Food"
            ]
        ]
    }
    mock_emb_svc.collection = mock_collection
    
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "You spent $45.99 at Walmart on groceries."
    mock_client.chat.completions.create.return_value = mock_response
    
    # Instantiate and call
    rag_svc = RAGService()
    start = date(2026, 2, 1)
    end = date(2026, 2, 28)
    answer, docs = rag_svc.answer_question(
        user_id=42,
        question="How much did I spend on groceries?",
        start_date=start,
        end_date=end,
        top_k=5
    )
    
    # Assertions
    assert answer == "You spent $45.99 at Walmart on groceries."
    assert len(docs) == 2
    assert "Walmart" in docs[0]
    
    # Verify model.encode called with question
    mock_model.encode.assert_called_once_with(
        ["How much did I spend on groceries?"],
        show_progress_bar=False
    )
    
    # Verify collection.query called with correct filters
    mock_collection.query.assert_called_once()
    call_kwargs = mock_collection.query.call_args[1]
    assert call_kwargs["n_results"] == 5
    assert call_kwargs["query_embeddings"] == [[0.1, 0.2, 0.3]]
    assert "$and" in call_kwargs["where"]
    filters = call_kwargs["where"]["$and"]
    assert filters[0] == {"user_id": 42}
    
    # Verify LLM called
    mock_client.chat.completions.create.assert_called_once()
    llm_kwargs = mock_client.chat.completions.create.call_args[1]
    assert llm_kwargs["model"] == "llama-3.1-8b-instant"
    assert llm_kwargs["temperature"] == 0.0
    assert "How much did I spend on groceries?" in llm_kwargs["messages"][0]["content"]




@patch("api.services.rag_service.OpenAI")
@patch("api.services.rag_service.EmbeddingService")
def test_answer_question_no_date_filter(mock_emb_svc_class, mock_openai_class):
    """Test answer_question without date filters."""
    # Setup mocks
    mock_emb_svc = MagicMock()
    mock_emb_svc_class.return_value = mock_emb_svc
    
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = [[0.2, 0.3, 0.4]]
    mock_emb_svc.model = mock_model
    
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Store: Amazon; Total: 29.99"]]
    }
    mock_emb_svc.collection = mock_collection
    
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "You spent $29.99 on Amazon."
    mock_client.chat.completions.create.return_value = mock_response
    
    # Instantiate and call (no date filters)
    rag_svc = RAGService()
    answer, docs = rag_svc.answer_question(
        user_id=10,
        question="Total spending?"
    )
    
    # Assertions
    assert answer == "You spent $29.99 on Amazon."
    assert len(docs) == 1
    
    # Verify collection.query called with only user_id filter (no $and)
    call_kwargs = mock_collection.query.call_args[1]
    assert call_kwargs["where"] == {"user_id": 10}