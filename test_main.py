import pytest
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app, SearchQuery

# Create test client
client = TestClient(app)


class TestDataIngestion:
    """Test suite for the /api/ingest endpoint."""
    
    def test_ingest_valid_json_file(self):
        """Test successful ingestion of a valid JSON file with required fields."""
        valid_json_data = [
            {
                "title": "Machine Learning in Healthcare",
                "authors": ["John Doe", "Jane Smith"],
                "abstract": "This paper explores applications of machine learning in medical diagnostics.",
                "conference_name": "IEEE Medical AI",
                "conference_year": 2024,
                "paper_id": "test_001"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_json_data, f)
            temp_file = f.name
        
        try:
            with open(temp_file, 'rb') as f:
                response = client.post(
                    "/api/ingest",
                    files={"file": ("test_valid.json", f, "application/json")}
                )
            
            assert response.status_code == 200
            result = response.json()
            assert "message" in result
            assert "valid_records_ingested" in result
            assert result["valid_records_ingested"] >= 0
        finally:
            Path(temp_file).unlink()
    
    def test_ingest_rejects_non_json_file(self):
        """Test that the endpoint rejects non-JSON files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not JSON")
            temp_file = f.name
        
        try:
            with open(temp_file, 'rb') as f:
                response = client.post(
                    "/api/ingest",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 400
            assert "Only JSON files are supported" in response.json()["detail"]
        finally:
            Path(temp_file).unlink()


class TestSearchEndpoint:
    """Test suite for the /api/search endpoint."""
    
    @patch('ollama.generate')
    @patch('main.collection.query')
    def test_search_returns_formatted_response(self, mock_query, mock_ollama):
        """Test that search endpoint returns properly formatted response."""
        # Mock ChromaDB results
        mock_query.return_value = {
            'documents': [[
                "This paper discusses machine learning approaches."
            ]],
            'metadatas': [[{
                'title': 'ML Approaches',
                'authors': 'Test Author',
                'conference': 'AI Conf 2024',
                'track': 'Machine Learning'
            }]]
        }
        
        # Mock Ollama response
        mock_ollama.return_value = {
            'response': 'Best Matching Paper\nTitle: ML Approaches\nAuthors: Test Author'
        }
        
        response = client.post(
            "/api/search",
            json={"query": "machine learning", "num_results": 5, "year": ""}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "query" in result
        assert "response" in result
        assert "papers" in result
        assert "num_papers_used" in result
        assert result["num_papers_used"] > 0
    
    @patch('ollama.generate')
    @patch('main.collection.query')
    def test_search_filters_by_year(self, mock_query, mock_ollama):
        """Test that search endpoint filters results by year when provided."""
        # Mock ChromaDB results with multiple years
        mock_query.return_value = {
            'documents': [["Paper 1 abstract", "Paper 2 abstract"]],
            'metadatas': [[
                {
                    'title': 'ML Paper 2024',
                    'authors': 'Author A',
                    'conference': 'AI Conf 2024',
                    'track': 'ML'
                },
                {
                    'title': 'ML Paper 2023',
                    'authors': 'Author B',
                    'conference': 'AI Conf 2023',
                    'track': 'ML'
                }
            ]]
        }
        
        # Mock Ollama response
        mock_ollama.return_value = {
            'response': 'Best Matching Paper\nTitle: ML Paper 2024\nAuthors: Author A'
        }
        
        response = client.post(
            "/api/search",
            json={"query": "machine learning", "num_results": 5, "year": "2024"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["num_papers_used"] == 1
        assert result["papers"][0]["conference"] == "AI Conf 2024"
    
    @patch('main.collection.query')
    def test_search_handles_no_results(self, mock_query):
        """Test that search gracefully handles when no papers are found."""
        # Mock empty ChromaDB results
        mock_query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        response = client.post(
            "/api/search",
            json={"query": "nonexistent topic xyz", "num_results": 5, "year": ""}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["response"] == "No relevant papers found."
        assert result["papers"] == []
    
    @patch('main.collection.query')
    def test_search_no_papers_in_requested_year(self, mock_query):
        """Test that search returns empty when year filter finds no matches."""
        # Mock results with papers from different year
        mock_query.return_value = {
            'documents': [["Paper abstract"]],
            'metadatas': [[
                {
                    'title': 'Old Paper',
                    'authors': 'Author C',
                    'conference': 'Conference 2020',
                    'track': 'Track'
                }
            ]]
        }
        
        response = client.post(
            "/api/search",
            json={"query": "test query", "num_results": 5, "year": "2024"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "No papers found from 2024" in result["response"]
        assert result["papers"] == []


class TestHealthCheck:
    """Test suite for the /api/health endpoint."""
    
    @patch('ollama.list')
    def test_health_check_when_ollama_running(self, mock_ollama_list):
        """Test health check returns healthy status when Ollama is running."""
        mock_ollama_list.return_value = MagicMock()
        
        response = client.get("/api/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "healthy"
        assert result["ollama_running"] is True
    
    @patch('ollama.list')
    def test_health_check_when_ollama_unavailable(self, mock_ollama_list):
        """Test health check returns unhealthy status when Ollama is not running."""
        mock_ollama_list.side_effect = Exception("Connection refused")
        
        response = client.get("/api/health")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "unhealthy"
        assert result["ollama_running"] is False
        assert "error" in result


# Run tests with: pytest test_main.py -v
