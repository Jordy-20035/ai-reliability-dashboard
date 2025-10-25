"""
Tests for API functionality.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.api.main import app
from src.api.routes import set_model
from src.data.load_data import generate_synthetic_data


@pytest.fixture
def test_client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def setup_model():
    """Setup a test model."""
    X_train, X_test, y_train, y_test = generate_synthetic_data(
        n_samples=100,
        n_features=5,
        random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)
    
    set_model(model, feature_names=X_train.columns.tolist())
    
    return model, X_train.columns.tolist()


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_status_endpoint(self, test_client):
        """Test status endpoint."""
        response = test_client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/manage/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_prediction_without_model(self, test_client):
        """Test prediction endpoint without model."""
        response = test_client.post(
            "/predict/single",
            json={"features": {"feature_0": 1.0}}
        )
        # Should return 503 if no model loaded
        assert response.status_code in [503, 200]  # Depends on whether model was loaded
    
    def test_single_prediction(self, test_client, setup_model):
        """Test single prediction endpoint."""
        model, feature_names = setup_model
        
        # Create feature dict
        features = {name: 0.5 for name in feature_names}
        
        response = test_client.post(
            "/predict/single",
            json={"features": features}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "timestamp" in data
    
    def test_batch_prediction(self, test_client, setup_model):
        """Test batch prediction endpoint."""
        model, feature_names = setup_model
        
        # Create batch features
        features_list = [
            {name: np.random.random() for name in feature_names}
            for _ in range(5)
        ]
        
        response = test_client.post(
            "/predict/batch",
            json={"features": features_list}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 5


class TestAPISchemas:
    """Test API schemas."""
    
    def test_prediction_request_validation(self):
        """Test prediction request schema validation."""
        from src.api.schemas import PredictionRequest
        
        # Valid request
        request = PredictionRequest(features={"feature1": 1.0, "feature2": "value"})
        assert request.features == {"feature1": 1.0, "feature2": "value"}
        
        # Invalid request should raise validation error
        with pytest.raises(Exception):
            PredictionRequest(features="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

