"""Unit tests for weather forecasting system."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.data import WeatherDataGenerator, WeatherDataProcessor, load_config
from src.models import create_model, LinearRegressionModel, RandomForestModel, XGBoostModel, NeuralNetworkModel
from src.eval import WeatherMetrics, ModelEvaluator, create_naive_forecast
from src.viz import WeatherVisualizer


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'weather_features': ['temperature', 'humidity', 'wind_speed', 'rainfall'],
        'simulation': {
            'n_samples': 100,
            'random_seed': 42,
            'temperature': {'mean': 25.0, 'std': 5.0},
            'humidity': {'mean': 60.0, 'std': 10.0},
            'wind_speed': {'mean': 10.0, 'std': 2.0},
            'rainfall': {'mean': 2.0, 'std': 1.5}
        },
        'model': {
            'train_test_split': 0.2,
            'random_seed': 42,
            'neural_network': {
                'hidden_layers': [32, 16],
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 0.01
            }
        }
    }


@pytest.fixture
def sample_data(sample_config):
    """Generate sample weather data for testing."""
    generator = WeatherDataGenerator(sample_config)
    X, y = generator.generate_weather_data(100)
    return X, y


class TestWeatherDataGenerator:
    """Test weather data generation."""
    
    def test_generate_weather_data(self, sample_config):
        """Test weather data generation."""
        generator = WeatherDataGenerator(sample_config)
        X, y = generator.generate_weather_data(50)
        
        assert X.shape == (50, 4)
        assert y.shape == (50, 4)
        assert np.all(X >= 0)  # Non-negative values
        assert np.all(y >= 0)  # Non-negative values
    
    def test_create_dataframe(self, sample_config, sample_data):
        """Test DataFrame creation."""
        generator = WeatherDataGenerator(sample_config)
        X, y = sample_data
        df = generator.create_dataframe(X, y)
        
        assert len(df) == len(X)
        assert 'temperature' in df.columns
        assert 'next_temperature' in df.columns
        assert 'season' in df.columns


class TestWeatherDataProcessor:
    """Test data processing."""
    
    def test_prepare_data(self, sample_config, sample_data):
        """Test data preparation."""
        processor = WeatherDataProcessor(sample_config)
        X, y = sample_data
        
        X_train, X_test, y_train, y_test = processor.prepare_data(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert X_train.shape[1] == X.shape[1]
        assert y_train.shape[1] == y.shape[1]
    
    def test_inverse_transform(self, sample_config, sample_data):
        """Test inverse transformation."""
        processor = WeatherDataProcessor(sample_config)
        X, y = sample_data
        X_train, X_test, y_train, y_test = processor.prepare_data(X, y)
        
        y_original = processor.inverse_transform_targets(y_test)
        assert y_original.shape == y_test.shape


class TestWeatherModels:
    """Test weather forecasting models."""
    
    def test_linear_regression_model(self, sample_config, sample_data):
        """Test linear regression model."""
        X, y = sample_data
        processor = WeatherDataProcessor(sample_config)
        X_train, X_test, y_train, y_test = processor.prepare_data(X, y)
        
        model = LinearRegressionModel(sample_config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert model.is_fitted
    
    def test_random_forest_model(self, sample_config, sample_data):
        """Test random forest model."""
        X, y = sample_data
        processor = WeatherDataProcessor(sample_config)
        X_train, X_test, y_train, y_test = processor.prepare_data(X, y)
        
        model = RandomForestModel(sample_config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert model.is_fitted
    
    def test_xgboost_model(self, sample_config, sample_data):
        """Test XGBoost model."""
        X, y = sample_data
        processor = WeatherDataProcessor(sample_config)
        X_train, X_test, y_train, y_test = processor.prepare_data(X, y)
        
        model = XGBoostModel(sample_config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert model.is_fitted
    
    def test_neural_network_model(self, sample_config, sample_data):
        """Test neural network model."""
        X, y = sample_data
        processor = WeatherDataProcessor(sample_config)
        X_train, X_test, y_train, y_test = processor.prepare_data(X, y)
        
        model = NeuralNetworkModel(sample_config)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert model.is_fitted
    
    def test_create_model(self, sample_config):
        """Test model creation function."""
        models = ['linear_regression', 'random_forest', 'xgboost', 'neural_network']
        
        for model_name in models:
            model = create_model(model_name, sample_config)
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')


class TestWeatherMetrics:
    """Test weather evaluation metrics."""
    
    def test_smape(self):
        """Test SMAPE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        smape = WeatherMetrics.smape(y_true, y_pred)
        assert isinstance(smape, float)
        assert smape >= 0
    
    def test_mase(self):
        """Test MASE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        y_naive = np.array([1, 2, 3, 4, 5])
        
        mase = WeatherMetrics.mase(y_true, y_pred, y_naive)
        assert isinstance(mase, float)
        assert mase >= 0
    
    def test_calculate_all_metrics(self):
        """Test calculation of all metrics."""
        y_true = np.array([[1, 2], [3, 4], [5, 6]])
        y_pred = np.array([[1.1, 1.9], [3.1, 3.9], [5.1, 5.9]])
        
        metrics = WeatherMetrics.calculate_all_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'smape' in metrics
        assert all(isinstance(v, float) for v in metrics.values())


class TestModelEvaluator:
    """Test model evaluation."""
    
    def test_evaluate_model(self, sample_config, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        processor = WeatherDataProcessor(sample_config)
        X_train, X_test, y_train, y_test = processor.prepare_data(X, y)
        
        model = LinearRegressionModel(sample_config)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        evaluator = ModelEvaluator(sample_config)
        results = evaluator.evaluate_model('test_model', y_test, y_pred)
        
        assert 'overall' in results
        assert 'temperature' in results
        assert 'rmse' in results['overall']
    
    def test_create_leaderboard(self, sample_config, sample_data):
        """Test leaderboard creation."""
        X, y = sample_data
        processor = WeatherDataProcessor(sample_config)
        X_train, X_test, y_train, y_test = processor.prepare_data(X, y)
        
        evaluator = ModelEvaluator(sample_config)
        
        # Evaluate multiple models
        for model_name in ['linear_regression', 'random_forest']:
            model = create_model(model_name, sample_config)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            evaluator.evaluate_model(model_name, y_test, y_pred)
        
        leaderboard = evaluator.create_leaderboard()
        assert len(leaderboard) == 2
        assert 'Model' in leaderboard.columns
        assert 'RMSE' in leaderboard.columns


class TestWeatherVisualizer:
    """Test weather visualization."""
    
    def test_visualizer_initialization(self, sample_config):
        """Test visualizer initialization."""
        visualizer = WeatherVisualizer(sample_config)
        assert visualizer.feature_names == sample_config['weather_features']
        assert 'primary' in visualizer.colors


def test_create_naive_forecast(sample_data):
    """Test naive forecast creation."""
    X, y = sample_data
    y_train = y[:80]
    y_test = y[80:]
    
    naive_forecast = create_naive_forecast(y_train, y_test)
    
    assert naive_forecast.shape == y_test.shape
    assert np.allclose(naive_forecast, y_train[-1])


def test_load_config():
    """Test configuration loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
weather_features: ['temp', 'humidity']
simulation:
  n_samples: 100
  random_seed: 42
""")
        config_path = Path(f.name)
    
    try:
        config = load_config(config_path)
        assert 'weather_features' in config
        assert config['simulation']['n_samples'] == 100
    finally:
        config_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
