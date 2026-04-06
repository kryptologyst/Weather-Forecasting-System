"""Weather forecasting models implementation."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

logger = logging.getLogger(__name__)


class BaseWeatherModel(ABC):
    """Abstract base class for weather forecasting models."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit the model to training data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        pass
    
    def save_model(self, filepath: Path) -> None:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path) -> None:
        """Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


class LinearRegressionModel(BaseWeatherModel):
    """Linear regression model for weather forecasting."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize linear regression model."""
        super().__init__(config)
        self.model = LinearRegression()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit linear regression model."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("Linear regression model fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using linear regression."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class RandomForestModel(BaseWeatherModel):
    """Random forest model for weather forecasting."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize random forest model."""
        super().__init__(config)
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit random forest model."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("Random forest model fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using random forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class XGBoostModel(BaseWeatherModel):
    """XGBoost model for weather forecasting."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize XGBoost model."""
        super().__init__(config)
        xgb_config = config.get('model', {}).get('xgboost', {})
        self.model = xgb.XGBRegressor(
            n_estimators=xgb_config.get('n_estimators', 100),
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            subsample=xgb_config.get('subsample', 0.8),
            colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
            random_state=xgb_config.get('random_state', 42),
            n_jobs=-1
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit XGBoost model."""
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.is_fitted = True
        logger.info("XGBoost model fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using XGBoost."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class WeatherNeuralNetwork(nn.Module):
    """Neural network architecture for weather forecasting."""
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], 
                 dropout_rate: float = 0.2) -> None:
        """Initialize neural network.
        
        Args:
            input_size: Number of input features
            output_size: Number of output targets
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class NeuralNetworkModel(BaseWeatherModel):
    """Neural network model for weather forecasting."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize neural network model."""
        super().__init__(config)
        
        # Get device
        self.device = self._get_device()
        
        # Model configuration
        nn_config = config.get('model', {}).get('neural_network', {})
        self.hidden_sizes = nn_config.get('hidden_layers', [64, 32])
        self.epochs = nn_config.get('epochs', 100)
        self.batch_size = nn_config.get('batch_size', 32)
        self.learning_rate = nn_config.get('learning_rate', 0.001)
        
        # Initialize model
        self.model = WeatherNeuralNetwork(
            input_size=4,  # temperature, humidity, wind_speed, rainfall
            output_size=4,  # next day predictions
            hidden_sizes=self.hidden_sizes
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Early stopping
        self.early_stopping = nn_config.get('early_stopping', {})
        self.patience = self.early_stopping.get('patience', 10)
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit neural network model."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        logger.info("Neural network model fitted")
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
        
        self.model.train()
        return val_loss / len(val_loader)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using neural network."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()


def create_model(model_name: str, config: Dict[str, Any]) -> BaseWeatherModel:
    """Create a model instance based on the model name.
    
    Args:
        model_name: Name of the model to create
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    model_map = {
        'linear_regression': LinearRegressionModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'neural_network': NeuralNetworkModel
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_map.keys())}")
    
    return model_map[model_name](config)
