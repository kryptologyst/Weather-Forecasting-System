"""Core data handling and preprocessing for weather forecasting system."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml

logger = logging.getLogger(__name__)


class WeatherDataGenerator:
    """Generate synthetic weather data for demonstration purposes."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the weather data generator.
        
        Args:
            config: Configuration dictionary containing data generation parameters.
        """
        self.config = config
        self.scaler = StandardScaler()
        
    def generate_weather_data(self, n_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic weather data.
        
        Args:
            n_samples: Number of samples to generate. If None, uses config default.
            
        Returns:
            Tuple of (features, targets) where features are current day weather
            and targets are next day weather predictions.
        """
        if n_samples is None:
            n_samples = self.config.get('simulation', {}).get('n_samples', 1000)
            
        # Set random seed for reproducibility
        random_seed = self.config.get('simulation', {}).get('random_seed', 42)
        np.random.seed(random_seed)
        
        # Current day weather features
        temp_config = self.config.get('simulation', {}).get('temperature', {'mean': 25.0, 'std': 5.0})
        humidity_config = self.config.get('simulation', {}).get('humidity', {'mean': 60.0, 'std': 10.0})
        wind_config = self.config.get('simulation', {}).get('wind_speed', {'mean': 10.0, 'std': 2.0})
        rain_config = self.config.get('simulation', {}).get('rainfall', {'mean': 2.0, 'std': 1.5})
        
        temp_today = np.random.normal(temp_config['mean'], temp_config['std'], n_samples)
        humidity_today = np.random.normal(humidity_config['mean'], humidity_config['std'], n_samples)
        wind_today = np.random.normal(wind_config['mean'], wind_config['std'], n_samples)
        rain_today = np.random.normal(rain_config['mean'], rain_config['std'], n_samples)
        
        # Simulate next-day values with trend and noise
        temp_next = temp_today + np.random.normal(0, 1.5, n_samples)
        humidity_next = humidity_today + np.random.normal(0, 5, n_samples)
        wind_next = wind_today + np.random.normal(0, 1, n_samples)
        rain_next = rain_today + np.random.normal(0, 1, n_samples)
        
        # Ensure non-negative values for physical constraints
        rain_today = np.maximum(rain_today, 0)
        rain_next = np.maximum(rain_next, 0)
        wind_today = np.maximum(wind_today, 0)
        wind_next = np.maximum(wind_next, 0)
        humidity_today = np.clip(humidity_today, 0, 100)
        humidity_next = np.clip(humidity_next, 0, 100)
        
        # Feature matrix and multi-output labels
        X = np.stack([temp_today, humidity_today, wind_today, rain_today], axis=1)
        y = np.stack([temp_next, humidity_next, wind_next, rain_next], axis=1)
        
        logger.info(f"Generated {n_samples} weather samples")
        logger.info(f"Feature shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def create_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Convert numpy arrays to pandas DataFrame with proper column names.
        
        Args:
            X: Feature matrix (current day weather)
            y: Target matrix (next day weather)
            
        Returns:
            Combined DataFrame with current and next day weather data.
        """
        feature_names = self.config.get('weather_features', ['temperature', 'humidity', 'wind_speed', 'rainfall'])
        target_names = [f"next_{name}" for name in feature_names]
        
        # Create DataFrame
        data = {}
        for i, name in enumerate(feature_names):
            data[name] = X[:, i]
            data[target_names[i]] = y[:, i]
            
        df = pd.DataFrame(data)
        
        # Add temporal features
        df['day_of_year'] = np.random.randint(1, 366, len(df))
        df['month'] = ((df['day_of_year'] - 1) // 30) + 1
        df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                       3: 1, 4: 1, 5: 1,   # Spring
                                       6: 2, 7: 2, 8: 2,   # Summer
                                       9: 3, 10: 3, 11: 3}) # Fall
        
        return df


class WeatherDataProcessor:
    """Process and prepare weather data for modeling."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the data processor.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, 
                                                                 np.ndarray, np.ndarray]:
        """Prepare data for training and testing.
        
        Args:
            X: Feature matrix
            y: Target matrix
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Train-test split
        test_size = self.config['model']['train_test_split']
        random_seed = self.config['model']['random_seed']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed
        )
        
        # Scale features and targets
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        logger.info(f"Data split: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    
    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled targets back to original scale.
        
        Args:
            y_scaled: Scaled target values
            
        Returns:
            Targets in original scale
        """
        return self.scaler_y.inverse_transform(y_scaled)
    
    def save_data(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save processed data to file.
        
        Args:
            data: Dictionary containing data arrays
            filepath: Path to save the data
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.npz':
            np.savez(filepath, **data)
        elif filepath.suffix == '.parquet':
            df = pd.DataFrame(data)
            df.to_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
        logger.info(f"Data saved to {filepath}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration.
    
    Args:
        config: Configuration dictionary containing logging settings
    """
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_file = Path(log_config.get('file', 'logs/weather_forecasting.log'))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
