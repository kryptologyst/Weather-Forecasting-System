"""Evaluation metrics and model comparison for weather forecasting."""

import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class WeatherMetrics:
    """Calculate various metrics for weather forecasting evaluation."""
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            SMAPE score
        """
        return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    
    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_naive: Naive forecast (e.g., previous value)
            
        Returns:
            MASE score
        """
        mae_pred = mean_absolute_error(y_true, y_pred)
        mae_naive = mean_absolute_error(y_true, y_naive)
        return mae_pred / mae_naive if mae_naive != 0 else float('inf')
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_naive: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate all regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_naive: Naive forecast for MASE calculation
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'smape': WeatherMetrics.smape(y_true, y_pred)
        }
        
        if y_naive is not None:
            metrics['mase'] = WeatherMetrics.mase(y_true, y_pred, y_naive)
        
        return metrics


class ModelEvaluator:
    """Evaluate weather forecasting models and create leaderboard."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}
        self.feature_names = config.get('weather_features', ['temperature', 'humidity', 'wind_speed', 'rainfall'])
        
    def evaluate_model(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                      y_naive: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate a single model.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            y_naive: Naive forecast for MASE calculation
            
        Returns:
            Dictionary containing evaluation results
        """
        # Calculate metrics for each weather variable
        results = {}
        
        for i, feature in enumerate(self.feature_names):
            y_true_feature = y_true[:, i]
            y_pred_feature = y_pred[:, i]
            y_naive_feature = y_naive[:, i] if y_naive is not None else None
            
            metrics = WeatherMetrics.calculate_all_metrics(
                y_true_feature, y_pred_feature, y_naive_feature
            )
            
            results[feature] = metrics
        
        # Calculate overall metrics (average across features)
        overall_metrics = {}
        for metric in ['rmse', 'mae', 'r2', 'smape']:
            values = [results[feature][metric] for feature in self.feature_names]
            overall_metrics[metric] = np.mean(values)
        
        if y_naive is not None:
            mase_values = [results[feature]['mase'] for feature in self.feature_names]
            overall_metrics['mase'] = np.mean(mase_values)
        
        results['overall'] = overall_metrics
        
        # Store results
        self.results[model_name] = results
        
        logger.info(f"Evaluated {model_name}: RMSE={overall_metrics['rmse']:.4f}, "
                   f"MAE={overall_metrics['mae']:.4f}, R²={overall_metrics['r2']:.4f}")
        
        return results
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create a leaderboard comparing all evaluated models.
        
        Returns:
            DataFrame with model rankings
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")
        
        leaderboard_data = []
        
        for model_name, results in self.results.items():
            overall_metrics = results['overall']
            leaderboard_data.append({
                'Model': model_name,
                'RMSE': overall_metrics['rmse'],
                'MAE': overall_metrics['mae'],
                'R²': overall_metrics['r2'],
                'SMAPE': overall_metrics['smape'],
                'MASE': overall_metrics.get('mase', np.nan)
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        
        # Sort by RMSE (lower is better)
        leaderboard = leaderboard.sort_values('RMSE').reset_index(drop=True)
        leaderboard['Rank'] = range(1, len(leaderboard) + 1)
        
        return leaderboard
    
    def plot_predictions(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                        save_path: Optional[Path] = None) -> None:
        """Plot predictions vs actual values for a model.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(self.feature_names):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Labels and title
            ax.set_xlabel(f'True {feature.title()}')
            ax.set_ylabel(f'Predicted {feature.title()}')
            ax.set_title(f'{model_name}: {feature.title()} Predictions')
            
            # Add R² score
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[Path] = None) -> None:
        """Plot residuals for a model.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(self.feature_names):
            ax = axes[i]
            
            residuals = y_true[:, i] - y_pred[:, i]
            
            # Histogram of residuals
            ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', alpha=0.8)
            
            ax.set_xlabel(f'Residuals ({feature.title()})')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{model_name}: {feature.title()} Residuals')
            
            # Add statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ax.text(0.05, 0.95, f'Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residuals plot saved to {save_path}")
        
        plt.show()
    
    def plot_time_series(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                        n_samples: int = 100, save_path: Optional[Path] = None) -> None:
        """Plot time series predictions for a subset of data.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            n_samples: Number of samples to plot
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot first n_samples
        indices = range(min(n_samples, len(y_true)))
        
        for i, feature in enumerate(self.feature_names):
            ax = axes[i]
            
            ax.plot(indices, y_true[indices, i], label='True', alpha=0.8)
            ax.plot(indices, y_pred[indices, i], label='Predicted', alpha=0.8)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(f'{feature.title()}')
            ax.set_title(f'{model_name}: {feature.title()} Time Series')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filepath: Path) -> None:
        """Save evaluation results to file.
        
        Args:
            filepath: Path to save the results
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(filepath)
        
        # Save leaderboard
        leaderboard = self.create_leaderboard()
        leaderboard_path = filepath.parent / f"{filepath.stem}_leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        
        logger.info(f"Results saved to {filepath}")
        logger.info(f"Leaderboard saved to {leaderboard_path}")


def create_naive_forecast(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Create naive forecast using the last training value.
    
    Args:
        y_train: Training targets
        y_test: Test targets
        
    Returns:
        Naive forecast for test data
    """
    # Use the last training value as forecast for all test samples
    last_train_value = y_train[-1]
    naive_forecast = np.tile(last_train_value, (len(y_test), 1))
    return naive_forecast
