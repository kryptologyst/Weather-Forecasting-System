"""Visualization utilities for weather forecasting system."""

import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins

logger = logging.getLogger(__name__)


class WeatherVisualizer:
    """Create visualizations for weather forecasting results."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_names = config.get('weather_features', ['temperature', 'humidity', 'wind_speed', 'rainfall'])
        self.colors = config.get('visualization', {}).get('colors', {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'quaternary': '#d62728'
        })
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        
    def plot_weather_distribution(self, df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
        """Plot distribution of weather variables.
        
        Args:
            df: DataFrame containing weather data
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(self.feature_names):
            ax = axes[i]
            
            # Histogram
            ax.hist(df[feature], bins=30, alpha=0.7, edgecolor='black',
                   color=self.colors['primary'])
            
            ax.set_xlabel(f'{feature.title()}')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {feature.title()}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Weather distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_weather_correlations(self, df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
        """Plot correlation matrix of weather variables.
        
        Args:
            df: DataFrame containing weather data
            save_path: Path to save the plot
        """
        # Select weather features
        weather_cols = [col for col in df.columns if col in self.feature_names or col.startswith('next_')]
        corr_matrix = df[weather_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Weather Variables Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """Plot comparison of different models.
        
        Args:
            results: Dictionary containing model evaluation results
            save_path: Path to save the plot
        """
        models = list(results.keys())
        metrics = ['rmse', 'mae', 'r2', 'smape']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            values = [results[model]['overall'][metric] for model in models]
            
            bars = ax.bar(models, values, color=[self.colors['primary'], self.colors['secondary'],
                                               self.colors['tertiary'], self.colors['quaternary']])
            
            ax.set_ylabel(metric.upper())
            ax.set_title(f'Model Comparison - {metric.upper()}')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_forecast_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       model_name: str, n_samples: int = 100) -> go.Figure:
        """Create interactive forecast plot using Plotly.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            n_samples: Number of samples to plot
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{feature.title()}' for feature in self.feature_names],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        indices = list(range(min(n_samples, len(y_true))))
        
        for i, feature in enumerate(self.feature_names):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=y_true[indices, i],
                    mode='lines',
                    name=f'True {feature.title()}',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=y_pred[indices, i],
                    mode='lines',
                    name=f'Predicted {feature.title()}',
                    line=dict(color=self.colors['secondary'], width=2, dash='dash')
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f'{model_name} Weather Forecast',
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_weather_map(self, df: pd.DataFrame, feature: str, 
                          center_lat: float = 40.7128, center_lon: float = -74.0060,
                          zoom: int = 10) -> folium.Map:
        """Create an interactive weather map.
        
        Args:
            df: DataFrame containing weather data
            feature: Weather feature to visualize
            center_lat: Center latitude
            center_lon: Center longitude
            zoom: Map zoom level
            
        Returns:
            Folium map object
        """
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='OpenStreetMap'
        )
        
        # Add sample points (simulated locations)
        n_points = min(50, len(df))
        sample_df = df.sample(n=n_points, random_state=42)
        
        # Generate random coordinates around center
        np.random.seed(42)
        lats = center_lat + np.random.normal(0, 0.1, n_points)
        lons = center_lon + np.random.normal(0, 0.1, n_points)
        
        # Add markers
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            value = sample_df.iloc[i][feature]
            
            # Color based on value
            if feature == 'temperature':
                color = 'red' if value > 25 else 'blue'
            elif feature == 'humidity':
                color = 'blue' if value > 60 else 'green'
            elif feature == 'wind_speed':
                color = 'orange' if value > 15 else 'green'
            elif feature == 'rainfall':
                color = 'blue' if value > 2 else 'green'
            else:
                color = 'blue'
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=f'{feature.title()}: {value:.2f}',
                color=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
        
        return m
    
    def save_all_plots(self, output_dir: Path) -> None:
        """Save all generated plots to the output directory.
        
        Args:
            output_dir: Directory to save plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"All plots will be saved to {output_dir}")


def create_summary_dashboard(results: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
    """Create a comprehensive dashboard summarizing all results.
    
    Args:
        results: Dictionary containing model evaluation results
        config: Configuration dictionary
        
    Returns:
        Plotly dashboard figure
    """
    models = list(results.keys())
    metrics = ['rmse', 'mae', 'r2', 'smape']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{metric.upper()} Comparison' for metric in metrics],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        values = [results[model]['overall'][metric] for model in models]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric.upper(),
                marker_color=colors[i],
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Weather Forecasting Model Performance Dashboard',
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig
