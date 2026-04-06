#!/usr/bin/env python3
"""Main training script for weather forecasting system."""

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data import WeatherDataGenerator, WeatherDataProcessor, load_config, setup_logging
from src.models import create_model
from src.eval import ModelEvaluator, create_naive_forecast
from src.viz import WeatherVisualizer

logger = logging.getLogger(__name__)


def main() -> None:
    """Main training pipeline."""
    # Load configuration
    config_path = Path("configs/config.yaml")
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config)
    logger.info("Starting weather forecasting training pipeline")
    
    # Set random seeds for reproducibility
    np.random.seed(config['simulation']['random_seed'])
    torch.manual_seed(config['simulation']['random_seed'])
    
    # Initialize components
    data_generator = WeatherDataGenerator(config)
    data_processor = WeatherDataProcessor(config)
    evaluator = ModelEvaluator(config)
    visualizer = WeatherVisualizer(config)
    
    # Generate data
    logger.info("Generating weather data...")
    X, y = data_generator.generate_weather_data()
    df = data_generator.create_dataframe(X, y)
    
    # Save raw data
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_dir / "weather_data.parquet")
    logger.info(f"Data saved to {data_dir / 'weather_data.parquet'}")
    
    # Prepare data for training
    logger.info("Preparing data for training...")
    X_train, X_test, y_train, y_test = data_processor.prepare_data(X, y)
    
    # Create naive forecast for comparison
    y_naive = create_naive_forecast(y_train, y_test)
    
    # Train and evaluate models
    models_to_train = config['model']['models']
    logger.info(f"Training models: {models_to_train}")
    
    for model_name in models_to_train:
        logger.info(f"Training {model_name}...")
        
        # Create model
        model = create_model(model_name, config)
        
        # Train model
        model.fit(X_train, y_train, X_test, y_test)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions back to original scale
        y_pred_original = data_processor.inverse_transform_targets(y_pred)
        y_test_original = data_processor.inverse_transform_targets(y_test)
        y_naive_original = data_processor.inverse_transform_targets(y_naive)
        
        # Evaluate model
        evaluator.evaluate_model(
            model_name, y_test_original, y_pred_original, y_naive_original
        )
        
        # Save model
        model_dir = Path("assets/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(model_dir / f"{model_name}.pkl")
        
        # Create visualizations
        assets_dir = Path("assets/plots")
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator.plot_predictions(
            model_name, y_test_original, y_pred_original,
            save_path=assets_dir / f"{model_name}_predictions.png"
        )
        
        evaluator.plot_residuals(
            model_name, y_test_original, y_pred_original,
            save_path=assets_dir / f"{model_name}_residuals.png"
        )
        
        evaluator.plot_time_series(
            model_name, y_test_original, y_pred_original,
            save_path=assets_dir / f"{model_name}_time_series.png"
        )
    
    # Create model comparison plots
    logger.info("Creating model comparison visualizations...")
    visualizer.plot_model_comparison(
        evaluator.results,
        save_path=assets_dir / "model_comparison.png"
    )
    
    # Create weather data visualizations
    visualizer.plot_weather_distribution(
        df, save_path=assets_dir / "weather_distribution.png"
    )
    
    visualizer.plot_weather_correlations(
        df, save_path=assets_dir / "weather_correlations.png"
    )
    
    # Save evaluation results
    evaluator.save_results(assets_dir / "evaluation_results.csv")
    
    # Create and display leaderboard
    leaderboard = evaluator.create_leaderboard()
    logger.info("Model Performance Leaderboard:")
    logger.info(f"\n{leaderboard.to_string(index=False)}")
    
    # Save leaderboard
    leaderboard.to_csv(assets_dir / "leaderboard.csv", index=False)
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Results saved to {assets_dir}")


if __name__ == "__main__":
    main()
