#!/usr/bin/env python3
"""Quick demo script for weather forecasting system."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import WeatherDataGenerator, WeatherDataProcessor, load_config
from src.models import create_model
from src.eval import ModelEvaluator, create_naive_forecast


def quick_demo():
    """Run a quick demonstration of the weather forecasting system."""
    print("🌤️  Weather Forecasting System - Quick Demo")
    print("=" * 50)
    
    # Load configuration
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print("❌ Configuration file not found. Please run from project root.")
        return
    
    config = load_config(config_path)
    print("✅ Configuration loaded")
    
    # Generate sample data
    print("\n📊 Generating weather data...")
    generator = WeatherDataGenerator(config)
    X, y = generator.generate_weather_data(100)  # Small sample for demo
    print(f"   Generated {len(X)} weather samples")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    processor = WeatherDataProcessor(config)
    X_train, X_test, y_train, y_test = processor.prepare_data(X, y)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train a simple model
    print("\n🤖 Training Linear Regression model...")
    model = create_model("linear_regression", config)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\n🔮 Making predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate
    print("\n📈 Evaluating model...")
    evaluator = ModelEvaluator(config)
    y_naive = create_naive_forecast(y_train, y_test)
    
    # Inverse transform for evaluation
    y_test_orig = processor.inverse_transform_targets(y_test)
    y_pred_orig = processor.inverse_transform_targets(y_pred)
    y_naive_orig = processor.inverse_transform_targets(y_naive)
    
    results = evaluator.evaluate_model("linear_regression", y_test_orig, y_pred_orig, y_naive_orig)
    
    # Display results
    print("\n📊 Model Performance:")
    overall = results['overall']
    print(f"   RMSE: {overall['rmse']:.3f}")
    print(f"   MAE:  {overall['mae']:.3f}")
    print(f"   R²:   {overall['r2']:.3f}")
    print(f"   SMAPE: {overall['smape']:.1f}%")
    
    # Show sample predictions
    print("\n🔍 Sample Predictions (True → Predicted):")
    feature_names = config.get('weather_features', ['temperature', 'humidity', 'wind_speed', 'rainfall'])
    for i in range(min(5, len(y_test_orig))):
        print(f"   Sample {i+1}:")
        for j, feature in enumerate(feature_names):
            true_val = y_test_orig[i, j]
            pred_val = y_pred_orig[i, j]
            print(f"     {feature.title()}: {true_val:.1f} → {pred_val:.1f}")
    
    print("\n✅ Demo completed!")
    print("\n🚀 To run the full system:")
    print("   python scripts/train.py")
    print("\n🌐 To launch interactive demo:")
    print("   streamlit run demo/app.py")


if __name__ == "__main__":
    quick_demo()
