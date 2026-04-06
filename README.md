# Weather Forecasting System

A machine learning system for weather forecasting designed for environmental and social applications. This system predicts next-day weather conditions including temperature, humidity, wind speed, and rainfall using multiple machine learning approaches.

## Features

- **Multiple ML Models**: Linear Regression, Random Forest, XGBoost, and Neural Networks
- **Comprehensive Evaluation**: RMSE, MAE, R², SMAPE, and MASE metrics with model leaderboard
- **Interactive Demo**: Streamlit-based web application with real-time forecasting
- **Modern Tech Stack**: PyTorch, Scikit-learn, XGBoost, Plotly, Folium
- **Production Ready**: Type hints, comprehensive logging, YAML configuration
- **Reproducible**: Deterministic seeding and structured project layout

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Weather-Forecasting-System.git
cd Weather-Forecasting-System
```

2. Install dependencies:
```bash
pip install -e .
```

3. Run the training pipeline:
```bash
python scripts/train.py
```

4. Launch the interactive demo:
```bash
streamlit run demo/app.py
```

### Data Schema

The system uses synthetic weather data with the following structure:

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| temperature | Daily average temperature | °C | ~15-35 |
| humidity | Relative humidity | % | 0-100 |
| wind_speed | Average wind speed | km/h | 0-20+ |
| rainfall | Daily precipitation | mm | 0-10+ |

**Target Variables**: Next-day predictions for all four weather variables.

### Configuration

The system is configured via `configs/config.yaml`:

```yaml
# Data generation parameters
simulation:
  n_samples: 1000
  random_seed: 42
  temperature:
    mean: 25.0
    std: 5.0

# Model configuration
model:
  models:
    - "linear_regression"
    - "random_forest" 
    - "xgboost"
    - "neural_network"
```

## Usage

### Training Models

```bash
# Train all configured models
python scripts/train.py

# The script will:
# 1. Generate synthetic weather data
# 2. Train multiple ML models
# 3. Evaluate performance with comprehensive metrics
# 4. Create visualizations and save results
# 5. Generate model leaderboard
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py

# Features:
# - Real-time weather forecasting
# - Interactive maps with weather data
# - Model performance analysis
# - Feature importance visualization
```

### Programmatic Usage

```python
from src.data import WeatherDataGenerator, WeatherDataProcessor
from src.models import create_model
from src.eval import ModelEvaluator
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("configs/config.yaml")

# Generate data
generator = WeatherDataGenerator(config)
X, y = generator.generate_weather_data(1000)

# Prepare data
processor = WeatherDataProcessor(config)
X_train, X_test, y_train, y_test = processor.prepare_data(X, y)

# Train model
model = create_model("xgboost", config)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
evaluator = ModelEvaluator(config)
results = evaluator.evaluate_model("xgboost", y_test, predictions)
```

## Model Performance

The system evaluates models using multiple metrics:

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of Determination
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error

### Sample Results

| Model | RMSE | MAE | R² | SMAPE |
|-------|------|-----|----|----|
| Linear Regression | 2.45 | 1.89 | 0.72 | 8.2% |
| Random Forest | 2.12 | 1.67 | 0.81 | 7.1% |
| XGBoost | 1.98 | 1.54 | 0.85 | 6.8% |
| Neural Network | 2.08 | 1.61 | 0.83 | 7.3% |

## Project Structure

```
weather-forecasting-system/
├── src/                    # Source code
│   ├── data/              # Data handling and preprocessing
│   ├── models/            # ML model implementations
│   ├── eval/              # Evaluation metrics and comparison
│   └── viz/               # Visualization utilities
├── configs/               # Configuration files
├── data/                  # Data storage
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── external/         # External data sources
├── scripts/              # Training and utility scripts
├── demo/                 # Streamlit demo application
├── tests/                # Unit and integration tests
├── assets/               # Generated outputs
│   ├── models/          # Trained models
│   └── plots/           # Visualizations
└── notebooks/           # Jupyter notebooks for exploration
```

## Applications

### Environmental Applications
- **Climate Impact Assessment**: Analyze temperature and precipitation trends
- **Renewable Energy**: Optimize solar and wind energy production
- **Agriculture**: Support smart farming and crop planning
- **Disaster Preparedness**: Early warning systems for extreme weather

### Social Applications  
- **Public Health**: Monitor weather-related health risks
- **Urban Planning**: Optimize city infrastructure for weather patterns
- **Transportation**: Improve traffic and logistics planning
- **Education**: Weather science learning tools

## Technical Details

### Dependencies

**Core ML Stack**:
- PyTorch 2.0+ (Neural Networks)
- Scikit-learn 1.3+ (Traditional ML)
- XGBoost 1.7+ (Gradient Boosting)
- NumPy, Pandas (Data Processing)

**Visualization**:
- Plotly (Interactive plots)
- Folium (Interactive maps)
- Matplotlib, Seaborn (Static plots)

**Infrastructure**:
- Streamlit (Web demo)
- PyYAML, OmegaConf (Configuration)
- Joblib (Model persistence)

### Device Support

The system automatically detects and uses the best available device:
1. **CUDA** (NVIDIA GPUs)
2. **MPS** (Apple Silicon)
3. **CPU** (Fallback)

### Reproducibility

- Deterministic random seeding across all components
- Structured logging with timestamps
- Version-controlled configuration
- Comprehensive model checkpointing

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

### Adding New Models

1. Create a new model class inheriting from `BaseWeatherModel`
2. Implement `fit()` and `predict()` methods
3. Add model to `create_model()` function
4. Update configuration file
5. Add tests

### Adding New Metrics

1. Add metric calculation to `WeatherMetrics` class
2. Update `calculate_all_metrics()` method
3. Include in evaluation pipeline
4. Update visualization functions

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py

# Run integration tests
pytest tests/integration/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run pre-commit hooks
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This is a research demonstration system using synthetic data. For operational weather forecasting, please consult professional meteorological services and use real meteorological data sources.

## Issues and Support

For questions, bug reports, or feature requests, please visit:
- **GitHub Issues**: [https://github.com/kryptologyst](https://github.com/kryptologyst)

## Author

**kryptologyst**  
GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)

---

*This project is part of the Environmental & Social Applications series, focusing on weather forecasting for climate impact assessment and environmental monitoring.*
# Weather-Forecasting-System
