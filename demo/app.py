"""Interactive Streamlit demo for weather forecasting system."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from pathlib import Path
import joblib
import yaml

from src.data import WeatherDataGenerator, WeatherDataProcessor, load_config
from src.models import create_model
from src.eval import ModelEvaluator
from src.viz import WeatherVisualizer, create_summary_dashboard


def load_trained_models() -> dict:
    """Load pre-trained models."""
    models = {}
    model_dir = Path("assets/models")
    
    if not model_dir.exists():
        st.error("No trained models found. Please run the training script first.")
        return models
    
    model_files = {
        "linear_regression": "linear_regression.pkl",
        "random_forest": "random_forest.pkl", 
        "xgboost": "xgboost.pkl",
        "neural_network": "neural_network.pkl"
    }
    
    for model_name, filename in model_files.items():
        model_path = model_dir / filename
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                models[model_name] = model
            except Exception as e:
                st.warning(f"Could not load {model_name}: {e}")
    
    return models


def generate_sample_data(config: dict, n_samples: int = 50) -> tuple:
    """Generate sample weather data for demo."""
    generator = WeatherDataGenerator(config)
    X, y = generator.generate_weather_data(n_samples)
    return X, y


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Weather Forecasting System",
        page_icon="🌤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌤️ Weather Forecasting System")
    st.markdown("A modern machine learning system for predicting next-day weather conditions")
    
    # Load configuration
    try:
        config = load_config(Path("configs/config.yaml"))
    except Exception as e:
        st.error(f"Could not load configuration: {e}")
        return
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    available_models = load_trained_models()
    if not available_models:
        st.stop()
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(available_models.keys()),
        help="Choose a pre-trained model for weather forecasting"
    )
    
    # Number of samples
    n_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=10,
        max_value=200,
        value=50,
        help="Number of weather samples to generate and forecast"
    )
    
    # Show uncertainty
    show_uncertainty = st.sidebar.checkbox(
        "Show Uncertainty",
        value=True,
        help="Display confidence intervals in forecasts"
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast", "🗺️ Map", "📈 Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Weather Forecast")
        
        # Generate sample data
        X_sample, y_sample = generate_sample_data(config, n_samples)
        
        # Make predictions
        model = available_models[selected_model]
        y_pred = model.predict(X_sample)
        
        # Create forecast plot
        feature_names = config['weather_features']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{feature.title()}' for feature in feature_names],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, feature in enumerate(feature_names):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            indices = list(range(len(X_sample)))
            
            # True values
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=y_sample[:, i],
                    mode='lines',
                    name=f'True {feature.title()}',
                    line=dict(color=colors[0], width=2)
                ),
                row=row, col=col
            )
            
            # Predicted values
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=y_pred[:, i],
                    mode='lines',
                    name=f'Predicted {feature.title()}',
                    line=dict(color=colors[1], width=2, dash='dash')
                ),
                row=row, col=col
            )
            
            # Uncertainty bands (simplified)
            if show_uncertainty:
                uncertainty = np.std(y_pred[:, i]) * 1.96  # 95% confidence
                fig.add_trace(
                    go.Scatter(
                        x=indices,
                        y=y_pred[:, i] + uncertainty,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                    )
                
                fig.add_trace(
                    go.Scatter(
                        x=indices,
                        y=y_pred[:, i] - uncertainty,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba({int(colors[1][1:3], 16)}, {int(colors[1][3:5], 16)}, {int(colors[1][5:7], 16)}, 0.2)',
                        name=f'{feature.title()} Uncertainty',
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=f'{selected_model.title()} Weather Forecast',
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sample predictions
        st.subheader("Sample Predictions")
        
        sample_df = pd.DataFrame({
            'Temperature (°C)': [f"{y_sample[i, 0]:.1f} → {y_pred[i, 0]:.1f}" for i in range(min(10, len(y_sample)))],
            'Humidity (%)': [f"{y_sample[i, 1]:.1f} → {y_pred[i, 1]:.1f}" for i in range(min(10, len(y_sample)))],
            'Wind Speed (km/h)': [f"{y_sample[i, 2]:.1f} → {y_pred[i, 2]:.1f}" for i in range(min(10, len(y_sample)))],
            'Rainfall (mm)': [f"{y_sample[i, 3]:.1f} → {y_pred[i, 3]:.1f}" for i in range(min(10, len(y_sample)))]
        })
        
        st.dataframe(sample_df, use_container_width=True)
    
    with tab2:
        st.header("Weather Map")
        
        # Generate sample data for mapping
        X_map, y_map = generate_sample_data(config, 100)
        df_map = pd.DataFrame(X_map, columns=feature_names)
        
        # Map configuration
        map_config = config.get('visualization', {}).get('map', {})
        center_lat = map_config.get('center_lat', 40.7128)
        center_lon = map_config.get('center_lon', -74.0060)
        zoom = map_config.get('zoom', 10)
        
        # Feature selection for mapping
        map_feature = st.selectbox(
            "Select Weather Feature to Map",
            feature_names,
            help="Choose which weather variable to visualize on the map"
        )
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='OpenStreetMap'
        )
        
        # Add sample points
        n_points = min(50, len(df_map))
        sample_df = df_map.sample(n=n_points, random_state=42)
        
        # Generate random coordinates
        np.random.seed(42)
        lats = center_lat + np.random.normal(0, 0.1, n_points)
        lons = center_lon + np.random.normal(0, 0.1, n_points)
        
        # Add markers
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            value = sample_df.iloc[i][map_feature]
            
            # Color based on value
            if map_feature == 'temperature':
                color = 'red' if value > 25 else 'blue'
            elif map_feature == 'humidity':
                color = 'blue' if value > 60 else 'green'
            elif map_feature == 'wind_speed':
                color = 'orange' if value > 15 else 'green'
            elif map_feature == 'rainfall':
                color = 'blue' if value > 2 else 'green'
            else:
                color = 'blue'
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=f'{map_feature.title()}: {value:.2f}',
                color=color,
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
        
        # Display map
        st_folium(m, width=700, height=500)
        
        # Map statistics
        st.subheader("Map Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{df_map[map_feature].mean():.2f}")
        with col2:
            st.metric("Std", f"{df_map[map_feature].std():.2f}")
        with col3:
            st.metric("Min", f"{df_map[map_feature].min():.2f}")
        with col4:
            st.metric("Max", f"{df_map[map_feature].max():.2f}")
    
    with tab3:
        st.header("Model Analysis")
        
        # Load evaluation results if available
        results_path = Path("assets/plots/evaluation_results.csv")
        if results_path.exists():
            try:
                results_df = pd.read_csv(results_path, index_col=0)
                st.subheader("Model Performance Comparison")
                st.dataframe(results_df, use_container_width=True)
                
                # Create performance dashboard
                if len(results_df) > 1:
                    dashboard_fig = create_summary_dashboard(
                        {model: {'overall': results_df.loc[model].to_dict()} 
                         for model in results_df.index}, 
                        config
                    )
                    st.plotly_chart(dashboard_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load evaluation results: {e}")
        else:
            st.info("No evaluation results found. Run the training script to generate performance metrics.")
        
        # Feature importance (for tree-based models)
        if selected_model in ['random_forest', 'xgboost']:
            st.subheader("Feature Importance")
            try:
                model = available_models[selected_model]
                if hasattr(model.model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = go.Figure(data=[
                        go.Bar(x=importance_df['Feature'], y=importance_df['Importance'])
                    ])
                    fig_importance.update_layout(
                        title=f'{selected_model.title()} Feature Importance',
                        xaxis_title='Features',
                        yaxis_title='Importance'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display feature importance: {e}")
    
    with tab4:
        st.header("About This System")
        
        st.markdown("""
        ## Weather Forecasting System
        
        This interactive demo showcases a modern machine learning system for weather forecasting,
        designed for environmental and social applications.
        
        ### Features
        
        - **Multiple Models**: Linear regression, Random Forest, XGBoost, and Neural Networks
        - **Comprehensive Evaluation**: RMSE, MAE, R², SMAPE, and MASE metrics
        - **Interactive Visualizations**: Time series plots, correlation matrices, and maps
        - **Real-time Predictions**: Generate forecasts for next-day weather conditions
        
        ### Weather Variables
        
        - **Temperature** (°C): Daily average temperature
        - **Humidity** (%): Relative humidity levels
        - **Wind Speed** (km/h): Average wind speed
        - **Rainfall** (mm): Daily precipitation amount
        
        ### Applications
        
        - Smart farming and agricultural planning
        - Energy optimization (solar, wind, HVAC)
        - Local weather station dashboards
        - Climate impact assessment
        - Disaster preparedness
        
        ### Technical Details
        
        - **Framework**: PyTorch, Scikit-learn, XGBoost
        - **Visualization**: Plotly, Folium, Matplotlib
        - **Interface**: Streamlit
        - **Configuration**: YAML-based configuration system
        
        ### Disclaimer
        
        This is a research demonstration system using synthetic data. 
        For operational weather forecasting, please consult professional meteorological services.
        """)
        
        st.subheader("Project Information")
        st.markdown("""
        **Author**: kryptologyst  
        **GitHub**: [https://github.com/kryptologyst](https://github.com/kryptologyst)  
        **License**: MIT  
        **Version**: 1.0.0
        """)


if __name__ == "__main__":
    main()
