# Intelligent Energy Management System

## Overview

The Intelligent Energy Management System is a comprehensive solution for forecasting energy demand and optimizing energy allocation across multiple feeders. The system uses machine learning (LSTM models) to predict energy consumption patterns and allocates available energy supply based on priority tiers.

## Features

- **Energy Demand Forecasting**: Utilizes LSTM neural networks to predict energy consumption for multiple feeders.
- **Priority-Based Energy Allocation**: Allocates available energy based on predefined priority tiers.
- **Interactive Dashboard**: Web-based interface built with Streamlit for real-time monitoring and simulation.
- **Comprehensive Visualizations**: Multiple visualization types including:
  - Daily, hourly, and monthly consumption patterns
  - Feeder comparisons and correlations
  - Allocation analysis across different tiers
  - Transformer loading visualizations
- **Model Explainability**: SHAP (SHapley Additive exPlanations) analysis for understanding model predictions.
- **Simulation Capabilities**: Run what-if scenarios to test different allocation strategies.

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Intelligent-Energy-Management
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

Launch the Streamlit application:
```bash
streamlit run app.py
```

The web interface allows you to:
- Configure input parameters (time, date, weekend status)
- Run energy allocation simulations
- View interactive visualizations of forecasts and allocations
- Analyze energy distribution across different feeders

### Exploring the Jupyter Notebook

The [`main.ipynb`](main.ipynb ) notebook contains the research, development, and analysis components:
- Data exploration and preprocessing
- Model development and training
- Feature importance analysis using SHAP
- Performance evaluation and visualization

## Project Structure

```
├── app.py                      # Streamlit web application
├── energy_utils.py             # Utility functions for energy forecasting and allocation
├── main.ipynb                  # Jupyter notebook for development and analysis
├── requirements.txt            # Python dependencies
├── data.xlsx                   # Energy consumption dataset
├── allocation_plots/           # Visualizations for energy allocation
├── energy_plots/               # Energy consumption pattern visualizations
├── prediction_plots/           # Model prediction visualizations
├── training_plots/             # Model training metrics visualizations
├── models/                     # Saved machine learning models
└── logs/                       # Application logs
```

## Key Components

### Energy Forecasting

The system uses LSTM (Long Short-Term Memory) neural networks to forecast energy demand for multiple feeders based on temporal features:
- Hour of day
- Day of week
- Month
- Day of month
- Year
- Weekend status

### Energy Allocation

Energy is allocated based on a tier system:
- **Tier 1**: Critical infrastructure (marked in red)
- **Tier 2**: High-priority facilities (marked in orange)
- **Tier 3**: Medium-priority areas (marked in teal)
- **Tier 4**: Low-priority consumers (marked in gray)

The allocate_energy function distributes available energy supply according to these priority tiers.

## Requirements

The project requires Python 3.6+ and various libraries including:
- TensorFlow for deep learning
- Streamlit for the web