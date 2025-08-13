import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import tensorflow as tf
#from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("energy_allocation_app")

# Log app startup
logger.info("Starting Intelligent Energy Allocation System application")

# Import from energy_utils instead of redefining
try:
    from energy_utils import (
        create_cyclical_features,
        prepare_input_features,
        allocate_energy,
        analyze_allocation,
        TIER1, TIER2, TIER3, TIER4,
        PRIORITY_ORDER, TRANSFORMERS,
        FEEDER_COLS
    )
    logger.info("Successfully imported utility functions from energy_utils")
except ImportError as e:
    logger.error(f"Failed to import from energy_utils: {str(e)}")
    st.error("Error: Could not load required utilities. Please check the application setup.")
    raise

# Set page configuration
st.set_page_config(
    page_title="Intelligent Energy Allocation System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .warning {
        color: #FF5722;
        font-weight: bold;
    }
    .success {
        color: #4CAF50;
        font-weight: bold;
    }
    .tier1 {
        background-color: rgba(220, 0, 0, 0.1);
        border-left: 5px solid darkred;
        padding: 10px;
    }
    .tier2 {
        background-color: rgba(255, 140, 0, 0.1);
        border-left: 5px solid orange;
        padding: 10px;
    }
    .tier3 {
        background-color: rgba(0, 128, 128, 0.1);
        border-left: 5px solid teal;
        padding: 10px;
    }
    .tier4 {
        background-color: rgba(128, 128, 128, 0.1);
        border-left: 5px solid gray;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load the model and scalers
@st.cache_resource
def load_prediction_resources():
    """Load the LSTM model and scalers"""
    try:
        # Check if model files exist
        model_path = os.path.join('models', 'lstm_model.keras')
        feature_scaler_path = os.path.join('models', 'feature_scaler.pkl')
        target_scaler_path = os.path.join('models', 'target_scaler.pkl')
        
        logger.info(f"Attempting to load model from {model_path}")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            return None, None, None, "Model file not found. Please train the model first."
        
        if not os.path.exists(feature_scaler_path) or not os.path.exists(target_scaler_path):
            logger.warning(f"Scaler files not found: {feature_scaler_path} or {target_scaler_path}")
            return None, None, None, "Scaler files not found. Please train the model first."
        
        # Load model and scalers
        logger.info("Loading LSTM model...")
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully. Summary: {model.summary()}")
        
        logger.info("Loading feature scaler...")
        try:
            feature_scaler = joblib.load(feature_scaler_path)
            if feature_scaler is None:
                logger.error("Feature scaler loaded as None. File may be corrupted.")
                return None, None, None, "Feature scaler file is corrupted or unreadable."
            logger.info("Feature scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading feature scaler: {str(e)}", exc_info=True)
            return None, None, None, f"Error loading feature scaler: {str(e)}"
        
        logger.info("Loading target scaler...")
        target_scaler = joblib.load(target_scaler_path)
        logger.info("Target scaler loaded successfully")
        
        return model, feature_scaler, target_scaler, None
    
    except Exception as e:
        logger.error(f"Error loading model resources: {str(e)}", exc_info=True)
        return None, None, None, f"Error loading model: {str(e)}"

# Function to create a fake model and scalers for demo purposes
@st.cache_resource
def create_demo_resources():
    """Create demo model and scalers for testing the app without the real model"""
    logger.info("Creating demo resources for testing")
    
    # Create a simple model that returns random predictions
    class DemoModel:
        def predict(self, X):
            batch_size = X.shape[0]
            # Generate random predictions for 24 hours forecast and 12 feeders
            logger.debug(f"Demo model predicting with input shape: {X.shape}")
            return np.random.rand(batch_size, 24, 12) * 0.5
    
    # Create demo scalers
    class DemoScaler:
        def transform(self, X):
            logger.debug(f"Demo scaler transforming data with shape: {X.shape}")
            return X * 0.5
        
        def inverse_transform(self, X):
            logger.debug(f"Demo scaler inverse transforming data with shape: {X.shape}")
            return X * 2.0
    
    model = DemoModel()
    feature_scaler = DemoScaler()
    target_scaler = DemoScaler()
    
    logger.info("Demo resources created successfully")
    return model, feature_scaler, target_scaler

# Function to generate input sequence for model
def generate_input_sequence(feature_scaler, input_features):
    """Generate input sequence for the LSTM model"""
    logger.info(f"Generating input sequence with shape: {input_features.shape}")
    
    # Scale the features
    try:
        scaled_features = feature_scaler.transform(input_features)
        logger.debug(f"Features scaled, new shape: {scaled_features.shape}")
        
        # Repeat the features to create a sequence
        sequence = np.repeat(scaled_features, 24, axis=0).reshape(1, 24, -1)
        logger.info(f"Input sequence generated with shape: {sequence.shape}")
        
        return sequence
    except Exception as e:
        logger.error(f"Error generating input sequence: {str(e)}", exc_info=True)
        raise

# Function to create an allocation visualization
def create_allocation_chart(allocation, forecasted_demand):
    """Create an interactive allocation chart using Plotly"""
    # Prepare data for visualization
    feeders = list(allocation.keys())
    tier_colors = {
        'SECURITY': 'darkred',
        'HEALTHCARE': 'red',
        'FINANCIAL': 'orange',
        'INDUSTRIAL': 'teal',
        'GENERAL': 'gray'
    }
    
    # Map feeders to their tiers for coloring
    feeder_tiers = []
    colors = []
    
    for feeder in feeders:
        if feeder in TIER1['SECURITY']:
            feeder_tiers.append('TIER1 (Security)')
            colors.append(tier_colors['SECURITY'])
        elif feeder in TIER2['HEALTHCARE']:
            feeder_tiers.append('TIER2 (Healthcare)')
            colors.append(tier_colors['HEALTHCARE'])
        elif feeder in TIER3['FINANCIAL']:
            feeder_tiers.append('TIER3 (Financial)')
            colors.append(tier_colors['FINANCIAL'])
        else:
            feeder_tiers.append('TIER4 (General)')
            colors.append(tier_colors['GENERAL'])
    
    # Sort by priority order
    sort_idx = [feeders.index(f) for f in PRIORITY_ORDER if f in feeders]
    feeders_sorted = [feeders[i] for i in sort_idx]
    feeder_tiers_sorted = [feeder_tiers[i] for i in sort_idx]
    colors_sorted = [colors[i] for i in sort_idx]
    
    # Prepare data for allocation vs demand bar chart
    demanded = [forecasted_demand[f] for f in feeders_sorted]
    allocated = [allocation[f] for f in feeders_sorted]
    allocation_pct = [allocation[f]/forecasted_demand[f]*100 if forecasted_demand[f] > 0 else 0 for f in feeders_sorted]
    
    # Create allocation vs demand chart
    fig1 = go.Figure()
    
    # Add demand bars
    fig1.add_trace(go.Bar(
        x=feeders_sorted,
        y=demanded,
        name='Demand',
        marker_color='lightgray',
        opacity=0.7
    ))
    
    # Add allocation bars
    fig1.add_trace(go.Bar(
        x=feeders_sorted,
        y=allocated,
        name='Allocated',
        marker_color=colors_sorted,
        text=[f"{a:.2f} MW" for a in allocated],
        textposition='auto'
    ))
    
    # Update layout
    fig1.update_layout(
        title='Energy Allocation vs Demand by Feeder',
        xaxis_title='Feeder',
        yaxis_title='Energy (MW)',
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        margin=dict(t=50, b=100)
    )
    
    # Create percentage allocation chart
    fig2 = go.Figure()
    
    # Add percentage bars
    fig2.add_trace(go.Bar(
        x=feeders_sorted,
        y=allocation_pct,
        marker_color=colors_sorted,
        text=[f"{p:.1f}%" for p in allocation_pct],
        textposition='auto'
    ))
    
    # Add 100% line
    fig2.add_shape(
        type="line",
        x0=-0.5,
        y0=100,
        x1=len(feeders_sorted)-0.5,
        y1=100,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        )
    )
    
    # Update layout
    fig2.update_layout(
        title='Energy Allocation Percentage by Feeder',
        xaxis_title='Feeder',
        yaxis_title='Allocation Percentage (%)',
        hovermode="x unified",
        margin=dict(t=50, b=100)
    )
    
    return fig1, fig2

# Function to create tier allocation visualization
def create_tier_charts(allocation, forecasted_demand):
    """Create tier-based allocation charts"""
    # Prepare data for tier-based pie charts
    tier_names = ['Security', 'Healthcare', 'Financial', 'General']
    tier_colors = ['darkred', 'red', 'orange', 'teal']

    tier_demands = [
        sum(forecasted_demand.get(f, 0) for f in TIER1['SECURITY']),
        sum(forecasted_demand.get(f, 0) for f in TIER2['HEALTHCARE']),
        sum(forecasted_demand.get(f, 0) for f in TIER3['FINANCIAL']),
        sum(forecasted_demand.get(f, 0) for f in TIER4['GENERAL'])
    ]
    
    tier_allocated = [
        sum(allocation.get(f, 0) for f in TIER1['SECURITY']),
        sum(allocation.get(f, 0) for f in TIER2['HEALTHCARE']),
        sum(allocation.get(f, 0) for f in TIER3['FINANCIAL']),
        sum(allocation.get(f, 0) for f in TIER4['GENERAL'])
    ]
    
    # Create subplots with 2 pie charts
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                       subplot_titles=['Demand Distribution by Priority Tier', 
                                      'Allocation Distribution by Priority Tier'])
    
    # Add demand pie chart
    fig.add_trace(go.Pie(
        labels=tier_names,
        values=tier_demands,
        textinfo='label+percent',
        marker=dict(colors=tier_colors),
        domain=dict(row=0, column=0)
    ), 1, 1)
    
    # Add allocation pie chart
    fig.add_trace(go.Pie(
        labels=tier_names,
        values=tier_allocated,
        textinfo='label+percent',
        marker=dict(colors=tier_colors),
        domain=dict(row=0, column=1)
    ), 1, 2)
    
    # Update layout
    fig.update_layout(
        title_text="Tier-Based Energy Distribution",
        height=500,
        margin=dict(t=80, b=20)
    )
    
    return fig

# Function to create transformer loading visualization
def create_transformer_chart(allocation):
    """Create a chart showing transformer loading"""
    tx_names = []
    tx_capacities = []
    tx_allocations = []
    tx_utilizations = []
    
    # Calculate transformer loading
    for tx_name, tx_info in TRANSFORMERS.items():
        tx_capacity = tx_info['capacity']
        tx_feeders = tx_info['feeders']
        tx_allocation = sum(allocation.get(f, 0) for f in tx_feeders)
        tx_utilization = tx_allocation / tx_capacity * 100
        
        tx_names.append(tx_name)
        tx_capacities.append(tx_capacity)
        tx_allocations.append(tx_allocation)
        tx_utilizations.append(tx_utilization)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for allocation
    fig.add_trace(
        go.Bar(
            x=tx_names,
            y=tx_allocations,
            name="Allocated Energy (MW)",
            marker_color='teal',
            text=[f"{a:.2f} MW" for a in tx_allocations],
            textposition='auto'
        ),
        secondary_y=False,
    )
    
    # Add bar chart for capacity
    fig.add_trace(
        go.Bar(
            x=tx_names,
            y=tx_capacities,
            name="Capacity (MW)",
            marker_color='lightgray',
            opacity=0.5
        ),
        secondary_y=False,
    )
    
    # Add line chart for utilization
    fig.add_trace(
        go.Scatter(
            x=tx_names,
            y=tx_utilizations,
            name="Utilization (%)",
            mode='lines+markers',
            marker=dict(color='red', size=10),
            line=dict(width=2, dash='dot')
        ),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_layout(
        title_text="Transformer Loading Analysis",
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=80, b=20)
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Energy (MW)", secondary_y=False)
    fig.update_yaxes(title_text="Utilization (%)", secondary_y=True)
    
    return fig

# Create directory for saving simulations if it doesn't exist
if not os.path.exists('simulations'):
    os.makedirs('simulations')

# Main application
def main():
    logger.info("Initializing main application")
    
    # App header
    st.markdown("<h1 class='main-header'>Intelligent Energy Allocation System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2553/2553624.png", width=80)
    st.sidebar.title("Configuration")
    
    # Load model and scalers
    logger.info("Loading prediction resources")
    model, feature_scaler, target_scaler, error_message = load_prediction_resources()
    
    # If model loading failed, create demo resources for testing
    # if model is None:
    #     logger.warning(f"Failed to load model: {error_message}. Switching to demo mode.")
    #     st.sidebar.warning("⚠️ Using demo mode: Model files not found")
    #     model, feature_scaler, target_scaler = create_demo_resources()
    
    if model is not None:
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Failed to load model: {error_message}. Switching to demo mode.")
        st.sidebar.warning("⚠️ Using demo mode: Model files not found")
        model, feature_scaler, target_scaler = create_demo_resources()

    # Input parameters section
    logger.info("Setting up user input parameters")
    st.sidebar.subheader("Date and Time Parameters")
    
    # Date input
    date_input = st.sidebar.date_input(
        "Select date",
        datetime.date.today()
    )
    
    # Time input
    hour_input = st.sidebar.slider(
        "Hour of day (0-23)",
        min_value=0,
        max_value=23,
        value=12
    )
    
    # Extract date components
    day = date_input.day
    month = date_input.month
    year = date_input.year
    day_of_week = date_input.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    logger.info(f"User selected date: {date_input}, hour: {hour_input}")
    logger.debug(f"Date components: day={day}, month={month}, year={year}, day_of_week={day_of_week}, is_weekend={is_weekend}")
    
    # Supply variation
    st.sidebar.subheader("Supply Configuration")
    supply_variation = st.sidebar.slider(
        "Supply variation (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Adjust available supply as a percentage of the forecasted demand"
    )
    
    # Custom supply
    use_custom_supply = st.sidebar.checkbox("Use custom supply value")
    custom_supply = None
    if use_custom_supply:
        custom_supply = st.sidebar.number_input(
            "Custom supply (MW)",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=5.0
        )
        logger.info(f"User set custom supply: {custom_supply} MW")
    else:
        logger.info(f"User set supply variation: {supply_variation}%")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This application uses an LSTM model to forecast energy demand "
        "and intelligently allocates available supply based on priority tiers."
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h2 class='sub-header'>Input Parameters</h2>", unsafe_allow_html=True)
        
        # Display input parameters
        params_df = pd.DataFrame({
            'Parameter': ['Date', 'Hour', 'Day of Week', 'Is Weekend'],
            'Value': [
                f"{date_input.strftime('%Y-%m-%d')}",
                f"{hour_input} ({hour_input}:00)",
                f"{day_of_week} ({['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]})",
                f"{is_weekend} ({'Yes' if is_weekend else 'No'})"
            ]
        })
        
        st.table(params_df)
    
    with col2:
        st.markdown("<h2 class='sub-header'>Priority Tiers</h2>", unsafe_allow_html=True)
        
        # Display priority tiers
        st.markdown("<div class='tier1'>", unsafe_allow_html=True)
        st.markdown("**TIER 1 (Security)**")
        st.markdown(f"Security: {', '.join(TIER1['SECURITY'])}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='tier2'>", unsafe_allow_html=True)
        st.markdown("**TIER 2 (Healthcare)**")
        st.markdown(f"Healthcare: {', '.join(TIER2['HEALTHCARE'])}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='tier3'>", unsafe_allow_html=True)
        st.markdown("**TIER 3 (Financial)**")
        st.markdown(f"Financial: {', '.join(TIER3['FINANCIAL'])}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='tier4'>", unsafe_allow_html=True)
        st.markdown("**TIER 4 (General)**")
        st.markdown(f"General: {', '.join(TIER4['GENERAL'])}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Run forecast and allocation
    if st.button("Run Energy Allocation Simulation", key="run_simulation"):
        logger.info("Starting energy allocation simulation")
        
        with st.spinner("Running energy allocation simulation..."):
            try:
                # Prepare input features
                logger.info("Preparing input features")
                input_features = prepare_input_features(
                    hour_input, day_of_week, month, day, year, is_weekend
                )
                logger.debug(f"Input features shape: {input_features.shape}")
                
                # Generate input sequence
                logger.info("Generating input sequence")
                input_sequence = generate_input_sequence(feature_scaler, input_features)
                
                # Generate forecast
                with st.spinner("Forecasting energy demand..."):
                    logger.info("Running model prediction")
                    try:
                        forecast = model.predict(input_sequence)
                        logger.info(f"Forecast generated with shape: {forecast.shape}")
                        
                        # Convert forecast to original scale
                        n_outputs = len(FEEDER_COLS)
                        logger.debug(f"Inverse transforming forecast. n_outputs: {n_outputs}")
                        forecast_orig = target_scaler.inverse_transform(
                            forecast.reshape(-1, n_outputs)
                        ).reshape(forecast.shape)
                        logger.info(f"Inverse transformed forecast shape: {forecast_orig.shape}")
                        
                        # Calculate total energy demand for each feeder for the next day
                        daily_demand = {}
                        for i, feeder in enumerate(FEEDER_COLS):
                            # Sum across 24 hours to get total daily demand
                            daily_demand[feeder] = np.sum(forecast_orig[0, :, i])
                        
                        # Calculate total demand
                        total_demand = sum(daily_demand.values())
                        logger.info(f"Total forecasted demand: {total_demand:.2f} MW")
                        logger.debug(f"Demand breakdown: {daily_demand}")
                    except Exception as e:
                        logger.error(f"Error during forecasting: {str(e)}", exc_info=True)
                        st.error(f"Error during forecasting: {str(e)}")
                        return
                
                # Determine available supply based on variation or custom value
                if use_custom_supply:
                    available_supply = custom_supply
                    logger.info(f"Using custom supply: {available_supply:.2f} MW")
                else:
                    available_supply = total_demand * (1 + supply_variation/100)
                    logger.info(f"Calculated available supply: {available_supply:.2f} MW ({supply_variation:+d}% of demand)")
                
                # Run energy allocation
                with st.spinner("Allocating energy based on priorities..."):
                    logger.info("Running energy allocation algorithm")
                    try:
                        allocation, allocation_log = allocate_energy(daily_demand, available_supply)
                        logger.info(f"Energy allocation completed. Total allocated: {sum(allocation.values()):.2f} MW")
                        logger.debug(f"Allocation breakdown: {allocation}")
                        
                        analysis_text, tier_analysis, tx_analysis = analyze_allocation(allocation, daily_demand)
                        logger.info("Allocation analysis completed")
                    except Exception as e:
                        logger.error(f"Error during energy allocation: {str(e)}", exc_info=True)
                        st.error(f"Error during energy allocation: {str(e)}")
                        return
                
                # Display results
                logger.info("Displaying simulation results")
                st.success("Energy allocation completed successfully!")
                
                # Forecasted Demand
                st.markdown("<h2 class='sub-header'>Forecasted Energy Demand</h2>", unsafe_allow_html=True)
                
                # Create a DataFrame for the forecast
                forecast_df = pd.DataFrame({
                    'Feeder': list(daily_demand.keys()),
                    'Demand (MW)': list(daily_demand.values()),
                    'Share (%)': [d/total_demand*100 for d in daily_demand.values()]
                })
                forecast_df = forecast_df.sort_values('Demand (MW)', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Plot demand pie chart
                    fig = px.pie(
                        forecast_df, 
                        values='Demand (MW)', 
                        names='Feeder',
                        title='Forecasted Energy Demand Distribution'
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(forecast_df.style.format({
                        'Demand (MW)': '{:.2f}',
                        'Share (%)': '{:.1f}%'
                    }))
                    st.markdown(f"**Total Demand**: {total_demand:.2f} MW")
                    st.markdown(f"**Available Supply**: {available_supply:.2f} MW ({supply_variation:+d}%)")
                
                # Allocation Results
                st.markdown("<h2 class='sub-header'>Energy Allocation Results</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Create allocation chart
                    fig1, fig2 = create_allocation_chart(allocation, daily_demand)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Create percentage chart
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Tier allocation
                st.markdown("<h3>Tier-Based Allocation</h3>", unsafe_allow_html=True)
                tier_chart = create_tier_charts(allocation, daily_demand)
                st.plotly_chart(tier_chart, use_container_width=True)
                
                # Transformer loading
                st.markdown("<h3>Transformer Loading</h3>", unsafe_allow_html=True)
                tx_chart = create_transformer_chart(allocation)
                st.plotly_chart(tx_chart, use_container_width=True)
                
                # Allocation details
                st.markdown("<h3>Detailed Allocation Results</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Create allocation DataFrame
                    alloc_df = pd.DataFrame({
                        'Feeder': list(allocation.keys()),
                        'Demand (MW)': [daily_demand[f] for f in allocation.keys()],
                        'Allocated (MW)': list(allocation.values()),
                        'Percentage (%)': [allocation[f]/daily_demand[f]*100 if daily_demand[f] > 0 else 0 
                                          for f in allocation.keys()]
                    })
                    
                    # Add tier information
                    def get_tier(feeder):
                        if feeder in TIER1['SECURITY']:
                            return "TIER1 (Security)"
                        elif feeder in TIER2['HEALTHCARE']:
                            return "TIER2 (Healthcare)"
                        elif feeder in TIER3['FINANCIAL']:
                            return "TIER3 (Financial)"
                        else:
                            return "TIER4 (General)"
                    
                    alloc_df['Priority Tier'] = alloc_df['Feeder'].apply(get_tier)
                    
                    # Sort by priority order
                    tier_order = {f: i for i, f in enumerate(PRIORITY_ORDER)}
                    alloc_df['Priority'] = alloc_df['Feeder'].map(tier_order).fillna(999)
                    alloc_df = alloc_df.sort_values('Priority').drop('Priority', axis=1)
                    
                    # Display table
                    st.dataframe(alloc_df.style.format({
                        'Demand (MW)': '{:.2f}',
                        'Allocated (MW)': '{:.2f}',
                        'Percentage (%)': '{:.1f}%'
                    }))
                
                with col2:
                    st.text_area("Allocation Log", allocation_log, height=250)
                
                # Allocation analysis
                st.markdown("<h3>Allocation Analysis</h3>", unsafe_allow_html=True)
                st.text_area("Analysis", analysis_text, height=300)
                
                # Save simulation results
                st.markdown("<h3>Download Simulation Results</h3>", unsafe_allow_html=True)
                
                # Generate a timestamp for the filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                default_filename = f"energy_allocation_sim_{timestamp}.csv"
                
                # Create metadata as a string
                metadata_str = "# Energy Allocation Simulation Results\n"
                metadata_str += "# -----------------------------------\n"
                metadata_str += f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                metadata_str += "# Simulation Parameters:\n"
                metadata_str += f"# Date: {date_input.strftime('%Y-%m-%d')}\n"
                metadata_str += f"# Hour: {hour_input}\n"
                metadata_str += f"# Total Demand (MW): {total_demand:.2f}\n"
                metadata_str += f"# Available Supply (MW): {available_supply:.2f}\n"
                metadata_str += f"# Supply Variation (%): {supply_variation:+d}\n"
                metadata_str += "# -----------------------------------\n\n"
                
                # Convert DataFrame to CSV string
                csv_data = alloc_df.to_csv(index=False)
                
                # Combine metadata and CSV data
                download_data = metadata_str + csv_data
                
                # Create download button
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=download_data,
                        file_name=default_filename,
                        mime="text/csv",
                    )
                with col2:
                    st.info("Click the button to download the simulation results directly to your device.")

                logger.info("Download option displayed for simulation results")
                
            except Exception as e:
                logger.error(f"Unhandled error during simulation: {str(e)}", exc_info=True)
                st.error(f"An error occurred during the simulation: {str(e)}")
    else:
        logger.info("User has not started simulation yet")
        # Display instructions when not running a simulation
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("### Instructions")
        st.markdown(
            "1. Set the date and time parameters in the sidebar\n"
            "2. Adjust the supply variation or set a custom supply value\n"
            "3. Click 'Run Energy Allocation Simulation' to generate results\n"
            "4. Review the allocation and save the results if needed"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    logger.info("Application UI rendered successfully")

if __name__ == "__main__":
    try:
        logger.info("Application startup")
        main()
        logger.info("Application finished normally")
    except Exception as e:
        logger.critical(f"Unhandled exception in main application: {str(e)}", exc_info=True)
        st.error(f"Critical error: {str(e)}")