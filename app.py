import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📈 Stock Price Predictor")
st.markdown("Predict stock prices using Facebook's Prophet forecasting model")

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Stock selection
popular_stocks = {
    'Apple': 'AAPL',
    'Microsoft': 'MSFT',
    'Google': 'GOOGL',
    'Amazon': 'AMZN',
    'Tesla': 'TSLA',
    'Meta': 'META',
    'Netflix': 'NFLX',
    'NVIDIA': 'NVDA',
    'AMD': 'AMD',
    'Intel': 'INTC'
}

# Stock selection dropdown
selected_stock_name = st.sidebar.selectbox(
    "Select a Stock",
    options=list(popular_stocks.keys()),
    index=0
)

# Custom ticker input
custom_ticker = st.sidebar.text_input(
    "Or enter custom ticker symbol",
    placeholder="e.g., MSFT, GOOGL"
)

# Use custom ticker if provided, otherwise use selected stock
if custom_ticker:
    ticker = custom_ticker.upper()
    stock_name = custom_ticker.upper()
else:
    ticker = popular_stocks[selected_stock_name]
    stock_name = selected_stock_name

# Date range selection
st.sidebar.subheader("Historical Data Range")
start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime.now() - timedelta(days=365*2),
    max_value=datetime.now()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.now(),
    max_value=datetime.now()
)

# Forecast period
st.sidebar.subheader("Forecast Settings")
forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=30,
    max_value=365,
    value=90,
    step=30
)

# Advanced settings
st.sidebar.subheader("Advanced Settings")
show_components = st.sidebar.checkbox("Show Forecast Components", value=False)
show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)

# Function to load data
@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None, "No data found for the given ticker and date range."
        return data, None
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

# Function to prepare data for Prophet
def prepare_prophet_data(data):
    df = pd.DataFrame(data)
    df = df[['Close']].copy()
    df.reset_index(inplace=True)
    df.columns = ['ds', 'y']
    return df

# Function to create forecast
@st.cache_data
def create_forecast(df, forecast_days):
    try:
        # Initialize Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Fit the model
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_days)
        
        # Make predictions
        forecast = model.predict(future)
        
        return model, forecast, None
    except Exception as e:
        return None, None, f"Error creating forecast: {str(e)}"

# Function to calculate metrics
def calculate_metrics(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# Main application logic
if st.sidebar.button("Generate Forecast", type="primary"):
    # Validate date range
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()
    
    # Load data
    with st.spinner(f"Loading data for {stock_name} ({ticker})..."):
        data, error = load_data(ticker, start_date, end_date)
    
    if error:
        st.error(error)
        st.stop()
    
    # Display stock info
    try:
        stock_info = yf.Ticker(ticker).info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Company", stock_info.get('shortName', ticker))
        with col2:
            st.metric("Sector", stock_info.get('sector', 'N/A'))
        with col3:
            st.metric("Market Cap", f"${stock_info.get('marketCap', 0):,.0f}" if stock_info.get('marketCap') else 'N/A')
        with col4:
            current_price = data['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
    except:
        st.info(f"Showing data for {ticker}")
    
    # Prepare data for Prophet
    df = prepare_prophet_data(data)
    
    # Create forecast
    with st.spinner("Creating forecast..."):
        model, forecast, error = create_forecast(df, forecast_days)
    
    if error:
        st.error(error)
        st.stop()
    
    # Display results
    st.success(f"Forecast generated successfully for {forecast_days} days!")
    
    # Create main forecast plot
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Historical Data',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast data
    future_dates = forecast['ds'][len(df):]
    forecast_values = forecast['yhat'][len(df):]
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(255,0,0,0.2)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        line=dict(color='rgba(255,0,0,0.2)'),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'{stock_name} ({ticker}) - Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display forecast summary
    st.subheader("Forecast Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        last_price = df['y'].iloc[-1]
        st.metric("Last Historical Price", f"${last_price:.2f}")
    
    with col2:
        forecast_price = forecast['yhat'].iloc[-1]
        st.metric("Forecasted Price", f"${forecast_price:.2f}")
    
    with col3:
        price_change = forecast_price - last_price
        change_percent = (price_change / last_price) * 100
        st.metric(
            "Predicted Change",
            f"${price_change:.2f}",
            f"{change_percent:.1f}%"
        )
    
    # Show model performance metrics
    st.subheader("Model Performance")
    
    # Calculate metrics on historical data
    historical_forecast = forecast[forecast['ds'] <= df['ds'].max()]
    historical_actual = df['y'].values
    historical_predicted = historical_forecast['yhat'].values[:len(historical_actual)]
    
    metrics = calculate_metrics(historical_actual, historical_predicted)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{metrics['RMSE']:.2f}")
    with col2:
        st.metric("MAE", f"{metrics['MAE']:.2f}")
    with col3:
        st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    with col4:
        # Calculate R-squared
        ss_res = np.sum((historical_actual - historical_predicted) ** 2)
        ss_tot = np.sum((historical_actual - np.mean(historical_actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        st.metric("R²", f"{r_squared:.3f}")
    
    # Show forecast components
    if show_components:
        st.subheader("Forecast Components")
        
        # Create components plot
        fig_components = go.Figure()
        
        # Trend
        fig_components.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='blue')
        ))
        
        if 'weekly' in forecast.columns:
            fig_components.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['weekly'],
                mode='lines',
                name='Weekly Seasonality',
                line=dict(color='green')
            ))
        
        if 'yearly' in forecast.columns:
            fig_components.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yearly'],
                mode='lines',
                name='Yearly Seasonality',
                line=dict(color='orange')
            ))
        
        fig_components.update_layout(
            title='Forecast Components',
            xaxis_title='Date',
            yaxis_title='Price Impact ($)',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_components, use_container_width=True)
    
    # Show raw data
    if show_raw_data:
        st.subheader("Raw Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Historical Data")
            st.dataframe(df.tail(10))
        
        with col2:
            st.write("Forecast Data")
            forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
            forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
            st.dataframe(forecast_display)
    
    # Download forecast data
    st.subheader("Download Forecast")
    
    # Prepare download data
    download_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    download_data.columns = ['Date', 'Forecast', 'Lower_Bound', 'Upper_Bound']
    download_data['Date'] = download_data['Date'].dt.strftime('%Y-%m-%d')
    
    csv = download_data.to_csv(index=False)
    st.download_button(
        label="Download Forecast as CSV",
        data=csv,
        file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    # Default display
    st.info("Configure your settings in the sidebar and click 'Generate Forecast' to begin.")
    
    # Show example
    st.subheader("How it works:")
    st.markdown("""
    1. **Select a stock** from the dropdown or enter a custom ticker symbol
    2. **Choose the date range** for historical data (more data = better predictions)
    3. **Set forecast period** (30-365 days)
    4. **Click 'Generate Forecast'** to see the prediction
    
    The model uses Facebook's Prophet algorithm to analyze historical patterns and predict future prices.
    """)
    
    # Show sample visualization
    st.subheader("Sample Forecast Visualization")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    fig_sample = go.Figure()
    fig_sample.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Sample Historical Data',
        line=dict(color='blue')
    ))
    
    # Add sample forecast
    future_dates = pd.date_range(start='2024-01-02', end='2024-03-01', freq='D')
    future_prices = prices[-1] + np.cumsum(np.random.randn(len(future_dates)) * 0.3)
    
    fig_sample.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        mode='lines',
        name='Sample Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig_sample.update_layout(
        title='Sample Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_sample, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** This is for educational purposes only. Not financial advice. Always consult with a financial advisor before making investment decisions.")