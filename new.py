# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import plotly.express as px
import yfinance as yf

import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np



from plotly import graph_objs as go
from prophet.plot import plot_plotly, plot_components_plotly
# def set_bg_image():
#     """
#     A function to unpack an image from root folder and set as bg.
#     Returns
#     -------
#     The background.
#     """
    
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://media.istockphoto.com/id/1349640220/photo/businessman-hand-pointing-finger-to-growth-success-finance-business-chart-of-metaverse.jpg");
#              background-size: cover;
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# set_bg_image()
import prophet as pt   
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('nifty 50 Stock Forecast Dashboard')

stocks =  (
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BHARTIARTL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "ADANIPORTS.NS",
    "HCLTECH.NS",
    "HDFC.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "IOC.NS",
    "ITC.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHREECEM.NS",
    "SUNPHARMA.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "UPL.NS",
    "WIPRO.NS"
)

selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
m = pt.Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
