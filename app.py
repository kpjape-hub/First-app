import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from nifty50_tickers import nifty50_tickers

st.set_page_config(page_title="NIFTY 50 Stock Forecast", layout="wide")

st.title("📈 NIFTY 50 Stock Price Forecasting App")
st.markdown("### Historical Data + ARIMA Forecast till June 2027")

# Sidebar
selected_stock = st.sidebar.selectbox(
    "Select NIFTY 50 Stock",
    nifty50_tickers.keys()
)

ticker = nifty50_tickers[selected_stock]

# Date range
start_date = "2021-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Download Data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

data = load_data(ticker)

st.subheader(f"{selected_stock} Historical Data")

st.dataframe(data.tail())

# Plot Historical Data
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='Close Price'
))

fig.update_layout(
    title=f"{selected_stock} Closing Price",
    xaxis_title="Date",
    yaxis_title="Price (INR)",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ARIMA Forecasting
st.subheader("📊 ARIMA Forecast")

close_data = data['Close'].dropna()

# Train Model
model = ARIMA(close_data, order=(5,1,0))
model_fit = model.fit()

# Forecast till June 2027
forecast_steps = 365

forecast = model_fit.forecast(steps=forecast_steps)

future_dates = pd.date_range(
    start=data.index[-1],
    periods=forecast_steps + 1,
    freq='D'
)[1:]

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast': forecast
})

# June 2027 Price
june_2027 = forecast_df[
    (forecast_df['Date'].dt.year == 2027) &
    (forecast_df['Date'].dt.month == 6)
]

predicted_price = june_2027.iloc[-1]['Forecast']

st.success(
    f"Predicted Price for June 2027: ₹{predicted_price:.2f}"
)

# Forecast Plot
forecast_fig = go.Figure()

forecast_fig.add_trace(go.Scatter(
    x=close_data.index,
    y=close_data,
    mode='lines',
    name='Historical'
))

forecast_fig.add_trace(go.Scatter(
    x=forecast_df['Date'],
    y=forecast_df['Forecast'],
    mode='lines',
    name='Forecast'
))

forecast_fig.update_layout(
    title=f"{selected_stock} Forecast till June 2027",
    xaxis_title="Date",
    yaxis_title="Price (INR)",
    template="plotly_dark"
)

st.plotly_chart(forecast_fig, use_container_width=True)

st.subheader("Forecast Data")
st.dataframe(forecast_df.tail(30))

nifty50_tickers = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Titan": "TITAN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "HCL Tech": "HCLTECH.NS",
    "ONGC": "ONGC.NS",
    "Coal India": "COALINDIA.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Grasim": "GRASIM.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Britannia": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Divis Labs": "DIVISLAB.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "BPCL": "BPCL.NS",
    "SBI Life": "SBILIFE.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "UPL": "UPL.NS",
    "Hindalco": "HINDALCO.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "LTIMindtree": "LTIM.NS",
    "Bharat Electronics": "BEL.NS"
}
