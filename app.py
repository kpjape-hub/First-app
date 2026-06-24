import streamlit as st
import yfinance as yf
import pandas as pd
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
    list(nifty50_tickers.keys())
)

ticker = nifty50_tickers[selected_stock]

# Date range
start_date = "2021-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Download Data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)

    # FIX FOR MULTI-INDEX ISSUE
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

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

# ARIMA Model
model = ARIMA(close_data, order=(5, 1, 0))
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
    'Forecast': forecast.values
})

# June 2027 Prediction
june_2027 = forecast_df[
    (forecast_df['Date'].dt.year == 2027) &
    (forecast_df['Date'].dt.month == 6)
]

if not june_2027.empty:
    predicted_price = june_2027.iloc[-1]['Forecast']

    st.success(
        f"Predicted Price for June 2027: ₹{predicted_price:.2f}"
    )
else:
    st.warning("June 2027 forecast not available.")

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
