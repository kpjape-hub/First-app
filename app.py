
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("NIFTY 50 Stock Forecasting using ARIMA")

nifty50_tickers = [
"ADANIPORTS.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS",
"BAJAJFINSV.NS","BEL.NS","BHARTIARTL.NS","CIPLA.NS","COALINDIA.NS",
"DRREDDY.NS","EICHERMOT.NS","ETERNAL.NS","GRASIM.NS","HCLTECH.NS",
"HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS",
"ICICIBANK.NS","INDUSINDBK.NS","INFY.NS","ITC.NS","JIOFIN.NS",
"JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS",
"NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS","RELIANCE.NS",
"SBILIFE.NS","SBIN.NS","SHRIRAMFIN.NS","SUNPHARMA.NS","TATACONSUM.NS",
"TATAMOTORS.NS","TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS",
"TRENT.NS","ULTRACEMCO.NS","WIPRO.NS","APOLLOHOSP.NS","BRITANNIA.NS"
]

ticker = st.selectbox("Select a NIFTY 50 Company", nifty50_tickers)

if st.button("Forecast"):
    data = yf.download(ticker, period="5y", auto_adjust=True)

    close = data["Close"].dropna()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(close.index, close.values)
    st.pyplot(fig)

    model = ARIMA(close, order=(5,1,0))
    fitted = model.fit()

    forecast = fitted.forecast(steps=12)

    forecast_df = pd.DataFrame({
        "Month": pd.date_range(close.index[-1], periods=13, freq="M")[1:],
        "Forecast Price": forecast.values
    })

    st.dataframe(forecast_df)
