
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="NIFTY 50 Analytics Dashboard", layout="wide")

st.markdown("<h1 style='text-align:center;color:#1f77b4;'>📈 NIFTY 50 Analytics Dashboard</h1>", unsafe_allow_html=True)

nifty50 = [
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

ticker = st.sidebar.selectbox("Select Company", nifty50)

stock = yf.Ticker(ticker)
info = stock.info
hist = stock.history(period="5y")

current_price = info.get("currentPrice", "NA")
market_cap = info.get("marketCap", "NA")
pe_ratio = info.get("trailingPE", "NA")
book_value = info.get("bookValue", "NA")
dividend_yield = info.get("dividendYield", "NA")
sector = info.get("sector", "NA")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Current Price", f"₹{current_price}")
col2.metric("Market Cap", f"{market_cap:,}")
col3.metric("P/E Ratio", pe_ratio)
col4.metric("Book Value", book_value)

col5,col6 = st.columns(2)
col5.metric("Dividend Yield", dividend_yield)
col6.metric("Sector", sector)

st.subheader("5-Year Price Trend")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(hist.index, hist["Close"])
ax.set_ylabel("Price")
ax.set_xlabel("Date")
st.pyplot(fig)

st.subheader("ARIMA Forecast")

close = hist["Close"].dropna()
model = ARIMA(close, order=(5,1,0))
fit = model.fit()

forecast = fit.forecast(steps=12)

forecast_df = pd.DataFrame({
"Month": pd.date_range(close.index[-1], periods=13, freq="M")[1:],
"Forecast Price": forecast.values
})

st.dataframe(forecast_df)

st.subheader("Company Information")
st.write(info)
