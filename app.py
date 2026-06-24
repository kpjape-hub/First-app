
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="NIFTY 50 Dashboard", layout="wide")

NIFTY50 = [
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

@st.cache_data(ttl=3600)
def load_data(symbol):
    return yf.download(symbol, period="5y", progress=False, auto_adjust=True)

st.title("📈 NIFTY 50 Dashboard")

ticker = st.sidebar.selectbox("Select Stock", NIFTY50)

try:
    df = load_data(ticker)

    if df.empty:
        st.error("No data available.")
        st.stop()

    current_price = round(float(df["Close"].iloc[-1]),2)
    high_52w = round(float(df["High"].max()),2)
    low_52w = round(float(df["Low"].min()),2)

    c1,c2,c3 = st.columns(3)
    c1.metric("Current Price", f"₹{current_price}")
    c2.metric("5Y High", f"₹{high_52w}")
    c3.metric("5Y Low", f"₹{low_52w}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=df["Close"],name="Close Price"))
    fig.update_layout(height=500)
    st.plotly_chart(fig,use_container_width=True)

    close = df["Close"].dropna()

    model = ARIMA(close,order=(5,1,0))
    fitted = model.fit()

    forecast = fitted.forecast(steps=12)

    forecast_df = pd.DataFrame({
        "Month": pd.date_range(start=close.index[-1], periods=13, freq="ME")[1:],
        "Forecast Price": forecast.values
    })

    st.subheader("12-Month Forecast")
    st.dataframe(forecast_df)

except Exception as e:
    st.error(f"Error: {e}")
