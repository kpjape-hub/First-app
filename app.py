
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NIFTY 50 Stock Forecast",
    page_icon="📈",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}
h1, h2, h3 {
    color: #FFFFFF;
}
.metric-container {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- NIFTY 50 STOCKS ----------------
nifty50_tickers = {
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "BEL": "BEL.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Britannia": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Eternal": "ETERNAL.NS",
    "Grasim": "GRASIM.NS",
    "HCL Tech": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Infosys": "INFY.NS",
    "ITC": "ITC.NS",
    "Jio Financial": "JIOFIN.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS",
    "Reliance Industries": "RELIANCE.NS",
    "SBI": "SBIN.NS",
    "SBI Life": "SBILIFE.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "Trent": "TRENT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS"
}

# ---------------- TITLE ----------------
st.title("📈 NIFTY 50 Stock Forecasting Dashboard")
st.markdown("### Interactive Historical Analysis + ARIMA Forecasting")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Controls")

selected_stock = st.sidebar.selectbox(
    "Select NIFTY 50 Company",
    sorted(nifty50_tickers.keys())
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=30,
    max_value=730,
    value=365
)

ticker = nifty50_tickers[selected_stock]

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(ticker):
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(
        ticker,
        start="2021-01-01",
        end=end_date,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data

data = load_data(ticker)

# ---------------- ERROR HANDLING ----------------
if data.empty:
    st.error("No stock data available.")
    st.stop()

# ---------------- METRICS ----------------
latest_close = float(data["Close"].iloc[-1])
highest_price = float(data["High"].max())
lowest_price = float(data["Low"].min())

col1, col2, col3 = st.columns(3)

col1.metric("📌 Latest Close", f"₹{latest_close:.2f}")
col2.metric("📈 5Y High", f"₹{highest_price:.2f}")
col3.metric("📉 5Y Low", f"₹{lowest_price:.2f}")

# ---------------- HISTORICAL DATA ----------------
st.subheader(f"📜 Historical Data - {selected_stock}")
st.dataframe(data.tail(15), use_container_width=True)

# ---------------- INTERACTIVE CHART ----------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index,
    y=data["Close"],
    mode="lines",
    name="Close Price"
))

fig.update_layout(
    title=f"{selected_stock} Closing Prices",
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Price (INR)",
    hovermode="x unified",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- ARIMA FORECAST ----------------
st.subheader("🤖 ARIMA Forecasting")

close_prices = data["Close"].dropna()

try:
    model = ARIMA(close_prices, order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_days)

    future_dates = pd.date_range(
        start=data.index[-1],
        periods=forecast_days + 1,
        freq="D"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast Price": forecast.values
    })

    # ---------------- FORECAST CHART ----------------
    forecast_fig = go.Figure()

    forecast_fig.add_trace(go.Scatter(
        x=close_prices.index,
        y=close_prices,
        mode="lines",
        name="Historical"
    ))

    forecast_fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast Price"],
        mode="lines",
        name="Forecast"
    ))

    forecast_fig.update_layout(
        title=f"{selected_stock} Forecast",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        height=600
    )

    st.plotly_chart(forecast_fig, use_container_width=True)

    # ---------------- JUNE 2027 PREDICTION ----------------
    june_2027 = forecast_df[
        (forecast_df["Date"].dt.year == 2027) &
        (forecast_df["Date"].dt.month == 6)
    ]

    if not june_2027.empty:
        june_price = float(june_2027.iloc[-1]["Forecast Price"])

        st.success(
            f"🎯 Predicted Price for June 2027: ₹{june_price:.2f}"
        )

    # ---------------- FORECAST TABLE ----------------
    st.subheader("📅 Forecasted Prices")
    st.dataframe(forecast_df.tail(30), use_container_width=True)

except Exception as e:
    st.error(f"Forecasting Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit, Yahoo Finance, Plotly & ARIMA"
)
