import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("Stock Price Forecasting using ARIMA")

ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Run Forecast"):
    data = yf.download(ticker, period="5y")

    if data.empty:
        st.error("No data found for the ticker.")
    else:
        st.subheader("Last 5 Years Stock Price Data")
        st.dataframe(data.tail())

        # Line chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data["Close"], label="Closing Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title(f"{ticker} Closing Prices - Last 5 Years")
        ax.legend()
        st.pyplot(fig)

        # ARIMA Model
        close_prices = data["Close"].dropna()
        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast until June 2027
        last_date = close_prices.index[-1]
        target_date = pd.Timestamp("2027-06-30")

        months = max(1, (target_date.year - last_date.year) * 12 +
                     (target_date.month - last_date.month))

        forecast = model_fit.forecast(steps=months)
        forecast_dates = pd.date_range(start=last_date, periods=months + 1, freq="M")[1:]

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecast Price": forecast.values
        })

        st.subheader("Forecasted Prices")
        st.dataframe(forecast_df)

        june_2027_price = forecast_df.iloc[-1]["Forecast Price"]
        st.success(f"Predicted Price for June 2027: {june_2027_price:.2f}")
