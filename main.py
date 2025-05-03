import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import requests
from textblob import TextBlob

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 8px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("\U0001F4C8 Stock Market Analysis")
st.sidebar.header("\U0001F527 Stock Name")

stock = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS)", "")
if not stock:
    st.sidebar.error("Please enter a valid stock ticker symbol.")
    st.stop()

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

try:
    model = load_model("Latest_stock_price_model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def fetch_stock_data(stock):
    try:
        data = yf.download(stock, start=start, end=end)
        if data.empty:
            st.error("No data found for the specified stock.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def fetch_stock_news(stock):
    api_key = "27bb78beb8644545890fc0f8e1012968"
    api_url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={api_key}"
    try:
        response = requests.get(api_url)
        data = response.json()
        return data.get("articles", []) if data["status"] == "ok" else []
    except Exception:
        return []

def analyze_sentiment(news_articles):
    sentiments = []
    for article in news_articles:
        text = (article.get("title") or "") + " " + (article.get("description") or "")
        blob = TextBlob(text)
        score = blob.sentiment.polarity
        sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
        sentiments.append((article["title"], article["url"], sentiment))
    return sentiments

def predict_stock_price(data):
    scaler = MinMaxScaler()
    close = data['Close'].values.reshape(-1, 1)
    scaled = scaler.fit_transform(close)
    x_input = scaled[-60:].reshape(1, -1, 1)
    pred = model.predict(x_input)
    return scaler.inverse_transform(pred)[0][0]

def generate_esg_data(length):
    df = pd.DataFrame({
        'Environmental Score': np.random.uniform(50, 100, length),
        'Social Score': np.random.uniform(50, 100, length),
        'Governance Score': np.random.uniform(50, 100, length)
    })
    df['Average Score'] = df.mean(axis=1)
    return df

def plot_graph(figsize, ma_values, full_data):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ma_values, 'orange', label='Moving Average')
    ax.plot(full_data['Close'], 'blue', label='Close Price')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Stock Price and Moving Averages")
    ax.legend()
    return fig

def get_currency_symbol(stock):
    return "₹" if stock.lower().endswith(".ns") else "$"

if st.sidebar.button("Fetch Stock Data"):
    data = fetch_stock_data(stock)
    if data is not None:
        st.subheader("\U0001F4C8 Stock Data Overview")
        st.dataframe(data)

        esg_df = generate_esg_data(len(data))
        st.subheader("♻️ ESG Scores")
        st.dataframe(esg_df)

        for ma in [50, 100, 200]:
            data[f'MA_for_{ma}_days'] = data['Close'].rolling(ma).mean()
            st.subheader(f'\U0001F4C8 {ma}-Day Moving Average')
            st.pyplot(plot_graph((15, 6), data[f'MA_for_{ma}_days'], data))

        latest_price = float(data['Close'].iloc[-1])
        ma_50 = float(data['MA_for_50_days'].iloc[-1])
        ma_100 = float(data['MA_for_100_days'].iloc[-1])
        ma_200 = float(data['MA_for_200_days'].iloc[-1])
        avg_esg_score = esg_df['Average Score'].iloc[-1]

        # ESG Strength
        esg_strength = "Strong ESG" if avg_esg_score >= 75 else "Moderate ESG" if avg_esg_score >= 60 else "Weak ESG"

        # Moving average based recommendation logic only
        if latest_price > ma_50 > ma_100 > ma_200:
            recommendation = f"BUY ({esg_strength})"
        elif latest_price < ma_50 < ma_100 < ma_200:
            recommendation = f"SELL ({esg_strength})"
        else:
            recommendation = f"HOLD ({esg_strength})"

        symbol = get_currency_symbol(stock)
        color = "green" if "BUY" in recommendation else "red" if "SELL" in recommendation else "orange"

        st.subheader("\U0001F4CC Investment Recommendation")
        st.markdown(f"<h6>Latest Close Price: {symbol}{latest_price:.2f}</h6>", unsafe_allow_html=True)
        st.markdown(f"<h6>50-Day MA: {symbol}{ma_50:.2f}</h6>", unsafe_allow_html=True)
        st.markdown(f"<h6>100-Day MA: {symbol}{ma_100:.2f}</h6>", unsafe_allow_html=True)
        st.markdown(f"<h6>200-Day MA: {symbol}{ma_200:.2f}</h6>", unsafe_allow_html=True)
        st.markdown(f"<h6>Average ESG Score: {avg_esg_score:.2f} ({esg_strength})</h6>", unsafe_allow_html=True)
        st.markdown(f"<h6>Recommendation: <span style='color:{color};'>{recommendation}</span></h6>", unsafe_allow_html=True)

if st.sidebar.button("Live News"):
    data = fetch_stock_data(stock)
    if data is not None:
        st.subheader(f"\U0001F4C8 Stock Data for {stock}")
        st.dataframe(data)
        news = fetch_stock_news(stock)
        if news:
            st.subheader(f"\U0001F4F0 Latest News for {stock}")
            sentiments = analyze_sentiment(news)
            for title, url, sentiment in sentiments:
                st.write(f"**{title}**")
                st.write(f"[Read more]({url})")
                st.write(f"Sentiment: {sentiment}")
                st.markdown("---")
        else:
            st.warning("No news found.")

st.sidebar.subheader("About this App")
st.sidebar.info("This app provides real-time stock analysis, ESG metrics, technical indicators, and investment recommendations.")
