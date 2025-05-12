import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("📈 Real-Time Stock Price Dashboard with Prediction")

# Sidebar
ticker = st.sidebar.text_input("Enter Stock Ticker", value='AAPL')
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2024-12-31'))

# Fetch stock data
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

df = load_data(ticker, start_date, end_date)


# Check if data is loaded
if df.empty:
    st.error("Failed to load data. Check the stock ticker or internet connection.")
    st.stop()
st.dataframe(df.tail(), use_container_width=True)
st.subheader(f"{ticker} Closing Price")
st.line_chart(df['Close'])

# Preprocessing for LSTM model
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

def create_dataset(dataset, time_step=60):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

x_train, y_train = create_dataset(train_data)
x_test, y_test = create_dataset(test_data)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

# Predict
predicted = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(y=actual_prices.flatten(), name="Actual"))
fig.add_trace(go.Scatter(y=predicted_prices.flatten(), name="Predicted"))
fig.update_layout(title="📉 Actual vs Predicted Closing Price", xaxis_title="Time", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)
