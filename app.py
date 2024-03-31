import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Title of the app
st.title('Stock Trend Prediction')

# Dates
start='2019-09-17'
end='2024-03-31'

# Get the Stock
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Scrape Stock Data
data = yf.download(user_input, start, end)

# Describe Data
st.subheader('Data from 2019-2024')
st.write(data.describe())

# Visualizations

# Closing Price
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)

# Moving Avg 100 Days
st.subheader('100 Days Moving Average')
ma100 = data.Close.rolling(100).mean()
fig1 = plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(ma100,'r')
st.pyplot(fig1)

# Moving Avg 200 Days
st.subheader('200 Days Moving Average')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig2)

# Train Test Split
price = data['Close']
training, testing = train_test_split(price, test_size=0.3, shuffle=False)

# Scaling data
scaler = MinMaxScaler(feature_range=(0,1))
training_trans = scaler.fit_transform(training.values.reshape(-1, 1))


# Load Pre-Trained Model
model = load_model('kerasmodel.h5')

# Testing
past100days = testing.tail(100)
final_df = pd.concat([past100days, testing], ignore_index=True)
input_data = scaler.fit_transform(final_df.values.reshape(-1, 1))

# X,Y Split for Testing
x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_predicted = model.predict(x_test)
scale = scaler.scale_
scale_factor = 1/scale[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Original vs Predicted
st.subheader('Original vs Predicted')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)