import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt

from datetime import date
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# initializing dates in yyyy-mm-dd format
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

scaler = MinMaxScaler(feature_range=(0,1))

st.title('Stock Trend Prediction')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select the dataset for prediction', stocks)
df = pdr.DataReader(selected_stock, 'yahoo', START, TODAY)

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading Data...')
data = load_data(selected_stock)
data_load_state.text(f'Data from 2015-01-01 to {date.today().strftime("%Y-%m-%d")}')

st.subheader('Raw Data')
st.write(df.describe())

# Plot raw data
def plot_raw_data():
	fig = plt.figure(figsize=(12,6))
	plt.plot(df.Close, 'b')
	st.pyplot(fig)
	
plot_raw_data()

# data training
data_training = pd.DataFrame(df['Close'][0: int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

data_training_array = scaler.fit_transform(data_training)

# splittin data into x_train and y_train
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
	x_train.append(data_training_array[i-100: i])
	y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# loading models
main_model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
	x_test.append(input_data[i-100: i])
	y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = main_model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

def plot_predicted_data():
	fig1 = plt.figure(figsize=(12,6))
	plt.plot(y_test, 'b', label = 'Original Price')
	plt.plot(y_predicted, 'r', label = 'Predicted Price')
	plt.xlabel('Time')
	plt.ylabel('Price')
	plt.legend()
	st.pyplot(fig1)
	
plot_predicted_data()