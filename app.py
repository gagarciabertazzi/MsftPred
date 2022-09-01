import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler as mini
from sklearn.model_selection import train_test_split
import streamlit as st
import requests
import mpld3
import streamlit.components.v1 as components

st.title('PREDICCIONES DE LOS STOCK DE MICROSOFT')

st.write("DATA DEL STOCK DE LOS ULTIMOS 5 ANOS:")

data = pd.read_csv("MSFT.csv", sep=',')
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data = data.set_index('Date')
data = data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis = 1)
data.dropna(subset=['Close'], inplace=True)
chart_data = pd.DataFrame(data)
st.line_chart(chart_data)

model = joblib.load('msft.pkl')

st.write("PREDICCIONES DE LOS PROXIMOS 35 DIAS:")

steps = 36
predictions = model.predict(steps=steps)
predictions = pd.DataFrame(predictions)
predictions = predictions.reset_index()
predictions = predictions.drop(['index'], axis = 1)
chart_data_pred = pd.DataFrame(predictions)
st.line_chart(chart_data_pred)

df = pd.DataFrame(predictions)
st.table(df)

