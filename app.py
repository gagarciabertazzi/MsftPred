import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler as mini
from sklearn.model_selection import train_test_split
#from streamlit_lottie import st_lottie
import streamlit as st
import requests

#def load_lottieurl(url: str):
#    r = requests.get(url)
#    if r.status_code != 200:
#        return None
#    return r.json()

#lottie_stock = load_lottieurl('https://assets4.lottiefiles.com/packages/lf20_jhu1lqdz.json')
#st_lottie(lottie_stock, speed=1, height=200, key="initial")
st.title('Microsoft Stock Predictor')

st.write("Stock Predictor for X Days:")

st.sidebar.title('Stock')
steps = st.sidebar.number_input(label='Day:', min_value=1, max_value=35, step=1)


model = joblib.load('msft.pkl')


y = model.predict(steps=steps)

print(y)
