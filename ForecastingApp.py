import streamlit as st
import pandas as pd
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import warnings
import numpy as np
from numpy import sqrt
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')




def main():
    st.sidebar.header('Times series Forcasting App')
    st.sidebar.info('Time Series prediction.Created and designed by Cristian. Make sure your data is a time series data with just two                          columns including date')
    st.sidebar.text('Only two columns including date')
    option = st.sidebar.selectbox('How do you want to get the data?', ['url', 'file'])
    if option == 'url':
        url = st.sidebar.text_input('Enter a url')
        if url:
            dataframe(url)
    else:
        file = st.sidebar.file_uploader('Choose a file', type=['csv'])
        if file:
            dataframe(file)

                    
def dataframe(df):
    st.header('Forecasting App')
    data = read_csv(df, header=0, parse_dates=True, index_col=0)
    to_do = st.radio('Select the action that you want to perform with your data', ['Visualize', 'Check for stationary', 'Forecast'])
    if to_do == 'Visualize':
        data_visualization(data)
    elif to_do == 'Check for stationary':
        stationary_test(data)
    else:
        forecast_data(data)



def data_visualization(data):
    button = st.button('Draw')
    if button:
        st.line_chart(data)

        
def stationary_test(data):
    res = testing(data)
    st.text(f'Augmented Dickey_fuller Statistical Test: {res[0]} \
           \np-values: {res[1]}')
    st.text('Critical values at different levels:')
    for k, v in res[4].items():
        st.text(f'{k}:{v}')
    if res[1] > 0.05:
        st.text('Your dataset is non-stationary and is being transformed \
                \ninto a stationary time series data while performing auto-arima')
    elif res[1] <= 0.05:
        st.text('Your dataset is stationary and is ready for training')

 
def testing(df):
    return adfuller(df)


def forecast_data(df):
    st.text('...searching for the optimum parameter')
    optimum_para(df)
    st.text('Enter the best parameter founded with Auto-arima')
    p = st.number_input('The p term')
    q = st.number_input('The q term')
    d = st.number_input('The d term')
    period = st.number_input('Enter the next period(s) you want to forecast', value=3)
    button = st.button('Forecast')
    if button:
        model_forecast(df, p, q, d, period)


def model_forecast(data, p, q, d, period):
    size = int(len(data) * .9)
    train, test = data[:size], data[size:]
    model = ARIMA(train.values, order=(p,q,d))
    model_fit = model.fit()
    output = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
    error = sqrt(mean_squared_error(output, test))
    st.text(f'MSE using {p,q,d}: {error}')
    st.text(f'Forecasting {period} future values')
    model_2 = ARIMA(data.values, order =(p,q,d)).fit()
    forecast = model_2.predict(start=len(data), end=len(data)+period, typ='levels')
    month = 1
    for i in forecast:
        st.text(f'Period {month}: {i}')
        month += 1



def optimum_para(df):
    size = int(len(df) * 0.9)
    train, test = df[:size], df[size:]
    
    model = auto_arima(train, trace=True, error_action='trace', suppress_warnings=True, n_jobs=-1, 
                   stationary=False, test='adf', max_d=5)
    model.fit(train)
    forecast = model.predict(start=len(train), end=len(train) + len(test)-1)
    st.text(f'Auto-arima best model: {model.fit(train)}')

    

        
if __name__ == '__main__':
    main()