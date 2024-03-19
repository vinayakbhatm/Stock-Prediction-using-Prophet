import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2018-01-01"

st.title('Stock Forecast App')

# Define market closing times
market_closing_times = {
    'INDIA': datetime.now().replace(hour=15, minute=30, second=0, microsecond=0),  # Indian market
    'US': datetime.now().replace(hour=16, minute=0, second=0, microsecond=0),  # US market (NYSE)
}

# Add an input box for the dataset
company_symbol = st.text_input('Enter the company symbol ')
selected_market = st.selectbox('Select market to load data', ['INDIA', 'US'])
n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365
# Function to get the market closing time for a specific market
def get_market_closing_time(market):
    current_time = datetime.now()
    closing_time = market_closing_times.get(market)
    if closing_time and current_time >= closing_time:
        closing_time += timedelta(days=1)  # If the current time is after the closing time, fetch data for the next day
    return closing_time

# Calculate the market closing times
market_closing_time = get_market_closing_time(selected_market)

# Function to load data with the specified end date and intraday data
# @st.cache_resource
def load_data(symbol, market):
    if market == 'INDIA':
        data = yf.download(symbol, start=START, end=market_closing_time, interval="1d")
    elif market == 'US':
        data = yf.download(symbol, start=START, end=market_closing_time, interval="1d")

    data.reset_index(inplace=True)

    # Extract only the date from the datetime in the "Date" column
    data['Date'] = data['Date'].dt.date

    return data

if company_symbol:
    data_load_state = st.text('Loading data...')
    data = load_data(company_symbol, selected_market)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data ', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)  # Default to one year forecast
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write(f'Forecast plot for {n_years} year')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)