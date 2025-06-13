import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Parsing portfolio data
data = {
    'Stock': ['AADI', 'ADRO', 'ANTM', 'BFIN', 'BJBR', 'BSSR', 'LPPF', 'PGAS', 'PTBA', 'UNVR', 'WIIM'],
    'Lot Balance': [5.0, 17.0, 15.0, 30.0, 23.0, 11.0, 5.0, 10.0, 4.0, 60.0, 5.0],
    'Balance': [500, 1700, 1500, 3000, 2300, 1100, 500, 1000, 400, 6000, 500],
    'Avg Price': [7300, 2605, 1423, 1080, 1145, 4489, 1700, 1600, 2400, 1860, 871],
    'Stock Value': [3650000, 4428500, 2135000, 3240000, 2633500, 4938000, 850000, 1600000, 960000, 11162500, 435714],
    'Market Price': [7225, 2200, 3110, 905, 850, 4400, 1745, 1820, 2890, 1730, 835],
    'Market Value': [3612500, 3740000, 4665000, 2715000, 1955000, 4840000, 872500, 1820000, 1156000, 10380000, 417500],
    'Unrealized': [-37500, -688500, 2530000, -525000, -678500, -98000, 22500, 220000, 196000, -782500, -18215]
}
df = pd.DataFrame(data)

# Simulated historical data for price prediction (replace with real data if available)
np.random.seed(42)
dates = pd.date_range(end='2025-05-31', periods=100, freq='D')
simulated_data = {}
for stock in df['Stock']:
    prices = np.random.normal(loc=df[df['Stock'] == stock]['Market Price'].iloc[0], scale=100, size=100)
    simulated_data[stock] = pd.DataFrame({'Date': dates, 'Price': prices})

# Simulated new stock recommendations
new_stocks = pd.DataFrame({
    'Stock': ['TLKM', 'BBCA', 'BMRI', 'ASII'],
    'Sector': ['Telecom', 'Banking', 'Banking', 'Automotive'],
    'Dividend Yield': [4.5, 3.2, 3.8, 2.9],
    'Growth Rate': [8.0, 10.0, 9.5, 7.0],
    'Current Price': [3500, 9500, 6000, 4500]
})

# Streamlit app
st.title("Interactive Portfolio Analysis Tool")

# Portfolio Analysis
st.header("Portfolio Analysis")
total_invested = df['Stock Value'].sum()
total_market_value = df['Market Value'].sum()
total_unrealized = df['Unrealized'].sum()
st.write(f"**Total Invested Value**: Rp {total_invested:,.0f}")
st.write(f"**Total Market Value**: Rp {total_market_value:,.0f}")
st.write(f"**Total Unrealized Gain/Loss**: Rp {total_unrealized:,.0f} ({(total_unrealized/total_invested)*100:.2f}%)")
st.subheader("Stock Details")
st.dataframe(df[['Stock', 'Balance', 'Avg Price', 'Market Price', 'Unrealized']])

# Plot portfolio composition
fig = px.pie(df, values='Market Value', names='Stock', title='Portfolio Composition')
st.plotly_chart(fig)

# Price Prediction (AI)
st.header("Stock Price Prediction")
selected_stock = st.selectbox("Select Stock for Prediction", df['Stock'])
if selected_stock:
    stock_data = simulated_data[selected_stock]
    X = np.arange(len(stock_data)).reshape(-1, 1)
    y = stock_data['Price'].values
    model = LinearRegression()
    model.fit(X, y)
    future_days = 30
    future_X = np.arange(len(stock_data), len(stock_data) + future_days).reshape(-1, 1)
    predictions = model.predict(future_X)
    future_dates = [stock_data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
    
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
    hist_df = stock_data[['Date', 'Price']].rename(columns={'Price': 'Predicted Price'})
    plot_df = pd.concat([hist_df, pred_df])
    
    fig = px.line(plot_df, x='Date', y='Predicted Price', title=f"Price Prediction for {selected_stock}")
    st.plotly_chart(fig)
    st.write(f"Predicted Price in 30 Days: Rp {predictions[-1]:,.0f}")

# What If Simulation
st.header("What If Simulation")
st.subheader("Simulate Price Change")
stock_sim = st.selectbox("Select Stock to Simulate", df['Stock'], key='sim_stock')
price_change = st.slider("Price Change (%)", -50.0, 50.0, 0.0)
if stock_sim:
    sim_df = df.copy()
    idx = sim_df[sim_df['Stock'] == stock_sim].index
    sim_df.loc[idx, 'Market Price'] *= (1 + price_change / 100)
    sim_df['Market Value'] = sim_df['Balance'] * sim_df['Market Price']
    sim_df['Unrealized'] = sim_df['Market Value'] - sim_df['Stock Value']
    new_total_market = sim_df['Market Value'].sum()
    new_unrealized = sim_df['Unrealized'].sum()
    st.write(f"**New Total Market Value**: Rp {new_total_market:,.0f}")
    st.write(f"**New Unrealized Gain/Loss**: Rp {new_unrealized:,.0f}")
    st.dataframe(sim_df[['Stock', 'Balance', 'Market Price', 'Unrealized']])

# Buy/Sell Recommendations
st.header("Buy/Sell Recommendations")
recommendations = []
for _, row in df.iterrows():
    unrealized_pct = (row['Unrealized'] / row['Stock Value']) * 100
    stock_data = simulated_data[row['Stock']]
    X = np.arange(len(stock_data)).reshape(-1, 1)
    y = stock_data['Price'].values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.coef_[0]
    if unrealized_pct < -10 or trend < 0:
        recommendations.append({'Stock': row['Stock'], 'Recommendation': 'Sell', 'Reason': 'Significant loss or downward trend'})
    elif unrealized_pct > 10 or trend > 0:
        recommendations.append({'Stock': row['Stock'], 'Recommendation': 'Hold/Buy', 'Reason': 'Good performance or upward trend'})
    else:
        recommendations.append({'Stock': row['Stock'], 'Recommendation': 'Hold', 'Reason': 'Stable performance'})
rec_df = pd.DataFrame(recommendations)
st.dataframe(rec_df)

# New Stock Recommendations
st.header("New Stock Recommendations")
st.write("Based on sector, dividend yield, and growth potential:")
st.dataframe(new_stocks)
selected_new_stock = st.selectbox("Select New Stock to Add", new_stocks['Stock'])
if st.button("Add to Portfolio"):
    new_row = pd.DataFrame({
        'Stock': [selected_new_stock],
        'Lot Balance': [1.0],
        'Balance': [100],
        'Avg Price': [new_stocks[new_stocks['Stock'] == selected_new_stock]['Current Price'].iloc[0]],
        'Stock Value': [100 * new_stocks[new_stocks['Stock'] == selected_new_stock]['Current Price'].iloc[0]],
        'Market Price': [new_stocks[new_stocks['Stock'] == selected_new_stock]['Current Price'].iloc[0]],
        'Market Value': [100 * new_stocks[new_stocks['Stock'] == selected_new_stock]['Current Price'].iloc[0]],
        'Unrealized': [0]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    st.write("Stock added to portfolio!")
    st.dataframe(df)

# Portfolio Modification
st.header("Modify Portfolio")
st.subheader("Remove Stock")
stock_remove = st.selectbox("Select Stock to Remove", df['Stock'], key='remove_stock')
if st.button("Remove Stock"):
    df = df[df['Stock'] != stock_remove]
    st.write(f"{stock_remove} removed from portfolio!")
    st.dataframe(df)

st.subheader("Adjust Stock Balance")
stock_adjust = st.selectbox("Select Stock to Adjust", df['Stock'], key='adjust_stock')
new_balance = st.number_input("New Balance (Shares)", min_value=0, value=int(df[df['Stock'] == stock_adjust]['Balance'].iloc[0]) if stock_adjust in df['Stock'].values else 0)
if st.button("Update Balance"):
    idx = df[df['Stock'] == stock_adjust].index
    df.loc[idx, 'Balance'] = new_balance
    df.loc[idx, 'Stock Value'] = new_balance * df.loc[idx, 'Avg Price']
    df.loc[idx, 'Market Value'] = new_balance * df.loc[idx, 'Market Price']
    df.loc[idx, 'Unrealized'] = df.loc[idx, 'Market Value'] - df.loc[idx, 'Stock Value']
    df.loc[idx, 'Lot Balance'] = new_balance / 100
    st.write(f"Balance for {stock_adjust} updated!")
    st.dataframe(df)

# Save updated portfolio (simulated, as file I/O is not allowed in Pyodide)
st.write("Updated portfolio (simulated save):")
st.dataframe(df)