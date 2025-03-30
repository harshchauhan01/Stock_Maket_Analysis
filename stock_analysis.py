import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import streamlit as st
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Streamlit app title
st.title("Stock Market Comparison Analysis")

# Sidebar for user inputs
st.sidebar.header("User Inputs")

# List of available stocks
available_stocks = [
    'BPCL.NS', 'RELIANCE.NS', 'TATAMOTORS.NS', 'INFY.NS', 'TCS.NS',
    'HDFCBANK.NS', 'SBIN.NS', 'ITC.NS', 'HINDUNILVR.NS', 'ASIANPAINT.NS',
    'ICICIBANK.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'SUNPHARMA.NS','DRREDDY.NS',
    'MARUTI.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'WIPRO.NS', 'ADANIENT.NS', 'ONGC.NS',
    'NTPC.NS', 'COALINDIA.NS', 'LT.NS', 'TECHM.NS'
]

selected_stocks = st.sidebar.multiselect(
    "Select Stocks to Compare", 
    options=available_stocks, 
    default=['BPCL.NS', 'RELIANCE.NS', 'TATAMOTORS.NS']
)

# Today's date (dynamically updated)
today = pd.to_datetime(datetime.now().date())

# Date range selection
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", value=today)

# Test size selection
test_size = st.sidebar.slider("Test Size (as % of data)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

# Define colors dictionary at global scope
colors = {
    'BPCL.NS': ('blue', 'orange'), 
    'RELIANCE.NS': ('green', 'red'), 
    'TATAMOTORS.NS': ('purple', 'brown'),
    'INFY.NS': ('cyan', 'magenta'),
    'TCS.NS': ('black', 'gray'),
    'HDFCBANK.NS': ('darkblue', 'lightblue'),
    'SBIN.NS': ('darkgreen', 'lime'),
    'ITC.NS': ('darkred', 'pink'),
    'HINDUNILVR.NS': ('darkcyan', 'lightcyan'),
    'ASIANPAINT.NS': ('darkmagenta', 'violet'),
    'ICICIBANK.NS': ('teal', 'coral'),
    'AXISBANK.NS': ('indigo', 'salmon'),
    'KOTAKBANK.NS': ('olive', 'gold'),
    'SUNPHARMA.NS': ('maroon', 'peachpuff'),
    'DRREDDY.NS': ('darkorange', 'lightyellow'),
    'MARUTI.NS': ('navy', 'skyblue'),
    'BAJFINANCE.NS': ('darkviolet', 'plum'),
    'HCLTECH.NS': ('steelblue', 'lightpink'),
    'WIPRO.NS': ('sienna', 'khaki'),
    'ADANIENT.NS': ('crimson', 'lavender'),
    'ONGC.NS': ('forestgreen', 'lightgreen'),
    'NTPC.NS': ('darkslategray', 'aquamarine'),
    'COALINDIA.NS': ('chocolate', 'bisque'),
    'LT.NS': ('darkgoldenrod', 'palegoldenrod'),
    'TECHM.NS': ('deeppink', 'lightcoral')
}

# Define features at global scope
features = ['Open', 'High', 'Low', 'Volume']

# Function to download and prepare stock data
def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data found for {ticker} in the specified date range.")
        stock_data.reset_index(inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        return stock_data[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']]
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {str(e)}")
        return None

# Function to filter data for a specific date range
def filter_data(df, start_date, end_date):
    return df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Function to predict future OHLC values
def predict_future_ohlc(stock, df, model, scaler, start_date, end_date):
    future_ohlc = []
    last_row = df.iloc[-1]
    
    # Generate dates from start_date to end_date
    current_date = start_date
    while current_date <= end_date:
        # Use the last known values for Open, High, Low, and Volume as a simple approximation
        future_features = np.array([
            [float(last_row['Open']), float(last_row['High']), float(last_row['Low']), float(last_row['Volume'])]
        ])  # Shape: (1, 4)
        
        # Debug: Check the shape of future_features
        print(f"future_features shape for {stock} on {current_date}: {future_features.shape}")
        
        # Scale the features
        future_features_scaled = scaler.transform(future_features)
        
        # Predict the Close price
        predicted_close = model.predict(future_features_scaled)[0]
        
        # Append the predicted OHLC data
        future_ohlc.append({
            'Stock': stock,
            'Date': current_date.strftime('%Y-%m-%d'),
            'Close': float(predicted_close)
        })
        
        # Move to the next day
        current_date += timedelta(days=1)
    
    return future_ohlc

# Function to run analysis and return results
def run_analysis(selected_stocks, start_date, end_date, test_size):
    results = {}
    ohlc_data = []
    future_predictions = []

    for stock in selected_stocks:
        # Determine the end date for historical data (up to today if end_date is in the future)
        historical_end_date = min(today, pd.to_datetime(end_date))
        
        # Get historical data up to today (add 1 day to include historical_end_date)
        df = get_stock_data(stock, start_date, historical_end_date + timedelta(days=1))
        if df is None or df.empty:
            continue
        
        # Store last day's and today's OHLC values
        # Last day (day before today)
        last_day_df = df[df['Date'] < today]
        last_day_row = last_day_df.iloc[-1] if not last_day_df.empty else None
        # Today (current date)
        today_df = df[df['Date'] == today]
        today_row = today_df.iloc[0] if not today_df.empty else None

        if last_day_row is not None:
            last_date = last_day_df['Date'].iloc[-1].strftime('%Y-%m-%d')  # Access Date directly from DataFrame
            ohlc_data.append({
                'Stock': stock,
                'Date': f"{last_date} ",
                'Open': float(last_day_row['Open']),
                'High': float(last_day_row['High']),
                'Low': float(last_day_row['Low']),
                'Close': float(last_day_row['Close'])
            })
        
        if today_row is not None:
            today_date = today_df['Date'].iloc[0].strftime('%Y-%m-%d')  # Access Date directly from DataFrame
            ohlc_data.append({
                'Stock': stock,
                'Date': f"{today_date} ",
                'Open': float(today_row['Open']),
                'High': float(today_row['High']),
                'Low': float(today_row['Low']),
                'Close': float(today_row['Close'])
            })
        else:
            st.warning(f"No data available for {stock} on {today.strftime('%Y-%m-%d')} (Today).")
        
        # Features and target for training
        X = df[features]
        y = df['Close']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Dates for plotting
        dates_train = df['Date'][:len(X_train)]
        dates_test = df['Date'][len(X_train):]
        
        # Create and train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions on test data
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)
        
        # Store results, including the trained model
        results[stock] = {
            'df': df,
            'dates_train': dates_train,
            'dates_test': dates_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'model': model,
            'scaler': scaler
        }
        
        # Predict future dates if end_date is in the future
        if pd.to_datetime(end_date) > today:
            future_start_date = today + timedelta(days=1)
            future_ohlc = predict_future_ohlc(stock, df, model, scaler, future_start_date, pd.to_datetime(end_date))
            future_predictions.extend(future_ohlc)
    
    return results, ohlc_data, future_predictions

# Button to run full analysis
if st.button("Run Full Analysis"):
    if not selected_stocks:
        st.error("Please select at least one stock to analyze.")
    else:
        # Run analysis for the full date range
        results, ohlc_data, future_predictions = run_analysis(selected_stocks, start_date, end_date, test_size)

        # Display last day's and today's OHLC values
        if ohlc_data:
            st.subheader("Last Day's and Today's STOCK Values")
            ohlc_df = pd.DataFrame(ohlc_data)
            st.table(ohlc_df)
        else:
            st.warning("No OHLC data available for the selected stocks.")

        # Display future predictions if any
        if future_predictions:
            st.subheader("Predicted STOCK Values for Future Dates")
            future_df = pd.DataFrame(future_predictions)
            st.table(future_df)

        # Display metrics for each stock
        for stock in selected_stocks:
            if stock in results:
                st.subheader(f"Results for {stock}")
                st.write(f"MSE: {results[stock]['mse']:.2f}")
                st.write(f"RMSE: {results[stock]['rmse']:.2f}")
                st.write(f"R² Score: {results[stock]['r2']:.2f}")
            else:
                st.warning(f"No results available for {stock}.")

        # Individual stock plots
        if results:
            st.subheader("Individual Stock Analysis (Full Period)")
            fig1, axes = plt.subplots(len(selected_stocks), 1, figsize=(15, 5 * len(selected_stocks)))
            if len(selected_stocks) == 1:
                axes = [axes]  # Ensure axes is iterable for a single stock
            
            for i, stock in enumerate(selected_stocks):
                if stock in results:
                    # Plot training data
                    axes[i].plot(results[stock]['dates_train'], results[stock]['y_train'], 
                                label='Actual Train', color='blue', alpha=0.5)
                    axes[i].plot(results[stock]['dates_train'], results[stock]['y_pred_train'], 
                                label='Predicted Train', color='green', linestyle='--')
                    
                    # Plot test data
                    axes[i].plot(results[stock]['dates_test'], results[stock]['y_test'], 
                                label='Actual Test', color='red')
                    axes[i].plot(results[stock]['dates_test'], results[stock]['y_pred_test'], 
                                label='Predicted Test', color='orange', linestyle='--')
                    
                    axes[i].set_title(f'{stock} Stock Price Analysis')
                    axes[i].set_xlabel('Date')
                    axes[i].set_ylabel('Close Price')
                    axes[i].legend()
                    axes[i].grid(True)
                    axes[i].tick_params(axis='x', rotation=45)
                else:
                    axes[i].text(0.5, 0.5, f"No data for {stock}", 
                                horizontalalignment='center', verticalalignment='center')
            
            plt.tight_layout()
            st.pyplot(fig1)

            # Comparison table
            st.subheader("Stock Comparison Summary (Full Period)")
            comparison_data = []
            for stock in selected_stocks:
                if stock in results:
                    comparison_data.append({
                        'Stock': stock,
                        'MSE': results[stock]['mse'],
                        'RMSE': results[stock]['rmse'],
                        'R² Score': results[stock]['r2']
                    })
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df.round(2))
            else:
                st.warning("No comparison data available.")

            # Combined Plot - Test Period Only
            st.subheader("Comparison of All Stocks - Test Period (Full Period)")
            fig2 = plt.figure(figsize=(12, 6))
            for stock in selected_stocks:
                if stock in results:
                    plt.plot(results[stock]['dates_test'], results[stock]['y_test'], 
                            label=f'{stock} Actual', color=colors[stock][0])
                    plt.plot(results[stock]['dates_test'], results[stock]['y_pred_test'], 
                            label=f'{stock} Predicted', color=colors[stock][1], linestyle='--')

            plt.title('Comparison of All Stocks - Test Period')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)

# Button for last week analysis
if st.button("Analyze Last Week"):
    if not selected_stocks:
        st.error("Please select at least one stock to analyze.")
    else:
        # Calculate date range for last week
        end_date_last_week = today  # Use today's date dynamically
        start_date_last_week = end_date_last_week - timedelta(days=7)

        # Run analysis for the full date range first
        results, ohlc_data, future_predictions = run_analysis(selected_stocks, start_date, end_date, test_size)

        # Filter data for last week and plot
        if results:
            st.subheader("Individual Stock Analysis (Last Week)")
            fig3, axes = plt.subplots(len(selected_stocks), 1, figsize=(15, 5 * len(selected_stocks)))
            if len(selected_stocks) == 1:
                axes = [axes]  # Ensure axes is iterable for a single stock

            for i, stock in enumerate(selected_stocks):
                if stock in results:
                    # Filter data for last week
                    df_last_week = filter_data(results[stock]['df'], start_date_last_week, end_date_last_week)
                    if df_last_week.empty:
                        st.warning(f"No data available for {stock} in the last week.")
                        axes[i].text(0.5, 0.5, f"No data for {stock} (Last Week)", 
                                    horizontalalignment='center', verticalalignment='center')
                        continue

                    # Features and target for last week
                    X_last_week = df_last_week[features]
                    y_last_week = df_last_week['Close']

                    # Scale features
                    scaler = StandardScaler()
                    X_last_week_scaled = scaler.fit_transform(X_last_week)

                    # Predict using the previously trained model
                    y_pred_last_week = results[stock]['model'].predict(X_last_week_scaled)

                    # Plot
                    axes[i].plot(df_last_week['Date'], y_last_week, 
                                label='Actual', color='blue')
                    axes[i].plot(df_last_week['Date'], y_pred_last_week, 
                                label='Predicted', color='orange', linestyle='--')
                    
                    axes[i].set_title(f'{stock} Stock Price Analysis (Last Week)')
                    axes[i].set_xlabel('Date')
                    axes[i].set_ylabel('Close Price')
                    axes[i].legend()
                    axes[i].grid(True)
                    axes[i].tick_params(axis='x', rotation=45)
                else:
                    axes[i].text(0.5, 0.5, f"No data for {stock}", 
                                horizontalalignment='center', verticalalignment='center')

            plt.tight_layout()
            st.pyplot(fig3)

            # Combined Plot - Last Week
            st.subheader("Comparison of All Stocks - Last Week")
            fig4 = plt.figure(figsize=(12, 6))
            for stock in selected_stocks:
                if stock in results:
                    # Filter data for last week
                    df_last_week = filter_data(results[stock]['df'], start_date_last_week, end_date_last_week)
                    if df_last_week.empty:
                        continue

                    # Features and target for last week
                    X_last_week = df_last_week[features]
                    y_last_week = df_last_week['Close']

                    # Scale features
                    scaler = StandardScaler()
                    X_last_week_scaled = scaler.fit_transform(X_last_week)

                    # Predict
                    y_pred_last_week = results[stock]['model'].predict(X_last_week_scaled)

                    # Plot
                    plt.plot(df_last_week['Date'], y_last_week, 
                            label=f'{stock} Actual', color=colors[stock][0])
                    plt.plot(df_last_week['Date'], y_pred_last_week, 
                            label=f'{stock} Predicted', color=colors[stock][1], linestyle='--')

            plt.title('Comparison of All Stocks - Last Week')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig4)
        else:
            st.warning("No data available for the selected stocks.")

else:
    st.info("Select stocks and click 'Run Full Analysis' or 'Analyze Last Week' to see the results.")