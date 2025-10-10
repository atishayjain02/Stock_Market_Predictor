import streamlit as st
from datetime import date, timedelta
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import google.generativeai as genai

# Page config and theming
st.set_page_config(page_title="Stock Prediction App", page_icon="📈", layout="wide")
PLOTLY_TEMPLATE = "plotly_dark"

st.title('📈 Stock Prediction App (ML Models)')

stocks = (
    'SUZLON.NS',
    'IDEA.NS',
    'SOUTHBANK.NS',
    'TRIDENT.NS',
    'BCG.NS',
    'JPPOWER.NS',
    'YESBANK.NS',
    'RELIANCE.NS',
    'TATAMOTORS.NS',
    'NHPC.NS'
)

# Sidebar controls
st.sidebar.header("Controls")

selected_stock = st.sidebar.selectbox('Select Stock', stocks, index=0)

default_start = date(2015, 1, 1)
default_end = date.today()
start_date = st.sidebar.date_input("Start date", default_start, min_value=date(2000, 1, 1), max_value=default_end)
end_date = st.sidebar.date_input("End date", default_end, min_value=start_date, max_value=default_end)

# Model controls
st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Model", ["Linear Regression", "Decision Tree", "Random Forest"], index=2)
test_size_pct = st.sidebar.slider("Test size (%)", 10, 40, 20, help="Portion of latest data used for testing (chronological split)")

if model_choice == "Decision Tree":
    dt_max_depth = st.sidebar.slider("Max depth", 3, 30, 10)
    dt_min_leaf = st.sidebar.slider("Min samples leaf", 1, 20, 5)
elif model_choice == "Random Forest":
    rf_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, step=50)
    rf_max_depth = st.sidebar.slider("Max depth", 3, 30, 12)
    rf_min_leaf = st.sidebar.slider("Min samples leaf", 1, 20, 5)

# Forecasting controls
st.sidebar.header("Forecast")
forecast_years = st.sidebar.slider("Years to forecast", 1, 4, 1)
forecast_days = forecast_years * 252  # Approx trading days per year

NUM_LAGS = 5

@st.cache_data(show_spinner=False)
def load_data(ticker, start_dt, end_dt):
    data = yf.download(ticker, start=start_dt, end=end_dt)
    data.reset_index(inplace=True)
    if data.empty or "Close" not in data.columns:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[0] else col[1] for col in data.columns]
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Close"])
    # Technicals
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["BB_mid"] = data["MA20"]
    data["BB_std"] = data["Close"].rolling(20).std()
    data["BB_upper"] = data["BB_mid"] + 2 * data["BB_std"]
    data["BB_lower"] = data["BB_mid"] - 2 * data["BB_std"]
    # RSI (14) using Wilder's smoothing
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    window = 14
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    data["RSI14"] = 100 - (100 / (1 + rs))
    return data

with st.spinner("Loading data…"):
    data = load_data(selected_stock, start_date, end_date)

if data.empty:
    st.error("No data found for this ticker and date range. Try another selection.")
    st.stop()

# Sidebar stock info metrics
today_open = data['Open'].iloc[-1]
today_close = data['Close'].iloc[-1]
prev_close = data['Close'].iloc[-2] if len(data) > 1 else today_open
change = today_close - prev_close
change_pct = (change / prev_close) * 100 if prev_close else 0.0

st.sidebar.subheader(f"Stock Info: {selected_stock}")
colA, colB = st.sidebar.columns(2)
colA.metric("Open", f"{today_open:.2f}", None)
colB.metric("Close", f"{today_close:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")

# Feature Engineering: Lag features
for lag in range(1, NUM_LAGS + 1):
    data[f'Close_lag{lag}'] = data['Close'].shift(lag)

data_ml = data.dropna().reset_index(drop=True)
feature_cols = [f'Close_lag{lag}' for lag in range(1, NUM_LAGS + 1)] + (['Open'] if 'Open' in data_ml.columns else [])
X = data_ml[feature_cols]
y = data_ml['Close']

# Layout tabs
tab_overview, tab_data, tab_features, tab_model, tab_forecast , tab_chat = st.tabs(["Overview", "Data", "Features", "Modeling", "Forecast" ,"Chatbot"])

with tab_overview:
    # Candlestick + Volume subplot
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.65, 0.2, 0.15]
    )

    fig.add_trace(go.Candlestick(
        x=data['Date'],
        open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
        name="OHLC",
        increasing_line_color='#2ECC71', decreasing_line_color='#E74C3C',
        showlegend=False
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['MA20'], mode='lines', name='MA20', line=dict(width=1.5, color='#F1C40F')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['MA50'], mode='lines', name='MA50', line=dict(width=1.5, color='#3498DB')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['MA200'], mode='lines', name='MA200', line=dict(width=1.5, color='#9B59B6')
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['BB_upper'], line=dict(width=1, color='rgba(255,255,255,0.3)'), name='BB Upper', showlegend=True
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['BB_lower'], line=dict(width=1, color='rgba(255,255,255,0.3)'), name='BB Lower', showlegend=True, fill='tonexty', fillcolor='rgba(255,255,255,0.05)'
    ), row=1, col=1)

    # Volume bars
    vol_colors = np.where(data['Close'] >= data['Open'], '#2ECC71', '#E74C3C')
    fig.add_trace(go.Bar(
        x=data['Date'], y=data['Volume'], marker_color=vol_colors, name='Volume', showlegend=False
    ), row=2, col=1)

    # RSI subplot
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['RSI14'], mode='lines', name='RSI(14)', line=dict(width=1.5, color='#1ABC9C')
    ), row=3, col=1)
    # RSI guide lines
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="#E74C3C", row=3, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="#2ECC71", row=3, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"{selected_stock} Price (Candlestick) with MAs & Volume",
        xaxis=dict(rangeslider=dict(visible=False), showspikes=True),
        xaxis2=dict(showspikes=True),
        xaxis3=dict(showspikes=True),
        yaxis_title="Price",
        yaxis2_title="Volume",
        yaxis3_title="RSI",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # Remove non-trading days gaps for clarity
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.subheader('Raw Data (tail)')
    st.dataframe(data.tail(200), use_container_width=True, height=400)

with tab_features:
    st.subheader("Feature Sample (X)")
    st.dataframe(X.head(), use_container_width=True)
    st.subheader("Target Sample (y)")
    st.write(y.head())

with tab_model:
    if len(X) < 30:
        st.warning("Not enough data after feature engineering to train a model.")
    else:
        # Chronological split
        test_size = test_size_pct / 100.0
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Model selection
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42, max_depth=dt_max_depth, min_samples_leaf=dt_min_leaf)
        else:
            model = RandomForestRegressor(
                n_estimators=rf_estimators, random_state=42, max_depth=rf_max_depth, min_samples_leaf=rf_min_leaf, n_jobs=-1
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_test_values = y_test.values.ravel()
        rmse = np.sqrt(mean_squared_error(y_test_values, y_pred))

        mcol1, mcol2 = st.columns(2)
        mcol1.metric("RMSE", f"{rmse:.2f}")
        mcol2.write(f"Model: {model_choice}")

        # Actual vs Predicted line chart
        dates_test = data_ml['Date'].iloc[split_idx:]
        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(x=dates_test, y=y_test_values, mode='lines', name='Actual', line=dict(color='#2ECC71')))
        fig_ap.add_trace(go.Scatter(x=dates_test, y=y_pred, mode='lines', name='Predicted', line=dict(color='#E74C3C', dash='dash')))
        fig_ap.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Actual vs Predicted (Test Set)",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_ap, use_container_width=True)

        # Scatter with ideal line
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=y_test_values, y=y_pred, mode='markers', name='Predicted', marker=dict(color='#3498DB', size=6, opacity=0.7)))
        minv = float(np.nanmin([y_test_values.min(), y_pred.min()]))
        maxv = float(np.nanmax([y_test_values.max(), y_pred.max()]))
        fig_sc.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode='lines', name='Ideal', line=dict(color='white', width=1)))
        fig_sc.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Predicted vs Actual Scatter",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Residuals
        residuals = y_test_values - y_pred
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=dates_test, y=residuals, mode='lines', name='Residuals', line=dict(color='#F39C12')))
        fig_res.add_hline(y=0, line_width=1, line_dash="dash", line_color="white")
        fig_res.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Residuals Over Time",
            xaxis_title="Date",
            yaxis_title="Residual",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_res, use_container_width=True)

        # Feature importances (if available)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fig_imp = go.Figure(go.Bar(
                x=feature_cols,
                y=importances,
                marker_color='#1ABC9C'
            ))
            fig_imp.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Feature Importances",
                xaxis_title="Features",
                yaxis_title="Importance",
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_imp, use_container_width=True)

with tab_forecast:
    if len(X) < 30:
        st.warning("Not enough data to produce a forecast.")
    else:
        # Refit model on full data for forecasting
        if model_choice == "Linear Regression":
            f_model = LinearRegression()
        elif model_choice == "Decision Tree":
            f_model = DecisionTreeRegressor(random_state=42, max_depth=dt_max_depth, min_samples_leaf=dt_min_leaf)
        else:
            f_model = RandomForestRegressor(
                n_estimators=rf_estimators, random_state=42, max_depth=rf_max_depth, min_samples_leaf=rf_min_leaf, n_jobs=-1
            )
        f_model.fit(X, y)

        # Prepare for forecasting beyond available data
        last_date = data_ml['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

        last_known_closes = data_ml['Close'].iloc[-NUM_LAGS:].tolist()
        last_open = data_ml['Open'].iloc[-1] if 'Open' in data_ml.columns else data_ml['Close'].iloc[-1]

        future_preds = []
        for _ in range(forecast_days):
            features = last_known_closes[-NUM_LAGS:] + ([last_open] if 'Open' in data_ml.columns else [])
            features_array = np.array(features).reshape(1, -1)
            pred = float(f_model.predict(features_array)[0])
            future_preds.append(pred)
            last_known_closes.append(pred)

        # Plot recent actuals + forecast
        lookback = min(252, len(data_ml))  # show last ~1y actuals
        recent_dates = list(data_ml['Date'].iloc[-lookback:])
        recent_actuals = list(data_ml['Close'].iloc[-lookback:])

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=recent_dates, y=recent_actuals, mode='lines', name='Actual (recent)', line=dict(color='#2ECC71')
        ))
        fig_fc.add_trace(go.Scatter(
            x=future_dates, y=future_preds, mode='lines', name='Forecast', line=dict(color='#E74C3C', dash='dash')
        ))
        fig_fc.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"{model_choice} Forecast for {forecast_years} Year(s)",
            xaxis_title="Date",
            yaxis_title="Stock Price",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        # Hide weekend gaps in x-axis for readability
        fig_fc.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig_fc, use_container_width=True)

with tab_chat:
    # --------------------------- GEMINI AI CHATBOT ---------------------------
    
    import yfinance as yf
    import streamlit as st
    import re
    import pandas as pd # Explicitly imported for clarity

    # --- API Key Setup ---
    # NOTE: It is HIGHLY recommended to use st.secrets or environment variables 
    # for your API key, not hardcoding it.
    GEMINI_API_KEY = "AIzaSyC2jh90m267UZEvJe3sXNMPQVHep7Qoe6g" 

    try:
        # ✅ Initialize Gemini Client
        client = genai.Client(api_key=)
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        # Use st.stop() if the client cannot be initialized
        # st.stop() 


    st.subheader("💬 Gemini AI Chatbot (Stock Market Assistant)")
    st.markdown("Ask about stock prices (e.g., **AAPL**, **TSLA**) or general market trends.")

    # Maintain chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        # Use st.chat_message for modern UI
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    # User input logic (using st.chat_input for better UX)
    if user_input := st.chat_input("Type your stock market question..."):

        # 1. Store and display user message
        st.session_state.messages.append({"role": "user", "text": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Process and generate response
        response_text = ""
        user_lower = user_input.lower()
        is_stock_query = False

        # Start a spinner while processing
        with st.spinner("Thinking..."):

            # --- Ticker Extraction Logic ---
            # 1. Look for explicit price/stock keywords
            if "price" in user_lower or "stock" in user_lower:
                is_stock_query = True
            
            # 2. Look for potential tickers (2-5 uppercase letters/numbers)
            ticker_match = re.search(r'\b[A-Z0-9.]{2,10}\b', user_input) # Updated to include '.' for indices/NSE
            
            # Combine conditions: if it looks like a stock query OR we found a strong ticker match
            if is_stock_query or ticker_match:
                
                # --- Define Ticker Symbol ---
                ticker_symbol = None
                
                # A. Prioritize a direct ticker match
                if ticker_match:
                    ticker_symbol = ticker_match.group(0).upper()
                
                # B. Static map for common Indian names (as a secondary check)
                company_map = {
                    "INFOSYS": "INFY.NS", "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS",
                    "HDFC": "HDFCBANK.NS", "ICICI": "ICICIBANK.NS", "WIPRO": "WIPRO.NS",
                    "HCL": "HCLTECH.NS", "SBI": "SBIN.NS"
                }
                
                # C. If a search term was found, check the map.
                if ticker_symbol:
                    # Check map for common names, otherwise use the found symbol
                    ticker_symbol = company_map.get(ticker_symbol, ticker_symbol)
                
                # If we have a potential ticker, attempt yfinance lookup
                if ticker_symbol:
                    try:
                        # --- Handle stock price queries via yfinance ---
                        stock = yf.Ticker(ticker_symbol)
                        hist = stock.history(period="1d")
                        
                        if not hist.empty:
                            price = hist["Close"].iloc[-1]
                            info = stock.info
                            full_name = info.get('longName', ticker_symbol)
                            currency = info.get('currency', '$')
                            
                            response_text = f"The latest closing price of **{full_name} ({ticker_symbol})** is **{currency}{price:,.2f}**."
                            is_stock_query = True # Confirm successful data retrieval
                        else:
                            # Data returned empty, fall through to LLM
                            response_text = f"I attempted to find price data for **{ticker_symbol}**, but the data source returned empty. I'll check my general knowledge instead."
                            is_stock_query = False 
                            
                    except Exception:
                        # yfinance failed (bad ticker, network issue), fall through to LLM
                        response_text = f"I couldn't fetch real-time data for **{ticker_symbol}**. I'll check my general knowledge instead."
                        is_stock_query = False
                else:
                    # Stock keywords were used, but no recognizable ticker was found
                    is_stock_query = False


            # --- Handle General Market Queries (or if yfinance failed) ---
            if not response_text or not is_stock_query:
                
                # Prepare conversation history for Gemini
                contents = []
                for msg in st.session_state.messages:
                    if msg["text"] != user_input: # Only include past messages for context
                        gemini_role = "user" if msg["role"] == "user" else "model"
                        contents.append({"role": gemini_role, "parts": [{"text": msg["text"]}]})

                # Add the current user input as the final message
                contents.append({"role": "user", "parts": [{"text": user_input}]})
                
                # Define system instruction (now supported by updated SDK)
                system_instruction = (
                    "You are an expert, helpful stock market assistant. "
                    "Provide concise and informative answers. If the question is not about "
                    "finance, stocks, or general knowledge, politely decline. "
                    "Maintain context from previous messages."
                )

                # Call Gemini (with the system_instruction parameter)
                try:
                    gemini_response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=contents,
                        system_instruction=system_instruction # This parameter requires the latest SDK
                    )
                    response_text = gemini_response.text
                except Exception as e:
                    response_text = f"⚠️ Gemini API Error: {str(e)}. Please check your API key and usage limits."


        # 3. Store and display model response
        st.session_state.messages.append({"role": "model", "text": response_text})
        
        with st.chat_message("model"):
            st.markdown(response_text)
            
        # 4. Rerun the script to display the new chat history properly
        st.rerun()

# Sidebar mini close price chart
mini = go.Figure()
mini.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close", mode='lines', line=dict(color='#1ABC9C', width=1.5)))
mini.update_layout(
    template=PLOTLY_TEMPLATE,
    margin=dict(l=0, r=0, t=20, b=0),
    height=180,
    xaxis=dict(visible=False),
    yaxis=dict(title="Close", tickfont=dict(size=10))
)
with st.sidebar:
    st.subheader("Close Price")
    st.plotly_chart(mini, use_container_width=True)
