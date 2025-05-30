import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime
import numpy as np

# Streamlit page configuration
st.set_page_config(
    page_title="Enhanced Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #2a2a2a;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #2a2a2a;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    h1, h2, h3, h4 {
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("Forecast Settings")
    forecast_days = st.slider(
        "Forecast Period (Days)",
        min_value=1,
        max_value=90,
        value=30,
        help="Select the number of days to forecast into the future."
    )
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This dashboard provides detailed sales analyses and forecasts using Prophet. Explore the tabs for insights.")

# Data loading and preparation functions
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_638832585317881453.csv')
        df = df[df['Date Paid'].notna() & (df['Amt Paid'] > 0)]
        df['Date Paid'] = pd.to_datetime(df['Date Paid'], format='%m/%d/%Y %I:%M:%S %p')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def prepare_daily_sales(df):
    orders = df.groupby('Record #').first()[['Date Paid', 'Amt Paid']]
    orders['ds'] = orders['Date Paid'].dt.normalize()
    daily_sales = orders.groupby('ds')['Amt Paid'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    return daily_sales

@st.cache_data
def prepare_monthly_sales(df):
    df['Month'] = df['Date Paid'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['Amt Paid'].sum().reset_index()
    monthly_sales['Month'] = monthly_sales['Month'].astype(str)
    monthly_sales['Growth'] = monthly_sales['Amt Paid'].pct_change() * 100
    return monthly_sales

@st.cache_data
def prepare_product_sales(df, product):
    product_df = df[df['Title'] == product]
    product_orders = product_df.groupby('Record #').first()[['Date Paid', 'Amt Paid']]
    product_orders['ds'] = product_orders['Date Paid'].dt.normalize()
    product_daily_sales = product_orders.groupby('ds')['Amt Paid'].sum().reset_index()
    product_daily_sales.columns = ['ds', 'y']
    return product_daily_sales

# Prophet model training
@st.cache_resource
def train_prophet_model(data, periods):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative'
    )
    model.fit(data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# Main app
st.title("Enhanced Sales Forecasting Dashboard")
st.markdown("Dive into detailed sales analyses, including monthly breakdowns, product forecasts, and more.")

# Load data
df = load_data()
if df is None:
    st.stop()

# Prepare datasets
daily_sales = prepare_daily_sales(df)
monthly_sales = prepare_monthly_sales(df)

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["Overall Sales", "Monthly Analysis", "Product Forecasts", "Insights"])

# Tab 1: Overall Sales
with tab1:
    st.header("Overall Sales")
    
    # Key metrics
    total_sales = daily_sales['y'].sum()
    avg_daily_sales = daily_sales['y'].mean()
    last_date = daily_sales['ds'].max()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Sales", f"${total_sales:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Daily Sales", f"${avg_daily_sales:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Last Date", last_date.strftime('%Y-%m-%d'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Cumulative Sales
    st.subheader("Cumulative Sales")
    cumulative_sales = daily_sales.set_index('ds')['y'].cumsum().reset_index()
    model, forecast = train_prophet_model(daily_sales, forecast_days)
    forecast_cumulative = forecast.set_index('ds')['yhat'].cumsum().reset_index()
    last_historical = cumulative_sales.iloc[-1]
    forecast_cumulative['yhat'] = forecast_cumulative['yhat'] + last_historical['y']
    combined_cumulative = pd.concat([cumulative_sales.rename(columns={'y': 'yhat'}), forecast_cumulative[forecast_cumulative['ds'] > last_historical['ds']]])
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=cumulative_sales['ds'], y=cumulative_sales['y'], mode='lines', name='Historical', line=dict(color='#007bff')))
    fig1.add_trace(go.Scatter(x=combined_cumulative['ds'], y=combined_cumulative['yhat'], mode='lines', name='Projected', line=dict(color='#ff6f61', dash='dash')))
    fig1.update_layout(title="Cumulative Sales Over Time", xaxis_title="Date", yaxis_title="Cumulative Sales ($)", template="plotly_dark", height=500)
    st.plotly_chart(fig1, use_container_width=True)

# Tab 2: Monthly Analysis
with tab2:
    st.header("Monthly Sales Analysis")
    
    # Monthly Sales Bar Chart
    st.subheader("Sales per Month")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=monthly_sales['Month'], y=monthly_sales['Amt Paid'], marker_color='#007bff'))
    fig2.update_layout(title="Monthly Sales", xaxis_title="Month", yaxis_title="Sales ($)", template="plotly_dark", height=500)
    st.plotly_chart(fig2, use_container_width=True)
    
    # In-Depth Monthly Analysis
    st.subheader("Month-to-Month Breakdown")
    monthly_sales_display = monthly_sales.copy()
    monthly_sales_display['Growth'] = monthly_sales_display['Growth'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    st.dataframe(monthly_sales_display.style.format({"Amt Paid": "${:,.2f}"}), use_container_width=True)
    
    st.markdown("""
    **Analysis:**
    - **Growth Trends**: Positive growth indicates increasing sales month-over-month, while negative growth suggests a decline.
    - **Seasonality**: Look for patterns that repeat annually, such as spikes during certain months.
    - **Current Month**: The latest month may be incomplete; compare with the forecast for a full-month projection.
    """)

# Tab 3: Product Forecasts
with tab3:
    st.header("Product Forecasts")
    
    # Top Products
    product_sales = df.groupby('Title')['Amt Paid'].sum().sort_values(ascending=False).head(10)
    top_products = product_sales.index.tolist()
    
    st.subheader("Top 10 Products by Sales")
    st.dataframe(product_sales.reset_index().rename(columns={'Amt Paid': 'Total Sales'}).style.format({"Total Sales": "${:,.2f}"}))
    
    # Product Selection and Forecast
    selected_product = st.selectbox("Select a Product for Forecast", top_products)
    product_daily_sales = prepare_product_sales(df, selected_product)
    
    if len(product_daily_sales) > 10:
        product_model, product_forecast = train_prophet_model(product_daily_sales, forecast_days)
        st.subheader(f"Forecast for {selected_product}")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=product_daily_sales['ds'], y=product_daily_sales['y'], mode='lines+markers', name='Historical', line=dict(color='#007bff')))
        fig3.add_trace(go.Scatter(x=product_forecast['ds'], y=product_forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#ff6f61', dash='dash')))
        fig3.update_layout(title=f"Daily Sales Forecast for {selected_product}", xaxis_title="Date", yaxis_title="Sales ($)", template="plotly_dark", height=500)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning(f"Insufficient data to forecast {selected_product}.")

# Tab 4: Additional Insights
with tab4:
    st.header("Additional Insights")
    
    # Sales by Store
    st.subheader("Sales by Store")
    store_sales = df.groupby('Store')['Amt Paid'].sum().reset_index()
    fig4 = go.Figure()
    fig4.add_trace(go.Pie(labels=store_sales['Store'], values=store_sales['Amt Paid'], marker_colors=['#007bff', '#ff6f61', '#28a745']))
    fig4.update_layout(title="Sales Distribution by Store", template="plotly_dark", height=500)
    st.plotly_chart(fig4, use_container_width=True)
    
    # Sales Trends
    st.subheader("Sales Trends")
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=daily_sales['ds'], y=daily_sales['y'], mode='lines', name='Daily Sales', line=dict(color='#007bff')))
    fig5.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend', line=dict(color='#ff6f61', dash='dash')))
    fig5.update_layout(title="Overall Sales Trends", xaxis_title="Date", yaxis_title="Sales ($)", template="plotly_dark", height=500)
    st.plotly_chart(fig5, use_container_width=True)

# Summary
st.markdown("### Summary")
st.write("""
This dashboard provides a deep dive into your sales data:
- **Overall Sales**: View cumulative sales and key metrics.
- **Monthly Analysis**: Examine sales per month with growth insights.
- **Product Forecasts**: Analyze top products and their future performance.
- **Insights**: Understand sales distribution and trends.
Use the sidebar to adjust the forecast period and explore the tabs for a comprehensive analysis.
""")