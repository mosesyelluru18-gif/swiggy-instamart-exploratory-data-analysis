"""
Hybrid Demand Sensing & Autonomous Replenishment Web Platform
Engineered for Web Execution (Streamlit + GBDT ML + Antigravity Agent Orchestrator + Webhooks)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import time
import requests
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingRegressor

st.set_page_config(
    page_title="Swiggy Instamart - Hybrid Demand Sensing Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .brand-title { font-size: 2.1rem; font-weight: 700; color: #FC8019; margin-bottom: 0px; }
    .brand-subtitle { font-size: 1.05rem; color: #535665; margin-bottom: 1.2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="brand-title">⚡ Swiggy Instamart: Hybrid Demand Sensing Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">ML Poisson Forecasting + Antigravity Decision Agent + Autonomous Webhook Dispatch</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# STEP 1: CACHED DATA INGESTION & FEATURE ENGINEERING
# -------------------------------------------------------------
@st.cache_data
def load_and_engineer_features():
    orders = pd.read_csv('order_transactions.csv')
    products = pd.read_csv('products.csv')
    stores = pd.read_csv('stores.csv')
    categories = pd.read_csv('categories.csv')
    suppliers = pd.read_csv('suppliers.csv')

    orders['OrderDate'] = pd.to_datetime(orders['OrderDate'])
    orders['Date'] = orders['OrderDate'].dt.date
    orders['Hour'] = orders['OrderDate'].dt.hour
    orders['DayOfWeek'] = orders['OrderDate'].dt.dayofweek
    orders['IsWeekend'] = orders['DayOfWeek'].isin([5, 6]).astype(int)

    hourly_demand = orders.groupby(['Date', 'StoreID', 'ProductID', 'Hour', 'DayOfWeek', 'IsWeekend']).agg(
        demand_units=('Quantity', 'sum'),
        gross_sales=('TotalPrice', 'sum'),
        discount_given=('DiscountApplied', 'mean')
    ).reset_index()

    dataset = (hourly_demand
               .merge(products[['ProductID', 'ProductName', 'CategoryID', 'UnitPrice', 'StockQuantity', 'SupplierID']], on='ProductID', how='left')
               .merge(stores[['StoreID', 'WarehouseCapacity']], on='StoreID', how='left')
               .merge(suppliers[['SupplierID', 'SupplierName', 'ContactEmail']], on='SupplierID', how='left'))

    return dataset, products, stores, suppliers

dataset, products, stores, suppliers = load_and_engineer_features()

# -------------------------------------------------------------
# STEP 2: ML SENSOR TRAINING (HISTOGRAM GBDT POISSON)
# -------------------------------------------------------------
@st.cache_resource
def train_gbdt_sensor(df):
    feature_cols = ['Hour', 'DayOfWeek', 'IsWeekend', 'UnitPrice', 'WarehouseCapacity']
    df_train = df.copy()

    for col in ['StoreID', 'ProductID', 'CategoryID']:
        means = df_train.groupby(col, observed=False)['demand_units'].mean()
        df_train[f'{col}_enc'] = df_train[col].map(means).fillna(df_train['demand_units'].mean())
        feature_cols.append(f'{col}_enc')

    dates = sorted(df_train['Date'].unique())
    split_idx = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    train_data = df_train[df_train['Date'].isin(train_dates)]
    test_data = df_train[df_train['Date'].isin(test_dates)].copy()

    model = HistGradientBoostingRegressor(
        loss='poisson',
        max_iter=100,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(train_data[feature_cols], train_data['demand_units'])
    test_data['predicted_rate'] = model.predict(test_data[feature_cols]).round(2)
    return model, test_data

ml_model, test_predictions = train_gbdt_sensor(dataset)

# -------------------------------------------------------------
# SIDEBAR CONTROLS & PARAMS
# -------------------------------------------------------------
st.sidebar.header("⚙️ Agent & Sensor Controls")
lead_time_days = st.sidebar.slider("Replenishment Lead Time (Days)", 0.5, 3.0, 1.5, 0.1)
safety_multiplier = st.sidebar.slider("Safety Stock Multiplier", 1.0, 2.0, 1.25, 0.05)
filter_store = st.sidebar.selectbox("Filter Store View", options=["ALL"] + sorted(list(stores['StoreID'].unique())))

st.sidebar.subheader("🔗 Webhook Integration")
webhook_endpoint = st.sidebar.text_input("Webhook Target URL", placeholder="https://webhook.site/your-id")
webhook_secret = st.sidebar.text_input("Webhook Secret Token", value="swiggy_secret_token_live", type="password")

# -------------------------------------------------------------
# STEP 3: AGENTIC REASONING & RUNTIME EVALUATION
# -------------------------------------------------------------
def run_agentic_policy(predictions_df, lead_time, multiplier, selected_store):
    df_eval = predictions_df.copy()
    if selected_store != "ALL":
        df_eval = df_eval[df_eval['StoreID'] == selected_store]

    actions = []
    for _, row in df_eval.iterrows():
        stock = row['StockQuantity']
        pred_rate = row['predicted_rate']
        price = row['UnitPrice']
        daily_proj = max(pred_rate * 24.0, 1.0)
        days_of_cover = round(stock / daily_proj, 2)
        rop = round(daily_proj * lead_time * multiplier, 1)

        if stock <= rop:
            reorder_units = int(np.ceil(rop * 2.0))
            order_cost = round(reorder_units * price, 2)
            severity = "CRITICAL" if days_of_cover < 1.0 else "WARNING"
            
            po_id = f"PO-SWG-{row['StoreID']}-{int(time.time()*1000)%1000000}"
            action = {
                "po_number": po_id,
                "timestamp": datetime.now().isoformat(),
                "severity": severity,
                "store_id": row['StoreID'],
                "product_id": row['ProductID'],
                "product_name": row['ProductName'],
                "current_stock": int(stock),
                "predicted_demand_rate_hr": float(pred_rate),
                "days_of_cover": days_of_cover,
                "reorder_threshold": rop,
                "recommended_units": reorder_units,
                "estimated_cost_inr": order_cost,
                "supplier_name": row['SupplierName'],
                "supplier_email": row['ContactEmail'],
                "agent_rationale": f"Hourly velocity ({pred_rate} u/hr) breaches buffer within {days_of_cover}d."
            }
            actions.append(action)
    return actions

agent_actions = run_agentic_policy(test_predictions, lead_time_days, safety_multiplier, filter_store)

# -------------------------------------------------------------
# WEB TABS
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 ML Sensing Telemetry", "🤖 Agent PO Reorder Queue", "⚡ Real-Time Webhook Dispatch"])

with tab1:
    col1, col2, col3, col4, col5 = st.columns(5)
    critical_actions = [a for a in agent_actions if a['severity'] == 'CRITICAL']
    warning_actions = [a for a in agent_actions if a['severity'] == 'WARNING']
    total_reorder_capital = sum(a['estimated_cost_inr'] for a in agent_actions)

    with col1:
        st.metric("SKU Nodes Evaluated", f"{len(test_predictions):,}")
    with col2:
        st.metric("Triggered PO Buffers", f"{len(agent_actions)}")
    with col3:
        st.metric("Critical Stockouts (<24h)", f"{len(critical_actions)}", delta_color="inverse")
    with col4:
        st.metric("Warning Buffer (<36h)", f"{len(warning_actions)}")
    with col5:
        st.metric("Committed Capital", f"₹{total_reorder_capital:,.0f}")

    st.markdown("---")

    g1, g2 = st.columns([1.2, 1])
    with g1:
        fig_scatter = px.scatter(
            test_predictions,
            x='StockQuantity',
            y='predicted_rate',
            size='UnitPrice',
            color='CategoryID',
            hover_name='ProductName',
            title="Sensed Hourly Demand Rate vs. Current Dark Store Stock",
            labels={'StockQuantity': 'Warehouse Stock', 'predicted_rate': 'Predicted Rate (Units/Hr)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with g2:
        top_replenish = pd.DataFrame(agent_actions).sort_values(by='estimated_cost_inr', ascending=False).head(8) if agent_actions else pd.DataFrame()
        if not top_replenish.empty:
            fig_bar = px.bar(
                top_replenish,
                x='estimated_cost_inr',
                y='product_name',
                orientation='h',
                title="Top Restock POs by Capital Allocation (₹)",
                color='severity',
                color_discrete_map={'CRITICAL': '#ef4444', 'WARNING': '#f59e0b'}
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No reorder actions generated for current filter selection.")

with tab2:
    st.subheader("Autonomous Agent Replenishment Ledger")
    if agent_actions:
        actions_df = pd.DataFrame(agent_actions)[[
            'severity', 'store_id', 'product_name', 'current_stock', 
            'predicted_demand_rate_hr', 'days_of_cover', 'recommended_units', 
            'estimated_cost_inr', 'supplier_name'
        ]]
        st.dataframe(actions_df, use_container_width=True, hide_index=True)
    else:
        st.success("All dark store nodes are operating within optimal safety buffers.")

with tab3:
    st.subheader("⚡ Live Webhook Event Dispatcher")
    st.markdown("Trigger real-time JSON webhooks to your central ERP, Slack webhook, or Webhook.site listener.")

    col_btn, col_count = st.columns([1, 4])
    with col_btn:
        dispatch_count = st.number_input("Max Events to Post", min_value=1, max_value=max(1, len(agent_actions)), value=min(5, max(1, len(agent_actions))))
        trigger_webhook = st.button("🚀 Fire Webhook Dispatch", type="primary", use_container_width=True)

    if trigger_webhook and agent_actions:
        events_to_post = agent_actions[:dispatch_count]
        st.write(f"Transmitting {len(events_to_post)} events...")
        results = []

        progress_bar = st.progress(0)
        for idx, act in enumerate(events_to_post):
            payload = {
                "event_id": f"evt_{int(time.time()*1000)}_{idx}",
                "event_name": "procurement.po_created",
                "timestamp": datetime.now().isoformat(),
                "severity": act['severity'],
                "purchase_order": {
                    "po_number": act['po_number'],
                    "store_id": act['store_id'],
                    "sku": act['product_id'],
                    "product_name": act['product_name'],
                    "reorder_units": act['recommended_units'],
                    "estimated_po_cost_inr": act['estimated_cost_inr']
                },
                "sensing_metrics": {
                    "current_stock": act['current_stock'],
                    "hourly_demand_rate": act['predicted_demand_rate_hr'],
                    "days_of_cover": act['days_of_cover'],
                    "reorder_threshold": act['reorder_threshold']
                },
                "supplier": {
                    "name": act['supplier_name'],
                    "contact": act['supplier_email']
                },
                "rationale": act['agent_rationale']
            }

            if webhook_endpoint and webhook_endpoint.startswith("http"):
                try:
                    res = requests.post(
                        webhook_endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json", "X-Webhook-Secret": webhook_secret},
                        timeout=5
                    )
                    status = f"HTTP {res.status_code}"
                except Exception as e:
                    status = f"FAILED: {str(e)[:40]}"
            else:
                status = "MOCK SIMULATION (200 OK)"

            results.append({
                "PO Number": act['po_number'],
                "Target SKU": act['product_name'],
                "Severity": act['severity'],
                "Reorder Units": act['recommended_units'],
                "Webhook Status": status
            })
            progress_bar.progress((idx + 1) / len(events_to_post))

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        st.success(f"Successfully processed {len(results)} webhook transactions.")

        st.subheader("Sample Webhook JSON Payload")
        st.json(payload)
