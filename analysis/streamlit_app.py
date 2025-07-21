import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

# Load Data
df = pd.read_csv("data/project_financials.csv", parse_dates=["Month"])

# Preprocessing
df = df.sort_values("Month")
df["Revenue_Variance"] = df["Actual_Revenue"] - df["Planned_Revenue"]
df["Cost_Variance"] = df["Actual_Cost"] - df["Planned_Cost"]
df["Profit"] = df["Actual_Revenue"] - df["Actual_Cost"]
df["Planned_Profit"] = df["Planned_Revenue"] - df["Planned_Cost"]
df["Profit_Variance"] = df["Profit"] - df["Planned_Profit"]
df["Price_Impact"] = df["Planned_Revenue"] * (df["Price_Index"] - 1)
df["Volume_Impact"] = df["Planned_Revenue"] * (df["Volume_Index"] - 1)
df["Mix_Impact"] = df["Planned_Revenue"] * (df["Mix_Index"] - 1)
df["PVM_Impact"] = df["Price_Impact"] + df["Volume_Impact"] + df["Mix_Impact"]

# Sidebar Filters
st.sidebar.title("🔧 Filters")
projects = st.sidebar.multiselect("Select Projects", df["Project"].unique(), default=df["Project"].unique())
date_range = st.sidebar.date_input("Select Date Range", [df["Month"].min(), df["Month"].max()])

filtered = df[(df["Project"].isin(projects)) &
              (df["Month"] >= pd.to_datetime(date_range[0])) &
              (df["Month"] <= pd.to_datetime(date_range[1]))]

# Title
st.title("📊 FP&A + PVM + Project Controlling Dashboard")

# KPI Metrics
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("💰 Total Profit", f"{filtered['Profit'].sum():,.0f}")
kpi2.metric("🏗️ Total CapEx", f"{filtered['CapEx'].sum():,.0f}")
kpi3.metric("🛠️ Total OpEx", f"{filtered['OpEx'].sum():,.0f}")
kpi4.metric("📊 PVM Impact", f"{filtered['PVM_Impact'].sum():,.0f}")

# Revenue & Profit Trend
st.subheader("📈 Revenue & Profit Trend")
fig1 = px.line(filtered, x="Month", y=["Planned_Revenue", "Actual_Revenue", "Profit"], color="Project", markers=True)
st.plotly_chart(fig1, use_container_width=True)

# PVM Breakdown
st.subheader("🔍 PVM Decomposition")
pvm_fig = go.Figure()
pvm_fig.add_trace(go.Scatter(x=filtered["Month"], y=filtered["Price_Impact"], name="Price Impact"))
pvm_fig.add_trace(go.Scatter(x=filtered["Month"], y=filtered["Volume_Impact"], name="Volume Impact"))
pvm_fig.add_trace(go.Scatter(x=filtered["Month"], y=filtered["Mix_Impact"], name="Mix Impact"))
pvm_fig.update_layout(title="PVM Breakdown", xaxis_title="Month", yaxis_title="Impact")
st.plotly_chart(pvm_fig, use_container_width=True)

# Variance Table
st.subheader("📉 Variance Table")
st.dataframe(filtered[["Month", "Project", "Revenue_Variance", "Cost_Variance", "Profit_Variance"]].set_index("Month"))

# CapEx & OpEx Trend
st.subheader("🏗️ CapEx & OpEx")
capex_fig = px.bar(filtered, x="Month", y=["CapEx", "OpEx"], color="Project", barmode="group")
st.plotly_chart(capex_fig, use_container_width=True)

# Forecasting Section
st.subheader("📊 Profit & PVM Forecast (Linear Regression)")
proj_forecast = st.selectbox("Select Project for Forecasting", df["Project"].unique())
df_proj = df[df["Project"] == proj_forecast].copy()
df_proj = df_proj.sort_values("Month")
df_proj["MonthIndex"] = np.arange(len(df_proj))

future_months = 6
future_index = np.arange(len(df_proj), len(df_proj) + future_months)
future_dates = pd.date_range(df_proj["Month"].max() + pd.DateOffset(months=1), periods=future_months, freq="MS")

# Forecast function
def forecast_column(df, target):
    X = df[["MonthIndex"]]
    y = df[target]
    model = make_pipeline(FunctionTransformer(np.reshape, kw_args={"newshape": (-1, 1)}), LinearRegression())
    model.fit(X, y)
    y_fit = model.predict(X)
    y_future = model.predict(future_index.reshape(-1, 1))
    return y_fit, y_future

profit_fit, profit_pred = forecast_column(df_proj, "Profit")
pvm_fit, pvm_pred = forecast_column(df_proj, "PVM_Impact")

# Plot Forecasts
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=df_proj["Month"], y=df_proj["Profit"], name="Actual Profit", line=dict(color="green")))
fig_forecast.add_trace(go.Scatter(x=df_proj["Month"], y=profit_fit, name="Fitted Profit", line=dict(color="lime", dash="dot")))
fig_forecast.add_trace(go.Scatter(x=future_dates, y=profit_pred, name="Forecasted Profit", line=dict(color="darkgreen", dash="dash")))
fig_forecast.add_trace(go.Scatter(x=df_proj["Month"], y=df_proj["PVM_Impact"], name="Actual PVM Impact", line=dict(color="blue")))
fig_forecast.add_trace(go.Scatter(x=df_proj["Month"], y=pvm_fit, name="Fitted PVM Impact", line=dict(color="skyblue", dash="dot")))
fig_forecast.add_trace(go.Scatter(x=future_dates, y=pvm_pred, name="Forecasted PVM Impact", line=dict(color="navy", dash="dash")))
fig_forecast.update_layout(title="📈 Forecast: Profit & PVM", xaxis_title="Month", yaxis_title="Amount")
st.plotly_chart(fig_forecast, use_container_width=True)

# Download Option
st.subheader("📥 Download Filtered Data")
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv, file_name="filtered_financials.csv", mime="text/csv")

