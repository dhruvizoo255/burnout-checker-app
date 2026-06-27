import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load(
    r"C:\Users\kapoo\burnsightAI\models\burnout_xgb.pkl"
)

st.set_page_config(
    page_title="BurnSight AI",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 BurnSight AI")
st.subheader(
    "Early Workforce Risk Intelligence System"
)

st.sidebar.header("Employee Information")

age = st.sidebar.slider(
    "Age",
    18,
    60,
    30
)

distance = st.sidebar.slider(
    "Distance From Home",
    1,
    50,
    10
)

income = st.sidebar.slider(
    "Monthly Income",
    10000,
    200000,
    50000
)

wlb = st.sidebar.slider(
    "Work Life Balance",
    1,
    4,
    3
)

js = st.sidebar.slider(
    "Job Satisfaction",
    1,
    4,
    3
)

es = st.sidebar.slider(
    "Environment Satisfaction",
    1,
    4,
    3
)

ji = st.sidebar.slider(
    "Job Involvement",
    1,
    4,
    3
)

rs = st.sidebar.slider(
    "Relationship Satisfaction",
    1,
    4,
    3
)

overtime = st.sidebar.selectbox(
    "OverTime",
    [0, 1]
)

overall_satisfaction = (
    es + js + rs
) / 3

engagement_score = (
    ji + wlb
) / 2

salary_growth_ratio = (
    income / (age - 17)
)

commute_score = distance

WCI = (
    25 * (wlb / 4)
    + 25 * (js / 4)
    + 20 * (es / 4)
    + 15 * (ji / 4)
    + 15 * (rs / 4)
)

sample = pd.DataFrame([{
    "Age": age,
    "DistanceFromHome": distance,
    "MonthlyIncome": income,
    "WorkLifeBalance": wlb,
    "JobSatisfaction": js,
    "EnvironmentSatisfaction": es,
    "JobInvolvement": ji,
    "RelationshipSatisfaction": rs,
    "OverTime": overtime,
    "overall_satisfaction":
        overall_satisfaction,
    "engagement_score":
        engagement_score,
    "salary_growth_ratio":
        salary_growth_ratio,
    "commute_score":
        commute_score,
    "WCI":
        WCI
}])

prediction = model.predict(sample)[0]
prob = model.predict_proba(sample).max()

risk_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}

st.metric(
    "Workforce Capacity Index",
    f"{WCI:.1f}/100"
)

st.metric(
    "Burnout Risk",
    risk_map[prediction]
)

st.metric(
    "Confidence",
    f"{prob:.1%}"
)
st.divider()
st.subheader("Employee Digital Twin")

future_wlb = min(
    wlb + 1,
    4
)

future_js = min(
    js + 1,
    4
)

future_es = min(
    es + 1,
    4
)

future_WCI = (
    25 * (future_wlb / 4)
    + 25 * (future_js / 4)
    + 20 * (future_es / 4)
    + 15 * (ji / 4)
    + 15 * (rs / 4)
)

gain = future_WCI - WCI

st.metric(
    "Projected WCI",
    f"{future_WCI:.1f}/100",
    delta=f"+{gain:.1f}"
)

st.info(
    "Simulation assumes improvements in work-life balance, satisfaction and work environment."
)