import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import time

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Churn Intelligence", layout="wide")

# =========================
# CUSTOM CSS (PRO UI)
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

/* Card */
.card {
    background: linear-gradient(145deg, #1c1f26, #111);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

/* Title */
h1, h2, h3 {
    color: white;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #f77fbe, #ff4ecd);
    color: white;
    border-radius: 10px;
    height: 3em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD
# =========================
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Customer Input")

tenure = st.sidebar.slider("Months", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0, 200, 50)

st.sidebar.markdown("---")
st.sidebar.caption("Adjust parameters to simulate customer behavior")

# =========================
# HEADER
# =========================
st.title("📊 Customer Churn Intelligence")
st.caption("AI-powered risk analysis dashboard")

# =========================
# INPUT PREP
# =========================
input_dict = {col: 0 for col in columns}

if "tenure" in input_dict:
    input_dict["tenure"] = tenure

if "MonthlyCharges" in input_dict:
    input_dict["MonthlyCharges"] = monthly

input_df = pd.DataFrame([input_dict])

# =========================
# LOADING
# =========================
with st.spinner("Analyzing customer behavior..."):
    time.sleep(1)

prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]

# =========================
# KPI CARDS
# =========================
col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='card'>📅 Tenure<br><h2>{tenure} เดือน</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'>💰 Charges<br><h2>{monthly} บาท</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'>⚠️ Risk<br><h2>{proba:.2%}</h2></div>", unsafe_allow_html=True)

st.divider()

# =========================
# GAUGE
# =========================
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=proba * 100,
    number={'suffix': "%"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#f77fbe"},
        'steps': [
            {'range': [0, 40], 'color': "#2ecc71"},
            {'range': [40, 70], 'color': "#f1c40f"},
            {'range': [70, 100], 'color': "#e74c3c"}
        ]
    }
))
st.plotly_chart(fig, use_container_width=True)

# =========================
# RESULT
# =========================
st.subheader("📌 Prediction")

if prediction == 1:
    st.error("High Risk: Customer likely to churn")
else:
    st.success("Low Risk: Customer likely to stay")

# =========================
# SMART INSIGHT ⭐
# =========================
st.subheader("🧠 AI Insight")

if proba > 0.7:
    st.warning("Immediate retention action recommended (discount / call)")
elif proba > 0.4:
    st.info("Monitor customer behavior closely")
else:
    st.success("Customer is stable")

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("📊 Key Drivers")

importances = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(8)

st.bar_chart(feat_df.set_index("Feature"))

# =========================
# FOOTER
# =========================
st.divider()
st.caption("© 2026 Churn Intelligence System | Educational Use Only")