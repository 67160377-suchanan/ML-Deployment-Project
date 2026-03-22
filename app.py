import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================
# STYLE (PRO UI)
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding: 2rem;
}
div[data-testid="stMetric"] {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
}
.stButton>button {
    background-color: #f77fbe;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Customer Input")

tenure = st.sidebar.slider("Months with company", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0, 200, 50)

st.sidebar.markdown("---")
st.sidebar.info("📌 ปรับค่าด้านบน แล้วดูผลลัพธ์")

# =========================
# HEADER
# =========================
st.title("📊 Customer Churn Intelligence Dashboard")
st.markdown("AI วิเคราะห์ความเสี่ยงการยกเลิกของลูกค้าแบบเรียลไทม์")

st.divider()

# =========================
# PREP INPUT
# =========================
input_dict = {col: 0 for col in columns}

if "tenure" in input_dict:
    input_dict["tenure"] = tenure

if "MonthlyCharges" in input_dict:
    input_dict["MonthlyCharges"] = monthly

input_df = pd.DataFrame([input_dict])

# =========================
# LOADING EFFECT
# =========================
with st.spinner("🔄 AI กำลังวิเคราะห์..."):
    time.sleep(1)

prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]

# =========================
# METRIC CARDS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("📅 Tenure", f"{tenure} เดือน")
col2.metric("💰 Monthly Charges", f"{monthly} บาท")
col3.metric("⚠️ Risk", f"{proba:.2%}")

st.divider()

# =========================
# GAUGE CHART (WOW)
# =========================
st.subheader("🎯 Churn Risk Gauge")

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
# RESULT MESSAGE
# =========================
st.subheader("📌 Prediction Result")

if prediction == 1:
    st.error("⚠️ High Risk: ลูกค้ามีแนวโน้มจะยกเลิก")
else:
    st.success("✅ Low Risk: ลูกค้าน่าจะอยู่ต่อ")

# =========================
# SMART RECOMMENDATION ⭐
# =========================
st.subheader("💡 Recommendation")

if proba > 0.7:
    st.warning("ควรให้โปรโมชั่นหรือโทรติดต่อลูกค้าโดยด่วน")
elif proba > 0.4:
    st.info("ควรติดตามพฤติกรรมลูกค้าเพิ่มเติม")
else:
    st.success("ลูกค้ากลุ่มนี้มีความเสี่ยงต่ำ")

# =========================
# PROGRESS BAR
# =========================
st.subheader("🔥 Risk Level")
st.progress(float(proba))

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("📊 Top Influencing Features")

importances = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

st.bar_chart(feat_df.set_index("Feature"))

# =========================
# FOOTER
# =========================
st.divider()
st.caption("⚠️ This AI model is for educational purposes only.")