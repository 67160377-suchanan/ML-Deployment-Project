# Customer Churn Prediction

## Problem
ลูกค้าบางส่วนมีแนวโน้มยกเลิกบริการ (Churn)  
ซึ่งส่งผลต่อรายได้ของธุรกิจ  

โปรเจคนี้ใช้ Machine Learning เพื่อทำนายลูกค้าที่มีความเสี่ยงจะยกเลิก

---

## Dataset
- Telco Customer Churn (Kaggle)
- ข้อมูลลูกค้า เช่น:
  - Tenure (ระยะเวลาใช้งาน)
  - Monthly Charges (ค่าบริการรายเดือน)

---

## Model
- Random Forest Classifier
- ใช้ metrics:
  - Precision
  - Recall
  - F1-score

---

## Web Application
พัฒนาโดยใช้ Streamlit

ฟีเจอร์:
- กรอกข้อมูลลูกค้า
- ทำนายความเสี่ยง (%)
- แสดง Gauge Chart
- แสดง Feature Importance
- Recommendation แนะนำการแก้ไข

---

## How to Run

```bash
streamlit run app.py