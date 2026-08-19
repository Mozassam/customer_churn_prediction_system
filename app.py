
import os, pickle, json
import pandas as pd
import streamlit as st

ROOT=os.path.dirname(os.path.abspath(__file__))
MODEL_PATH=os.path.join(ROOT,"models","churn_model.pkl")
with open(MODEL_PATH,"rb") as f: model=pickle.load(f)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Prediction System")
st.write("Enter customer information to estimate churn risk and support retention decisions.")

with st.form("prediction_form"):
    c1,c2=st.columns(2)
    with c1:
        age=st.number_input("Age",18,100,35)
        tenure=st.number_input("Tenure (months)",1,120,12)
        monthly=st.number_input("Monthly charges",0.0,500.0,75.0)
        support=st.number_input("Support tickets",0,20,1)
        logins=st.number_input("Monthly logins",0,100,12)
        late=st.selectbox("Late payment",[0,1],format_func=lambda x:"No" if x==0 else "Yes")
        satisfaction=st.slider("Satisfaction score",1,10,7)
    with c2:
        contract=st.selectbox("Contract type",["Month-to-month","One year","Two year"])
        internet=st.selectbox("Internet service",["DSL","Fiber optic","None"])
        payment=st.selectbox("Payment method",["Electronic check","Credit card","Bank transfer","Mailed check"])
        paperless=st.selectbox("Paperless billing",[0,1],format_func=lambda x:"No" if x==0 else "Yes")
        dependents=st.selectbox("Has dependents",[0,1],format_func=lambda x:"No" if x==0 else "Yes")
        usage=st.number_input("Data usage (GB)",0.0,1000.0,50.0)
    submitted=st.form_submit_button("Predict churn risk",use_container_width=True)

if submitted:
    row=pd.DataFrame([{
        "age":age,"tenure_months":tenure,"monthly_charges":monthly,
        "support_tickets":support,"monthly_logins":logins,"late_payment":late,
        "contract_type":contract,"internet_service":internet,"payment_method":payment,
        "paperless_billing":paperless,"has_dependents":dependents,
        "satisfaction_score":satisfaction,"data_usage_gb":usage
    }])
    prob=float(model.predict_proba(row)[0,1])
    pred=int(prob>=0.5)
    st.subheader("Prediction")
    st.metric("Churn probability",f"{prob:.1%}")
    if prob>=.70:
        st.error("HIGH RISK — prioritize retention outreach.")
    elif prob>=.40:
        st.warning("MEDIUM RISK — consider a proactive retention offer.")
    else:
        st.success("LOW RISK — customer is currently less likely to churn.")
    st.progress(prob)
    st.caption("This prediction is a decision-support estimate, not a guarantee of future customer behavior.")
