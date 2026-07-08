import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("🏦 Bank Term Deposit Subscription Predictor")
st.write("Kya customer term deposit karega?")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_models()

# Input Fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    job = st.selectbox("Job", ["admin", "management", "blue-collar", "student", "retired", "unemployed"])
    marital = st.selectbox("Marital Status", ["single", "married", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    balance = st.number_input("Average Yearly Balance", value=500.0)

with col2:
    housing = st.selectbox("Housing Loan", ["yes", "no"])
    loan = st.selectbox("Personal Loan", ["yes", "no"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
    month = st.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
    duration = st.number_input("Duration of Last Call (seconds)", min_value=0, value=180)

# More inputs
campaign = st.number_input("Number of contacts in this campaign", min_value=1, value=2)
pdays = st.number_input("Days passed after previous campaign", value=-1)
previous = st.number_input("Number of previous contacts", min_value=0, value=0)
poutcome = st.selectbox("Previous Campaign Outcome", ["success", "failure", "unknown", "other"])

if st.button("🔮 Predict"):
    # Load encoders and training columns
    job_encoder = joblib.load('models/job_encoder.pkl')
    month_encoder = joblib.load('models/month_encoder.pkl')
    train_columns = joblib.load('models/train_columns.pkl')
    
    # Prepare data
    input_data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': 'no',
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': 15,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }

    input_df = pd.DataFrame([input_data])

    # Apply One-Hot Encoding (same as training)
    onehot_cols = ["marital", "education", "default", "housing", "loan", "contact", "poutcome"]
    input_df = pd.get_dummies(input_df, columns=onehot_cols, drop_first=True)

    # Apply Target Encoding (same as training)
    input_df['job'] = job_encoder.transform(input_df[['job']])
    input_df['month'] = month_encoder.transform(input_df[['month']])

    # Align columns with training data
    input_df = input_df.reindex(columns=train_columns, fill_value=0)

    # Convert to float
    input_df = input_df.astype(float)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Customer will subscribe to Term Deposit!")
    else:
        st.error("❌ Customer will NOT subscribe.")

    st.info(f"Confidence: {round(model.predict_proba(input_scaled)[0][1]*100, 2)}%")