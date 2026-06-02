"""
Loan Approval Prediction - Streamlit App
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: #666; font-size: 1rem; margin-bottom: 2rem;
    }
    .result-approved {
        background: #e8f5e9; border-left: 6px solid #2ecc71;
        padding: 1.5rem; border-radius: 8px; margin-top: 1rem;
    }
    .result-rejected {
        background: #fce4e4; border-left: 6px solid #e74c3c;
        padding: 1.5rem; border-radius: 8px; margin-top: 1rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 8px;
        padding: 1rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏦 Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter applicant details to predict loan approval</div>', unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_names.pkl')

@st.cache_resource
def load_model():
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

model_loaded = os.path.exists(MODEL_PATH)
if model_loaded:
    model, scaler, feature_names = load_model()
    st.success("✅ Model loaded successfully!")
else:
    st.warning("⚠️ Model not found. Please run the Jupyter notebook first to train and save the model.")
    st.info("Run: `jupyter notebook notebook/loan_approval_prediction.ipynb`")

st.markdown("---")

# ── Input Form ────────────────────────────────────────────────
st.subheader("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    income_annum = st.number_input("Annual Income (₹)", min_value=0, max_value=10000000, value=500000, step=10000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, max_value=50000000, value=1000000, step=50000)

with col2:
    loan_term = st.slider("Loan Term (years)", min_value=2, max_value=20, value=10)
    cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=650)
    residential_assets = st.number_input("Residential Assets Value (₹)", min_value=0, max_value=50000000, value=2000000, step=100000)
    commercial_assets = st.number_input("Commercial Assets Value (₹)", min_value=0, max_value=50000000, value=500000, step=100000)
    luxury_assets = st.number_input("Luxury Assets Value (₹)", min_value=0, max_value=50000000, value=300000, step=100000)
    bank_asset = st.number_input("Bank Assets Value (₹)", min_value=0, max_value=50000000, value=1000000, step=100000)

# ── CIBIL Score Helper ────────────────────────────────────────
if cibil_score < 500:
    st.warning("⚠️ Low CIBIL score — approval chances are low.")
elif cibil_score < 650:
    st.info("ℹ️ Average CIBIL score — approval depends on other factors.")
else:
    st.success("✅ Good CIBIL score — improves approval chances.")

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────
if st.button("🔍 Predict Loan Approval", use_container_width=True, type="primary"):
    if not model_loaded:
        st.error("Please train the model first by running the Jupyter notebook.")
    else:
        # Build input dict matching feature names
        edu_enc = 1 if education == "Graduate" else 0
        emp_enc = 1 if self_employed == "Yes" else 0

        input_data = {
            'no_of_dependents': no_of_dependents,
            'education': edu_enc,
            'self_employed': emp_enc,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets,
            'commercial_assets_value': commercial_assets,
            'luxury_assets_value': luxury_assets,
            'bank_asset_value': bank_asset,
        }

        # Align columns to trained feature names
        input_df = pd.DataFrame([input_data])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        try:
            # Try with scaling first (for LR/KNN), else without
            try:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]
            except Exception:
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]

            approved = prediction == 1
            confidence = max(proba) * 100

            if approved:
                st.markdown(f"""
                <div class="result-approved">
                    <h2 style='color:#2ecc71; margin:0'>✅ LOAN APPROVED</h2>
                    <p style='color:#333; font-size:1.1rem; margin:0.5rem 0'>
                        The applicant is likely to get the loan approved.
                    </p>
                    <p style='color:#555'>Confidence: <b>{confidence:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-rejected">
                    <h2 style='color:#e74c3c; margin:0'>❌ LOAN REJECTED</h2>
                    <p style='color:#333; font-size:1.1rem; margin:0.5rem 0'>
                        The application does not meet the required criteria.
                    </p>
                    <p style='color:#555'>Confidence: <b>{confidence:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

            # Show key factors
            st.markdown("### 📊 Key Factors Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("CIBIL Score", cibil_score, "Good" if cibil_score >= 650 else "Low")
            c2.metric("Annual Income", f"₹{income_annum:,.0f}")
            c3.metric("Total Assets", f"₹{residential_assets+commercial_assets+luxury_assets+bank_asset:,.0f}")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Make sure the model was trained on the same features.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.8rem;'>"
    "Loan Approval Prediction"
    "</p>",
    unsafe_allow_html=True
)
