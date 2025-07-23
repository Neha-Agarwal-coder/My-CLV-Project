import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("MY RanForest Model")
feature_cols = joblib.load("MY FeatureCol")
scaler = joblib.load("MyScaler")

# Load dataset
df = pd.read_csv("digital_wallet.csv")

# Streamlit page configuration
st.set_page_config(page_title="CLV Predictor", layout="wide")

st.markdown("""
    <h1 style='text-align: center; font-size: 108px; font-weight: bold; color: #1f77b4; margin-bottom: 20px;'>
        🧮 Customer Lifetime Value (CLV) Predictor
    </h1>
""", unsafe_allow_html=True)



# Custom styling
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab"] {
            font-size: 32px !important;
            font-weight: 600 !important;
            padding: 12px 20px !important;
        }
        h1, h2, h3 {
            font-size: 26px !important;
            font-weight: 700;
        }
        .stButton>button {
            font-size: 18px;
            padding: 10px 22px;
        }
        .stSlider > div {
            padding-top: 10px;
            padding-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📈 CLV Prediction", "📊 Dataset Overview", "ℹ️ About"])

# --- Home Tab ---
with tab1:
    st.markdown("## Welcome to CLV Predictor")
    st.markdown("""
    Want to know how valuable your customers really are?  
    Just enter a few simple details and get an instant estimate of their lifetime value.  
    Powered by Machine Learning. Built for smarter decisions.
    """)

    st.markdown("---")

    st.markdown("### What This App Does")
    st.markdown("""
    - Predicts **Customer Lifetime Value (CLV)**  
    - Helps you identify high-value customers  
    - Gives insights to improve marketing and retention  
    """)
    st.markdown("👉 Start by heading to the **CLV Prediction** tab.")


# --- Prediction Tab ---
with tab2:
    st.header("Predict Customer Lifetime Value")
    st.markdown("Enter the customer's information below:")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 80, 30)
            income_level = st.radio("Income Level", ["Low", "Medium", "High"])
            total_txns = st.number_input("Total Transactions", 0, 1000, 10)
            avg_txn_val = st.number_input("Avg. Transaction Value", 100.0, 200000.0, 500.0)
            total_spent = st.number_input("Total Spent", 100.0, 1500000.0, 5000.0)

        with col2:
            active_days = st.slider("Active Days in a Year", 1, 365, 180)
            last_txn_days = st.slider("Days Since Last Transaction", 0, 365, 30)
            satisfaction = st.slider("Customer Satisfaction Score", 1, 5, 4)
            payment_method = st.selectbox("Preferred Payment Method", [
                "Credit Card", "Debit Card", "PayPal", "UPI", "Net Banking", "Wallet Balance"
            ])

        submitted = st.form_submit_button("Predict CLV")

    if submitted:
        income_map = {"Low": 0, "Medium": 1, "High": 2}
        income_encoded = income_map[income_level]

        input_data = {
            "Age": age,
            "Income_Level": income_encoded,
            "Total_Transactions": total_txns,
            "Avg_Transaction_Value": avg_txn_val,
            "Total_Spent": total_spent,
            "Active_Days": active_days,
            "Last_Transaction_Days_Ago": last_txn_days,
            "Customer_Satisfaction_Score": satisfaction,
        }

        for col in feature_cols:
            if col.startswith("Preferred_Payment_Method_"):
                input_data[col] = 1 if col == f"Preferred_Payment_Method_{payment_method}" else 0

        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)
        
         # Scale numeric features
        num_cols = ['Age', 'Total_Transactions', 'Avg_Transaction_Value', 'Total_Spent', 'Active_Days', 'Last_Transaction_Days_Ago']
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Estimated Customer Lifetime Value: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# --- Dataset Overview Tab ---
with tab3:
    st.header("Dataset Overview")
    st.markdown(f"**Rows:** {df.shape[0]} &nbsp;&nbsp;&nbsp; **Columns:** {df.shape[1]}")
    st.write("### Sample Dataset:")
    st.dataframe(df.head(10), use_container_width=True)

# --- About Tab ---
with tab4:
    st.header("About this App")
    st.markdown("""
    This tool predicts **Customer Lifetime Value (CLV)** using machine learning.  
    It analyzes past transaction history, spending behavior, and customer engagement.

    **Features used**:
    - Age, Income Level  
    - Transaction Stats (Count, Value, Spent)  
    - Engagement (Active Days, Time since Last Purchase)  
    - Satisfaction Score  
    - Payment Preferences  

    Created using **Streamlit**, **Scikit-learn**, and a **Random Forest** model.
    """)
