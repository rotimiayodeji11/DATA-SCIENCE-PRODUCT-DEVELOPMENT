import streamlit as st
import pickle 
import pandas as pd

# Load data
df = pd.read_csv("Training Data.csv")

# Title and description
st.markdown(
    """
    <div style="text-align:center">
        <h1>Loan Prediction App</h1>
        <p>This app predicts loan risk using different models.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Choose Model for Predictions")
st.sidebar.markdown(
    """
    This sidebar allows you to choose the model for making predictions.
    Different models may provide different results based on the same input data.
    """
)
selected_model = st.sidebar.radio("Select Model", ("Random Forest", "Decision Tree", "GBC Model"))

# User inputs
with st.form("user_input_form"):
    st.header("User Information")
    income = st.number_input("Enter your income", min_value=10310, step=100)
    age = st.slider("Pick your age", min_value=21, max_value=79)
    experience = st.number_input("Years of Experience", min_value=0, max_value=20)
    marital = st.radio("Marital Status", df["Married/Single"].unique(), index=0)
    house = st.radio("House Ownership", df["House_Ownership"].unique(), index=0)
    car = st.radio("Car Ownership", df["Car_Ownership"].unique(), index=0)
    profession = st.selectbox("Select your Profession", df["Profession"].unique())
    state = st.selectbox("Select your State", df["STATE"].unique())
    city = st.selectbox("Select your City", df["CITY"].unique())
    job_years = st.slider("Current Job Years", min_value=0, max_value=20)
    house_years = st.number_input("Current House Years", min_value=10)
    submitted = st.form_submit_button("Predict")

# Display prediction result
if submitted:
    collected_data = {
        "Income": income,
        "Age": age,
        "Experience": experience,
        "Married/Single": marital,
        "House_Ownership": house,
        "Car_Ownership": car,
        "Profession": profession,
        "STATE": state,
        "CITY": city,
        "CURRENT_JOB_YRS": job_years,
        "CURRENT_HOUSE_YRS": house_years
    }

    collected_data_df = pd.DataFrame(collected_data, index=[0])

    # Perform one-hot encoding
    with open("one_hot_encoder.pkl", "rb") as filename:
        one_hot_encoder = pickle.load(filename)
    encoded_data = one_hot_encoder.transform(collected_data_df)

    # Load the selected model
    if selected_model == "Random Forest":
        with open("bestmodel.pkl", "rb") as model_file:
            model = pickle.load(model_file)
    elif selected_model == "Decision Tree":
        with open("dt_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
    elif selected_model == "GBC Model":
        with open("gbc_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)

    # Predict
    prediction = model.predict(encoded_data)[0]

    # Display prediction
    st.subheader("Prediction Result")
    if prediction == 0:
        st.error("High Risk! Your loan application might be risky.")
    else:
        st.success("Low Risk! Your loan application seems safe.")
