import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Water Risk Prediction", layout="centered")

st.title("💧 Water Risk Prediction & Monitoring System")

st.write("This application predicts water risk based on supplied and consumed water data.")

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("AquaBalance_Coimbatore_Water_Dataset-1.csv")
except:
    st.error("Dataset file not found. Please upload the CSV file.")
    st.stop()

# -----------------------------
# Create Target Column
# -----------------------------
df["Water_Risk"] = np.where(
    df["water_consumed_liters"] > df["water_supplied_liters"], 1, 0
)

# -----------------------------
# Try Loading Model (Optional)
# -----------------------------
try:
    model = pickle.load(open("water_risk_model.pkl", "rb"))
    model_loaded = True
except:
    model_loaded = False

# -----------------------------
# User Input Section
# -----------------------------
st.header("🔍 Predict Water Risk")

water_supplied = st.number_input("Enter Water Supplied (liters)", min_value=0.0)
water_consumed = st.number_input("Enter Water Consumed (liters)", min_value=0.0)

if st.button("Predict"):

    if model_loaded:
        prediction = model.predict([[water_supplied, water_consumed]])

        if prediction[0] == 1:
            st.error("⚠ High Water Risk Detected!")
        else:
            st.success("✅ No Water Risk Detected")
    else:
        # Simple logic if model not available
        if water_consumed > water_supplied:
            st.error("⚠ High Water Risk Detected!")
        else:
            st.success("✅ No Water Risk Detected")

# -----------------------------
# Data Visualization
# -----------------------------
st.header("📊 Data Visualization")

fig, ax = plt.subplots()
ax.scatter(df["water_supplied_liters"], df["water_consumed_liters"])
ax.set_xlabel("Water Supplied (liters)")
ax.set_ylabel("Water Consumed (liters)")
ax.set_title("Water Supplied vs Consumed")

st.pyplot(fig)

# -----------------------------
# Model Performance Section
# -----------------------------
st.header("📌 Model Performance")

X = df[["water_supplied_liters", "water_consumed_liters"]]
y = df["Water_Risk"]

if model_loaded:
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    st.write("Model Accuracy:", round(acc * 100, 2), "%")

    cm = confusion_matrix(y, y_pred)

    fig2, ax2 = plt.subplots()
    ax2.imshow(cm)
    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    st.pyplot(fig2)
else:
    st.info("Model file not found. Upload water_risk_model.pkl to see performance metrics.")
