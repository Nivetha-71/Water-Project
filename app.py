import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/AquaBalance_Coimbatore_Water_Dataset-1.csv")
display(df.head())

df.isnull().sum()
df.fillna(0, inplace=True)

df["Water_Waste"] = np.where(
    df["water_consumed_liters"] > df["water_supplied_liters"], 1, 0
)

df["Water_Risk"] = np.where(df["supply_status"] == "Normal", 0, 1)

display(df.groupby("area_name")["complaints_count"].mean().sort_values(ascending=False).head())

plt.figure(figsize=(10,5))
sns.lineplot(x=df["date"], y=df["water_supplied_liters"], label="Supplied")
sns.lineplot(x=df["date"], y=df["water_consumed_liters"], label="Consumed")
plt.xticks(rotation=45)
plt.title("Water Supply vs Consumption")
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(x="area_name", y="complaints_count", data=df)
plt.xticks(rotation=90)
plt.title("Area-wise Complaints")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Prepare features (X) and target (y)
# Drop non-numeric and target-related columns from features
X = df.drop(columns=['date', 'area_name', 'supply_status', 'Water_Risk'])
y = df['Water_Risk']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train a Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

X = df[[
    "population",
    "rainfall_mm",
    "temperature_c",
    "complaints_count",
    "leak_reports"
]]

y = df["Water_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
display(accuracy)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))

import pickle
pickle.dump(model, open("water_risk_model.pkl", "wb"))

!pip install streamlit
import streamlit as st
import pandas as pd
import pickle

df = pd.read_csv("AquaBalance_Coimbatore_Water_Dataset-1.csv")
model = pickle.load(open("water_risk_model.pkl", "rb"))

st.title("💧 Water Issue Reporting System")
