#!/usr/bin/env python
# coding: utf-8
# ### Audit Risk ML Modelling -
# by E E Ajaegbu
#
# Goal:The goal is to help the auditors by building a classification model
# that can predict the fraudulent firm on the basis of the present and historical risk factors.
#
# Data: Exhaustive one year non-confidential data in the year 2015 to 2016 of firms is collected
# Source: Nishtha Hooda, CSED, TIET, Patiala
# LinK: https://archive.ics.uci.edu/ml/datasets/Audit+Data

# Load the Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

## Title to the website
st.title("Audit Risk ML Application")
st.write("""
A machine learning Application that can predict fraudulent firm on the basis of the present and historical risk factors.
""")
if st.button("Relevant Research"):
    st.write("""
    Hooda, Nishtha, Seema Bawa, and Prashant Singh Rana. 'Fraudulent Firm Classification:
     A Case Study of an External Audit.' Applied Artificial Intelligence 32.1 (2018): 48-64.
     """)

### Application Image
image = Image.open("auditrisk.png")
st.image(image, caption="Audit Risk imagery", use_column_width=True)

# Load the dataset
st.header("Origin Dataset:")
st.write("The Url to the Dataset on UCI https://archive.ics.uci.edu/ml/datasets/Audit+Data")
data = pd.read_csv("audit_risk.csv")
st.dataframe(data.head(10))

# Basic Description - Statistical Summary
st.subheader("Brief Statistical Summary")
st.dataframe(data.describe())

#### Checking for basic information about the data - Missing values etc.
# The format of each variable (displayed via st.write for Streamlit)
st.subheader("Data Info")
st.write(data.info())
st.write("Missing Values:")
st.dataframe(data.isnull().sum())

# Data Preprocessing -
# Dropping Irrelevant variables and making the data suitable for machine learning Modelling
### Looking for Linear Relationships and Correlation - Association among the variables
st.header("Preprocessed and Feature Scaled Data: ")
st.write("""
Irrelevant features were removed and the data was scaled to aid effective machine learning Modelling.
""")

## Selecting variables with strong positive correlation with the Independent variables (Corr > 0.4)
ndata = data.loc[:, ["Score_A", "Score_B", "Score_MV", "District_Loss", "RiSk_E", "Score", "CONTROL_RISK", "Risk"]]
st.dataframe(ndata.head(5))

st.subheader("Heatmap of the Correlation Matrix")
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(ndata.corr(), cmap="YlGn", annot=True)
st.pyplot(fig)

# Independent and Dependent Variables
X = ndata.iloc[:, 0:7].values
y = ndata.iloc[:, 7].values

# Feature Scaling - Standard Normal Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split for better evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Model Selection
st.sidebar.subheader("Best Model")
st.sidebar.write("Logistic Regression, Random Forest, Support Vector Machine and Decision tree were modelled on the Dataset.")
st.sidebar.write("The Support Vector Machine was the best Model with the Highest Average Accuracy score of (96.26%).")

## Models Accuracy (example values; in practice, compute them)
Accuracy = pd.DataFrame([0.957494, 0.962639, 0.958768, 0.954897], columns=["Accuracy"], index=["Logistic Regression", "Support Vector Machine", "Random Forest", "Decision Tree"])
st.sidebar.dataframe(Accuracy)

### SVM Parameters
SVM_par = pd.DataFrame({'C': 38, 'gamma': 'auto', 'kernel': 'linear'}, index=["Parameters"])
st.sidebar.write("SVM Parameters")
st.sidebar.dataframe(SVM_par)

# Cross Validation as confirmation
svmclassifier = SVC(C=38, gamma='auto', kernel='linear')
val_score = cross_val_score(svmclassifier, X_train, y_train, cv=5)
st.sidebar.write("Cross-Validation Scores:", val_score)
st.sidebar.write("Mean CV Score:", val_score.mean())

# Fitting the Model to the Training Data
svmclassifier.fit(X_train, y_train)

# Test Accuracy
test_acc = accuracy_score(y_test, svmclassifier.predict(X_test))
st.sidebar.write(f"Test Accuracy (hold-out set): {test_acc:.4f}")

### Predicting the Risks for the First 10 Cases in our Dataset
actual_value = pd.DataFrame(y[0:10], columns=["Actual Value"])
predicted_value = pd.DataFrame(svmclassifier.predict(X_scaled[0:10, :]), columns=["Predicted Value"])
actual_predicted = pd.concat([actual_value, predicted_value], axis=1)
st.subheader("SVM Model on the Dataset")
st.write("Predicting the Risk of the 10 First Company/cases in the Dataset with SVM model.")
st.write("Actual VS Predicted")
st.dataframe(actual_predicted)

### Prediction
st.sidebar.subheader("Predict If a Firm is Fraudulent:")
st.sidebar.write("Enter Values Below")

def user_input():
    # Use sliders with approximate ranges based on data
    score_a = st.sidebar.slider("Score A", min_value=float(ndata["Score_A"].min()), max_value=float(ndata["Score_A"].max()), value=0.5)
    score_b = st.sidebar.slider("Score B", min_value=float(ndata["Score_B"].min()), max_value=float(ndata["Score_B"].max()), value=0.5)
    score_mv = st.sidebar.slider("Score MV", min_value=float(ndata["Score_MV"].min()), max_value=float(ndata["Score_MV"].max()), value=0.5)
    district_loss = st.sidebar.slider("District Loss", min_value=float(ndata["District_Loss"].min()), max_value=float(ndata["District_Loss"].max()), value=2.0)
    risk_e = st.sidebar.slider("Risk E", min_value=float(ndata["RiSk_E"].min()), max_value=float(ndata["RiSk_E"].max()), value=0.4)
    score = st.sidebar.slider("Score", min_value=float(ndata["Score"].min()), max_value=float(ndata["Score"].max()), value=3.0)
    control_risk = st.sidebar.slider("Control Risk", min_value=float(ndata["CONTROL_RISK"].min()), max_value=float(ndata["CONTROL_RISK"].max()), value=0.4)
    
    ### Dictionary of Input
    input_user = {
        "Score_A": score_a,
        "Score_B": score_b,
        "Score_MV": score_mv,
        "District_Loss": district_loss,
        "RiSk_E": risk_e,
        "Score": score,
        "CONTROL_RISK": control_risk
    }
    
    ### Converting to a DataFrame
    input_user = pd.DataFrame(input_user, index=[0])
    return input_user

input_value = user_input()

### Feature Scaling using the original scaler
input_scaled = scaler.transform(input_value)

if st.sidebar.button("Predict"):
    try:
        Prediction = svmclassifier.predict(input_scaled)[0]
        if Prediction == 1:
            result = pd.DataFrame({"Risk": [Prediction], "Info": ["This Organization is Risky"]})
        else:
            result = pd.DataFrame({"Risk": [Prediction], "Info": ["This Organization is Not Risky"]})
        
        st.write("""
        # The Result of the Classification:
        """)
        st.dataframe(result)
    except Exception as e:
        st.error(f"Prediction Error: {e}")

if st.sidebar.button("Developer"):
    st.sidebar.write("""
    Name: Ebuka E Ajaegbu
    """)
    st.sidebar.write("""
    gmail: ajaegbu35@gmail.com
    """)
    st.sidebar.write("""
    github: https://github.com/EEAjaegbu
    """)
