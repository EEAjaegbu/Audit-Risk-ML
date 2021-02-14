#!/usr/bin/env python
# coding: utf-8

# ### Audit Risk ML Modelling -
# by E E Ajaegbu 
# 
# Goal:The goal is to help the auditors by building a classification model
#  that can predict the fraudulent firm on the basis of the present and historical risk factors.
#  
# Data: Exhaustive one year non-confidential data in the year 2015 to 2016 of firms is collected 
# Source: Nishtha Hooda, CSED, TIET, Patiala
# LinK: https://archive.ics.uci.edu/ml/datasets/Audit+Data

# #### Load the Libraries 

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
import streamlit as st
from PIL import Image


## Tittle to the website
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
image= Image.open("auditrisk.png")
st.image(image, caption =" Audit Risk imagery",use_column_width=True)


# #### Load the dataset
st.header("Origin Dataset:")
st.write("The Url to the Dataset on UCI https://archive.ics.uci.edu/ml/datasets/Audit+Data")
# In[3]:

data =  pd.read_csv("audit_risk.csv")
st.dataframe(data.head(10))

# Basic Description- Statictical Summary
st.subheader(" Brief Statistical Summary")
st.dataframe(data.describe())


# In[4]:
#### Checking for basic information about the data- Missing values etc.

print(data.info()) # The format of each varaibles 
data.isnull().sum() # Checking for missing values 


# #### Data Preprocessing -
# Dropping Irrelevant varaibles and making the data suitable for machine learning Modelling 

### Looking for Linear Releationships and Correlation- Association among the variables 
data.corr()

st.header("Preprocessed  and Feature Scaled Data: ")
st.write("""
Irrelevant features was removed and the data was scaled to aid effective machine learning Modelling.
""")
## Selecting variables with strong postive correlation with the Independent varaibles(Corr > 0.4)

print([data.corr().iloc[:,25:] > 0.4])
ndata = data.loc[:,["Score_A","Score_B","Score_MV","District_Loss","RiSk_E","Score","CONTROL_RISK","Risk"]]
st.dataframe(ndata.head(5))

st.subheader("Heatmap of the Correlation Matrix")
fig, ax = plt.subplots(figsize=(12,5))
sns.heatmap(ndata.corr(),cmap="YlGn", annot=True)
#plt.title("Heatmap of the Correlation")
st.pyplot(fig)


# In[10]
# Model Selection
st.sidebar.subheader("Best Model")
st.sidebar.write("Logistic Regression, Random Forest, Support Vector Machine and  Decision tree was modelled on the Dataset.")
st.sidebar.write("The Support Vector Machine was the best Model with the Highest Average Accuracy score of (96.26%).")

## Models Accuracy  
Accuracy = pd.DataFrame([0.957494,0.962639,0.958768,0.954897],columns=["Accuracy"],index=["Logistic Regression","Support Vector Machine","Random Forest","Decision Tree"])
st.sidebar.dataframe(Accuracy)
# In[12]:

### SVM Parameters
SVM_par = pd.DataFrame({'C': 38, 'gamma': 'auto', 'kernel': 'linear'}, index=["Parameters"])
st.sidebar.write("SVM Parameters")
st.sidebar.dataframe(SVM_par)

### Independent and Dependent Variables
X = ndata.iloc[:,0:7].values
y = ndata.iloc[:,7].values

# In[13]:


## Feature Scaling -  Standard Noramal Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score



# ####  Cross Validation
# As confirmation for the GridSerach CV.
# In[20]:

svmclassifier = SVC( C=  38, gamma='auto', kernel=  'linear')
val_score= cross_val_score(svmclassifier, X,y, cv=5)
print(val_score)
val_score.mean()


# In[21]:


## Fitting the Model to the Data
svmclassifier = SVC( C=  38, gamma='auto', kernel=  'linear')
svmclassifier.fit(X,y)


# #### Predictions using the Model

# In[22]:


input_val = np.array([ 1.42984618, -0.6667522 , -0.56989549, -0.41140172, -0.41041721,-0.35250258,0.543289])
input_val = input_val.reshape(1,-1)

svmclassifier.predict(input_val)


# In[23]:


svmclassifier.predict(X[0:10,:])


# In[27]:


### Predicitng the Risks for the First 10 Cases in our Dataset
actual_value = pd.DataFrame(y[0:10],columns=["Actual Value"])
predicted_value= pd.DataFrame(svmclassifier.predict(X[0:10,:]),columns=["Predicted Value"])
actual_predicted = pd.concat([actual_value,predicted_value],axis=1)

st.subheader("SVM Model on the Dataset")
st.write("Predicting the Risk of the 10 First Company/cases  in the Dataset with SVM model.")
st.write("Actual VS Predicted")
st.dataframe(actual_predicted.transpose())


### Prediction 
st.sidebar.subheader("Predict If a Firm is Fradulent:")
st.sidebar.write("Enter Values Below")

def user_input():
    score_a = st.sidebar.number_input("Score A",min_value=0.00, max_value= 0.99,value=0.5)
    score_b = st.sidebar.number_input("Score B",min_value=0.00, max_value= 0.99,value=0.5)
    score_mv = st.sidebar.number_input("Score MV",min_value=0.00, max_value= 0.99,value=0.5)
    district_loss = st.sidebar.number_input("District Loss",min_value=1, max_value=10,value=5)
    risk_e = st.sidebar.number_input("Risk E",min_value=0.00, max_value= 3.00,value=1.50)
    score = st.sidebar.number_input("Score",min_value=0.00, max_value= 6.00,value=3.50)
    control_risk = st.sidebar.number_input("Control Risk",min_value=0.00, max_value= 6.00,value=2.50)
        
    ### Dictionaries of Input
    input_user= {"Score A":score_a ,"Score B":score_b,"Score MV":score_mv,"District Loss":district_loss,"Risk E":risk_e,"Score":score, "Control Risk":control_risk}
               
    ### Converting to a Dataframes
    input_user =pd.DataFrame(input_user,index=[0])
    return input_user

input_value = user_input() 
print(input_value)
### Feature Scaling

scaler_input = StandardScaler()
input_value = scaler_input.fit_transform(input_value)

if st.sidebar.button("Predict"):
    Prediction = svmclassifier.predict(input_value.reshape(1,-1))
    if Prediction == 1:
        result = pd.DataFrame({"Risk":Prediction,"Info":"This  Organization is Risky"})
    else:
        result = pd.DataFrame({"Risk":Prediction,"Info":"This  Organization is Not Risky"})                      
    
    st.write("""
    # The Result of the Classification:
    """) 
    st.dataframe(result)
                                


if st.sidebar.button("Developer"):
    st.sidebar.write("""
    Name:Ebuka E Ajaegbu
    """)
    st.sidebar.write("""
    gmail: ajaegbu35@gmail.com
    """)
    st.sidebar.write("""
    github: https://github.com/EEAjaegbu
    """)
