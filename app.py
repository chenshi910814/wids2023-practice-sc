import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv("/Users/shichen/Dropbox/UCSC Extension/Object-oriented design/widsdatathon2023/train_data.csv")

# Dropping index column and filling missing values in the training dataset using bfill method
# There are no missing values in test dataset
data.drop('index',axis=1,inplace=True)
data.fillna(inplace=True, method='bfill')

# Preprocess the dataset
X = data[['contest-wind-h500-14d__wind-hgt-500', 'contest-slp-14d__slp', 'nmme-tmp2m-34w__gfdlflorb', 'contest-prwtr-eatm-14d__prwtr']]
y = data['contest-tmp2m-14d__tmp2m']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Streamlit app
#st.title("XGBoost Prediction Model")
st.markdown("<h1 style='text-align: center; color: white;'>XGBoost Weather Prediction Model</h1>", unsafe_allow_html=True)

# Model description
model_description = '''
This model predicts the target column 'contest-tmp2m-14d__tmp2m' using XGBoost, a popular machine learning algorithm. 
The model uses the following five input features:
1. contest-wind-h500-14d__wind-hgt-500
2. contest-slp-14d__slp
3. nmme-tmp2m-34w__gfdlflorb
4. contest-pres-sfc-gauss-14d__pre
5. contest-prwtr-eatm-14d__prwtr
'''
st.markdown(model_description)


# Evaluate the model
st.subheader("Model Evaluation")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write("Mean Squared Error:", mse)
st.write("R-squared Score:", r2)

# User input
# User input
st.sidebar.header("Input Features")
input_features = [
    st.sidebar.slider("geopotential height at 500 millibars", float(X['contest-wind-h500-14d__wind-hgt-500'].min()), float(X['contest-wind-h500-14d__wind-hgt-500'].max()), step=0.1, value=float(X['contest-wind-h500-14d__wind-hgt-500'].mean())),
    st.sidebar.slider("sea level pressure (slp)", float(X['contest-slp-14d__slp'].min()), float(X['contest-slp-14d__slp'].max()), step=0.1, value=float(X['contest-slp-14d__slp'].mean())),
    st.sidebar.slider("weeks 3-4 weighted average of most recent monthly NMME model forecasts", float(X['nmme-tmp2m-34w__gfdlflorb'].min()), float(X['nmme-tmp2m-34w__gfdlflorb'].max()), step=0.1, value=float(X['nmme-tmp2m-34w__gfdlflorb'].mean())),
    st.sidebar.slider("precipitable water for entire atmosphere", float(X['contest-prwtr-eatm-14d__prwtr'].min()), float(X['contest-prwtr-eatm-14d__prwtr'].max()), step=0.1, value=float(X['contest-prwtr-eatm-14d__prwtr'].mean()))
]

# Predict and display the result
st.subheader("Model Prediction")
input_data = pd.DataFrame([input_features], columns=X.columns)
prediction = model.predict(input_data)
st.write("Predicted Temperature:", prediction[0])


