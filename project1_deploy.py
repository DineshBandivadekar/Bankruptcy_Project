#-*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
import pickle

st.title('Model Deployment: Logistic Regression')
st.sidebar.header('User Input Parameters')

def user_input_features():
    industrial_risk=st.sidebar.selectbox('industrial_risk',('0','0.5','1'))
    management_risk=st.sidebar.selectbox('management_risk',('0','0.5','1'))
    financial_flexibility=st.sidebar.selectbox('financial_flexibility',('0','0.5','1'))
    credibility=st.sidebar.selectbox('credibility',('0','0.5','1'))
    competitiveness=st.sidebar.selectbox('competitiveness',('0','0.5','1'))
    operating_risk= st.sidebar.selectbox('operating_risk',('0','0.5','1'))
    data = {'industrial_risk':industrial_risk,
              'management_risk':management_risk,
              'financial_flexibility':financial_flexibility,
              'credibility':credibility,
              'competitiveness':competitiveness,
              'operating_risk':operating_risk  }

    features=pd.DataFrame(data,index=[0])
    return features

df=user_input_features()
st.subheader('User Input parameters')
st.write(df)

with open(r'C:\\Users\\Admin\\Excel_R\\Bankruptcy_project\\Logistic_regression.pkl','rb') as file:
    LogisticRegression_model = pickle.load(file)
prediction = LogisticRegression_model.predict(df)
prediction_proba = LogisticRegression_model.predict_proba(df)


st.subheader('Predicted Result')
st.write('Bankruptcy Yes' if prediction_proba[0][1]>0.5 else 'Bankruptcy No')

st.subheader('Prediction Probability')
st.write(round(prediction_proba [0][1],4))