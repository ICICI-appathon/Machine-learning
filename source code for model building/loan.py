import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
encode_features = ['Gender','Married','Education','Self_Employed','Dependents','Loan_Status']
fillna_withmean = ['LoanAmount','Loan_Amount_Term']
fillna_withmostcommon = ['Dependents','Gender','Credit_History','Married','Self_Employed']
    
def transform_df(data):
    
    #Removing Loans_ID
    df = data #.drop('Loan_ID',axis=1) 
    
    # Filling NaN values 
    for feature in fillna_withmean:
        if feature in data.columns.values:
            df[feature] = df[feature].fillna(df[feature].mean()) 
        
    for feature in fillna_withmostcommon:
        if feature in data.columns.values:
            df[feature] = df[feature].fillna(df[feature].value_counts().index[0])
    
    # Encoding Features
    for feature in encode_features:
        if feature in data.columns.values:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
    
    # Adding Applicant and Coapplicant Incomes as Household
    df['Household_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df = df.drop(['ApplicantIncome','CoapplicantIncome'],axis=1)
     
    # Transforming some other values   
    dummies = pd.get_dummies(df.Property_Area)
    df = pd.concat([df,dummies],axis=1)
    df = df.drop('Property_Area',axis=1)

    
    return df

train_df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv',index_col=0)
test_df=pd.read_csv('train_u6lujuX_CVtuZ9i.csv',index_col=0)
col_names = train_df.columns.tolist()
train_df = transform_df(train_df)
test_df = transform_df(test_df)
train_df.insert(len(train_df.columns)-1,'Loan_Status',train_df.pop('Loan_Status'))
scale_features = ['LoanAmount','Loan_Amount_Term','Household_Income']
train_df[scale_features] = train_df[scale_features].apply(lambda x:(x.astype(int) - min(x))/(max(x)-min(x)), axis = 0)
test_df[scale_features] = test_df[scale_features].apply(lambda x:(x.astype(int) - min(x))/(max(x)-min(x)), axis = 0)
X_train = train_df.iloc[:, :-1]
Y_train = train_df.iloc[:,-1]
clf = linear_model.BayesianRidge()
clf = clf.fit(X_train, Y_train)
filename='loan.sav'
pickle.dump(clf,open(filename,'wb'))

    
