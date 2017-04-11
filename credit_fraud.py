import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


df_credit = pd.read_csv('creditcard.csv',header=0)
df_credit
x =df_credit.drop(['Class'], axis=1)
y = df_credit['Class']

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=0)
Kfold =KFold(len(df_credit),n_folds=100, shuffle=False)

#using the gradient boosting algorithm
GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1,random_state=0)
GBC = GBC.fit(x,y)
filename = 'credit_fraud.sav'
pickle.dump(GBC, open(filename, 'wb'))

