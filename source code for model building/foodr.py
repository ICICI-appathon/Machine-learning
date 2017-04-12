from sklearn import tree
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import Imputer
dataset=pd.read_csv('food.csv')
coloumns=list(dataset)
labels=[]
data=[]
for index,row in dataset.iterrows():
	list=[]
	for c in coloumns[:6]:
		list.append(row[c])
	c=coloumns[-1]
	labels.append(row[c])
	data.append(list)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(data)
data = imp.transform(data)
clf=linear_model.BayesianRidge()
clf.fit(data,labels)
filename='food_checkr.sav';
pickle.dump(clf, open(filename, 'wb'))

	

