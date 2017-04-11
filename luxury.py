from sklearn import tree
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
dataset=pd.read_csv('luxury.csv')
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
clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
		         algorithm="SAMME",
		         n_estimators=200)
clf.fit(data,labels)
filename='luxury_check.sav';
pickle.dump(clf, open(filename, 'wb'))



