from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
dataset=pd.read_csv('/home/sukhad/Downloads/Supporting files/Data/derived_features_validated.csv')
data=[]
coloumns=list(dataset)
print coloumns
labels=[]
for index,row in dataset.iterrows():
	list=[]
	for c in coloumns[:11]:
		list.append(row[c])
	c=coloumns[18]
	labels.append(row[c])
	data.append(list)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(data)
data = imp.transform(data)
clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
clf.fit(data,labels)
a=clf.predict(data)


