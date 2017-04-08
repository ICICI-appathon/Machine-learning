from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
dataset=pd.read_csv('luxury.csv')
coloumns=list(dataset)
def luxury():
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

	data1=[0.986021972,0.518867925,0.963674868,0.743931132,0.613207547,0.905005093]
	a=clf.predict(data1)
	return a
luxury()


