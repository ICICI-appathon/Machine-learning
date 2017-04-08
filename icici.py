from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
dataset=pd.read_csv('necessity.csv')
coloumns=list(dataset)
def necc():
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

	data1=[0.841987824,0.804347826,0.9074916,0.874692437,0.52173913,0.925664503]
	a=clf.predict(data1)
	print a;
	return a
necc()


