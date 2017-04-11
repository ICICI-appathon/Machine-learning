import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
def luxuryr() :
	filename='luxury_check.sav'
	clf = pickle.load(open(filename, 'rb'))
	data1=[0.986021972,0.518867925,0.963674868,0.743931132,0.613207547,0.905005093]
	a=clf.predict(data1)
	return a
