import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
def foodr() :
	filename='food_checkr.sav'
	clf = pickle.load(open(filename, 'rb'))
	data1=[0.98978884,0.52,0.848129517,0.920248868,0.64,0.808885879]
	a=clf.predict(data1)
	print a*100
	return a
foodr()
