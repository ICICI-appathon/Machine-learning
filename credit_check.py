import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
def credit_fraud() :
	filename='credit_fraud.sav'
	X_test=[406	,-2.3122265423	,1.9519920106	,-1.6098507323	,3.9979055875	,-0.5221878647	,-1.4265453192,	-2.5373873062,	1.3916572483,	-2.7700892772,	-2.7722721447,	3.2020332071,	-2.8999073885,	-0.5952218813,	-4.2892537824,	0.3897241203,	-1.1407471798,	-2.8300556745,	-0.0168224682,	0.416955705,	0.1269105591,	0.5172323709,	-0.0350493686,	-0.4652110762,	0.3201981985,	0.0445191675,	0.1778397983,	0.2611450026,	-0.1432758747,	0
	]
	Y_test=[]
	loaded_model = pickle.load(open(filename, 'rb'))
	Y_test = loaded_model.predict(X_test)
	print Y_test
credit_fraud()