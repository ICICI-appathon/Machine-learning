
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np


# Reading the given data

# In[3]:

train_data = pd.read_csv('./dataset/train_indessa.csv')
test_data = pd.read_csv('./dataset/test_indessa.csv')
mem_data = pd.read_csv('./dataset/test_indessa.csv')


# In[4]:

print test_data.shape
print train_data.shape


# # Feature Engineering 

# ### For Training Data

# In[5]:

change_to_months = lambda x : x.split(' ')[0]
change_to_weeks = lambda x : x.split(' ')[0]
remove_th = lambda x : x.replace('th' , '')
change_zip_code = lambda x : x.replace('xx' , '')
change_batch_enrolled = lambda x :str(x).replace('BAT' , '')
fill_batch_enrolled = lambda x : str(x).replace(' ','0')
fill_last_week_pay = lambda x : x.replace('NA' , '0')
fill_nan_batch = lambda x : x.replace('nan' , '0')


# In[6]:

# train_data['term'] = train_data['term'].apply(change_to_numeric_months)
train_data['last_week_pay'] = train_data['last_week_pay'].apply(remove_th)
train_data['last_week_pay'] = train_data['last_week_pay'].apply(change_to_weeks)
train_data['zip_code'] = train_data['zip_code'].apply(change_zip_code)
train_data['batch_enrolled'] = train_data['batch_enrolled'].apply(change_batch_enrolled)
train_data['batch_enrolled'] = train_data['batch_enrolled'].apply(fill_batch_enrolled)
train_data['last_week_pay'] = train_data['last_week_pay'].apply(fill_last_week_pay)


# In[7]:

train_data['batch_enrolled'] = train_data['batch_enrolled'].apply(fill_nan_batch)


# Below function can split feature values to dummy values and further concatenate in the data set.

# In[8]:

def getdummiesconcatdropcols(colname , df , check = 'len') :
    if check == 'len' :
        print len(df[colname].unique())
    elif check == 'val' :
        print df[colname].unique()
    else :
        print df[colname].unique()
        df_dummy = pd.get_dummies(df[colname])
        df = pd.concat([df , df_dummy] , axis = 1)
        df = df.drop(colname , axis = 1)
    return df


# In[9]:

dummycols = ['verification_status_joint' , 'term' , 'grade' , 'sub_grade' , 'home_ownership' , 'verification_status' , 'purpose' , 'emp_length' , 'application_type', 'initial_list_status', 'pymnt_plan' , 'addr_state' , 'Not Verified']


# In[12]:

for dc in dummycols :
    try :
        train_data = getdummiesconcatdropcols(dc , train_data , 'r' )
    except :
        pass


# In[13]:

dropcols =['desc'  , 'emp_title' ]
dropcols.append('title')
dropcols.append('n/a')


# In[14]:

train_data = train_data.drop(dropcols , axis = 1)


# In[33]:

print len(train_data.columns)


# In[16]:

train_data = train_data.fillna('0')


# In[17]:

train_data = train_data.drop(['0', 'ANY'], axis = 1)


# In[34]:

print len(train_data.columns)


# In[18]:

train_labels = train_data['loan_status']


# In[19]:

new_train_data = train_data.drop('loan_status' , axis = 1)


# In[20]:

from sklearn.decomposition import PCA


# In[21]:

first_k_data = new_train_data.shape[0]
size = 10000
train = new_train_data[:first_k_data]
# test = test_data[first_k_data:first_k_data+size]
test = new_train_data[first_k_data:first_k_data+size]
labels = train_labels[:first_k_data]
test_labels = train_labels[first_k_data:first_k_data+size]


# In[22]:

print train.shape


# In[ ]:




# In[ ]:




# PCA is used so that dependency of features upon each other can be well analysed.

# In[23]:

pca = PCA()
x = pca.fit_transform(train)
print x.shape
print pca.explained_variance_ratio_


# ### Now For test Data 
# Following the same procedure as done for training data.

# In[26]:

test_data.columns


# In[27]:

# train_data['term'] = train_data['term'].apply(change_to_numeric_months)
test_data['last_week_pay'] = test_data['last_week_pay'].apply(remove_th)
test_data['last_week_pay'] = test_data['last_week_pay'].apply(change_to_weeks)
test_data['zip_code'] = test_data['zip_code'].apply(change_zip_code)
test_data['batch_enrolled'] = test_data['batch_enrolled'].apply(change_batch_enrolled)
test_data['batch_enrolled'] = test_data['batch_enrolled'].apply(fill_batch_enrolled)
test_data['last_week_pay'] = test_data['last_week_pay'].apply(fill_last_week_pay)
test_data['batch_enrolled'] = test_data['batch_enrolled'].apply(fill_nan_batch)


# In[28]:

len(test_data.columns)


# In[29]:

dummycols = [  'verification_status_joint' ,  'term' ,'grade'  , 'sub_grade' , 'home_ownership' , 'verification_status' , 'purpose' , 'emp_length' , 'application_type', 'initial_list_status', 'pymnt_plan' , 'addr_state'  ]


# In[30]:

for dcc in dummycols :
    test_data = getdummiesconcatdropcols( dcc, test_data , 'r' )


# In[31]:

test_data = test_data.drop(dropcols , axis = 1)


# In[32]:

len(test_data.columns)


# Below code can be used to find NaN values or any text in the cell that might be causing any kind of error with sklearn.

# In[ ]:

# for c in test_data.columns :
#     for r in test_data[c] :
#         try :
#             if 'NaN' in r:
#                 print c , r
#                 break
#         except :
#             pass


# Filling NaN values.

# In[35]:

test_data = test_data.fillna('0')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# Using DecisionTreeClassifier for the problem , others like RandomForest and GradientBoosting can be used as well 

# In[ ]:




# In[ ]:




# In[24]:

from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier



# Below code can be used to check that DecisionTreeClassifier gives max accuracy at max_depth = 14 , when the training data is itself split to train and test sets for cross-validation.

# In[ ]:

# for i in range(8,20) :
#     clf = DecisionTreeClassifier(max_depth=i)
#     clf.fit(x , labels)


# In[25]:

clf = DecisionTreeClassifier(max_depth=14)
clf.fit(x , labels)
# pred = clf.predict(test)


# Rather than direct classification , we give the prediction probability 

# In[36]:

finalpredprob = clf.predict_proba(test_data)


# In[37]:

test_data.shape


# In[38]:

finalpredprob.shape


# In[40]:

sol = {}
sol['member_id'] = mem_data['member_id']
sol['loan_status'] = finalpredprob[:,1]
ans = pd.DataFrame(sol)


# Epsilon is added/subtracted so that the probability is never exactly 0.0 or 1.0 

# In[46]:

epsilon = 0.000099
ansloan = []
for x in ans['loan_status'] :
    if x ==1.0 or x == 1 :
        x = x - epsilon
    elif x == 0.0 or x == 0 :
        x = x + epsilon
    else:
        pass 
    ansloan.append(x)
ans['loan_status'] = ansloan[:]   


# In[47]:

ans = ans[['member_id' , 'loan_status']]
print ans[:5]


# OUTPUT csv file 

# In[ ]:

ans.to_csv('outputans.csv' , index = False )

