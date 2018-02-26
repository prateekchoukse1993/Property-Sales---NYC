import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

#Reading data
data = pd.read_csv('nyc-rolling-sales.csv')

#Data cleaning
#Removing bad sale prices
data = data.drop(data[(data['SALE PRICE'] == ' -  ')].index)
data['SALE PRICE'] = data['SALE PRICE'].astype(float)
#16 percentile is used because values before 16 is 100 or less than hundred
data = data.drop(data[(data['SALE PRICE'] > data['SALE PRICE'].quantile(q=0.16))].index)
#Need to discuss on this in team meeting
#data = data.drop(data[(data['SALE PRICE'] < data['SALE PRICE'].quantile(q=0.96))].index)

#Removing bad sq ft data
data = data.drop(data[(data['GROSS SQUARE FEET'] == ' -  ') | (data['GROSS SQUARE FEET'] == '0') |
    (data['LAND SQUARE FEET'] == ' -  ') | (data['LAND SQUARE FEET'] == '0')].index)

#Removing bad YEAR BUILT data
#data = data.drop(data[data['YEAR BUILT'] < 1650].index)

#Removing EASE-MENT
data = data.drop('EASE-MENT', axis=1)

#Removing bad TAX CLASS AT PRESENT data
data = data.drop(data[data['TAX CLASS AT PRESENT']==' '].index)

#Removing bad ZIP code data
data = data.drop(data[data['ZIP CODE']==0].index)

#Removing Unnamed: 0
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

#Removing duplicate rows
columns = data.columns
data = data.drop_duplicates(columns, keep='last')

#BOROUGH Decoding
data.loc[data['BOROUGH'] == 1, 'BOROUGH']= 'Manhatten'
data.loc[data['BOROUGH'] == 2, 'BOROUGH']= 'Bronx'
data.loc[data['BOROUGH'] == 3, 'BOROUGH']= 'Brooklyn'
data.loc[data['BOROUGH'] == 4, 'BOROUGH']= 'Queens'
data.loc[data['BOROUGH'] == 5, 'BOROUGH']='Staten Island'

#Datatype Transformation
data['BLOCK'] = data['BLOCK'].astype(str)
data['LOT'] = data['LOT'].astype(str)
data['ZIP CODE'] = data['ZIP CODE'].astype(int)
data['LAND SQUARE FEET'] = data['LAND SQUARE FEET'].astype(float)
data['GROSS SQUARE FEET'] = data['GROSS SQUARE FEET'].astype(float)
data['TAX CLASS AT TIME OF SALE'] = data['TAX CLASS AT TIME OF SALE'].astype(str)
data['TAX CLASS AT PRESENT'] = data['TAX CLASS AT PRESENT'].astype(str)
data['SALE PRICE'] = data['SALE PRICE'].astype(float)

#Adding Month column
data['SALE DATE'] = pd.to_datetime(data['SALE DATE'])
data['SALE MONTH'] = data['SALE DATE'].dt.strftime('%B')
#ZIP CODE Binning
data['ZIP_CODE_BIN'] = pd.cut(data['ZIP CODE'],bins=10,labels=['Area_1','Area_2','Area_3','Area_4','Area_5','Area_6','Area_7','Area_8','Area_9','Area_10'])
data['ZIP_CODE_BIN'] = data['ZIP_CODE_BIN'].astype(str)
#Creating seperate column for Age
data['Building_Age'] = 2017 - data['YEAR BUILT']


relevant_columns = ['BOROUGH', 'BUILDING CLASS CATEGORY', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS',
		'TOTAL UNITS','GROSS SQUARE FEET', 
		'TAX CLASS AT TIME OF SALE',
		'SALE PRICE', 'Building_Age', 'SALE MONTH','ZIP_CODE_BIN']

encode_columns = ['BUILDING CLASS CATEGORY','SALE MONTH','ZIP_CODE_BIN']

fewer_columns = ['BOROUGH', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS',
		'TOTAL UNITS','GROSS SQUARE FEET', 
		'TAX CLASS AT TIME OF SALE',
		'SALE PRICE', 'Building_Age', 'SALE MONTH']

relevant_columns = fewer_columns


data = data[relevant_columns]

data[data["SALE MONTH"] == "January"] = 1
data[data["SALE MONTH"] == "February"] = 2
data[data["SALE MONTH"] == "March"] = 3
data[data["SALE MONTH"] == "April"] = 4
data[data["SALE MONTH"] == "May"] = 5
data[data["SALE MONTH"] == "June"] = 6
data[data["SALE MONTH"] == "July"] = 7
data[data["SALE MONTH"] == "August"] = 8
data[data["SALE MONTH"] == "September"] = 9
data[data["SALE MONTH"] == "October"] = 10
data[data["SALE MONTH"] == "November"] = 11
data[data["SALE MONTH"] == "December"] = 12


model_data = data
#model_data = pd.get_dummies(data[encode_columns])
#data = data.drop(encode_columns, axis=1)
#model_data = pd.concat([data, model_data], axis=1)

cols = model_data.columns.values
scaler = MinMaxScaler()
model_data = scaler.fit_transform(model_data)
model_data = pd.DataFrame(model_data, columns=cols)
print(type(model_data))

y = model_data["SALE PRICE"]
X = model_data.drop(["SALE PRICE"], axis=1)


#data = data["SALE DATE"].to_datetime()
#print(data["SALE DATE"][0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print(X.columns.values)
print(len(X))
print(len(y))

print("Support Vector Regression - RBF Kernel, Metric: R^2 ")
clf = svm.SVR(kernel='rbf', C=.1, gamma=1000, max_iter=5000).fit(X, y)
print(clf.score(X,y))

print("K=5 Cross Validation")
scores = cross_val_score(clf, X, y, cv=5)
print(scores)


print("Support Vector Regression - Linear, Metric: R^2 ")
clf = svm.SVR(kernel='linear', C=.1, gamma=1000, max_iter=5000).fit(X, y)
print(clf.score(X,y))

print("K=5 Cross Validation")
scores = cross_val_score(clf, X, y, cv=5)
print(scores)