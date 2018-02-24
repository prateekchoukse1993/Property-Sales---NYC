import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import calendar

#Reading data
data = pd.read_csv('nyc-rolling-sales.csv')

#Data cleaning
#Removing bad sale prices
data = data.drop(data[(data['SALE PRICE'] == ' -  ')].index)
data['SALE PRICE'] = data['SALE PRICE'].astype(float)
data = data.drop(data[(data['SALE PRICE'] < 500000)].index)
data = data.drop(data[(data['SALE PRICE'] > data['SALE PRICE'].quantile(q=0.96))].index)

#Removing bad sq ft data
data = data.drop(data[(data['GROSS SQUARE FEET'] == ' -  ') | (data['GROSS SQUARE FEET'] == '0') |
    (data['LAND SQUARE FEET'] == ' -  ') | (data['LAND SQUARE FEET'] == '0')].index)

#Removing bad YEAR BUILT data
#data = data.drop(data[data['YEAR BUILT'] < 1650].index)

#Removing EASE-MENT
data = data.drop('EASE-MENT', axis=1)

#Removing bad TAX CLASS AT PRESENT data
data = data.drop(data[data['TAX CLASS AT PRESENT']==' '].index)
data.loc[data['TAX CLASS AT PRESENT'] == '2A', 'TAX CLASS AT PRESENT']= '2'
data.loc[data['TAX CLASS AT PRESENT'] == '2B', 'TAX CLASS AT PRESENT']= '2'
data.loc[data['TAX CLASS AT PRESENT'] == '2C', 'TAX CLASS AT PRESENT']= '2'

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
data['SALE DATE'] = pd.to_datetime(data['SALE DATE'])

#Plotting Data
#sample = data.sample(frac=0.1)
#sns.distplot(sample['SALE PRICE'], bins=50, hist=True)
#plt.show()

#Creating seperate column for Age
data['Building_Age'] = 2017 - data['YEAR BUILT']


#Outliers Treatment

#data['ZIP CODE'].groupby(data['ZIP CODE']).agg(['count'])

#Writing data to new csv file
#data.to_csv('data_v1.csv', index = False)

#Sale Price Log conversion
data["SALE PRICE"] = np.log1p(data["SALE PRICE"])

#ZIP CODE Binning
data['ZIP_CODE_BIN'] = pd.cut(data['ZIP CODE'],bins=10,labels=['Area_1','Area_2','Area_3','Area_4','Area_5','Area_6','Area_7','Area_8','Area_9','Area_10'])
data['ZIP_CODE_BIN'] = data['ZIP_CODE_BIN'].astype(str)

#Adding Month column
data['SALE MONTH'] = data['SALE DATE'].dt.strftime('%B')
#data['TAX CLASS AT TIME OF SALE'] = data['TAX CLASS AT TIME OF SALE'].astype(str)


#Dummy Variable for Categorical Variables
relevant_columns = ['BOROUGH', 'BUILDING CLASS CATEGORY', 'TOTAL UNITS','GROSS SQUARE FEET',
       'TAX CLASS AT TIME OF SALE',
       'SALE PRICE', 'Building_Age', 'SALE MONTH','ZIP_CODE_BIN']

data = data.loc[:,relevant_columns]

encode_columns = ['BOROUGH','BUILDING CLASS CATEGORY','TAX CLASS AT TIME OF SALE','SALE MONTH','ZIP_CODE_BIN']

#longest_str = max(encode_columns, key=len)
#total_num_unique_categorical = 0
#for feature in encode_columns:
 #   num_unique = len(data[feature].unique())
#    print('{col:<{fill_col}} : {num:d} unique categorical values.'.format(col=feature, 
#                                                                          fill_col=len(longest_str),
#                                                                         num=num_unique))
#    total_num_unique_categorical += num_unique
#print('{total:d} columns will be added.'.format(total=total_num_unique_categorical))

model_data = pd.get_dummies(data[encode_columns])
data = data.drop(encode_columns, axis=1)
model_data = pd.concat([data, model_data], axis=1)
#data.info(verbose=True, memory_usage=True, null_counts=True)

#Split data into training and testing set with 80% of the data going into training
training, testing = train_test_split(model_data, test_size=0.2, random_state=0)
Train = training.loc[:,model_data.columns]
x_Train = Train.drop(['SALE PRICE'], axis=1)
y_Train = Train.loc[:, ['SALE PRICE']]
Test = testing.loc[:,model_data.columns]
x_Test = Test.drop(['SALE PRICE'], axis=1)
y_Test = Test.loc[:, ['SALE PRICE']]

#Linear Regression
results = sm.OLS(y_Train,x_Train.astype(float)).fit()

#Random Forest
#regr_rf = RandomForestRegressor(max_depth=None, random_state=0, n_estimators=50)
#regr_rf.fit(x_Train, y_Train)
#y_pred = regr_rf.predict(x_Test)
#regr_rf.score(x_Train, y_Train, sample_weight=None)
#regr_rf.score(x_Test, y_Test, sample_weight=None)
	   
#Writing data to new csv file
model_data.to_csv('data_v2.csv', index = False)