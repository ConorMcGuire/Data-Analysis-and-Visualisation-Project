# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:42:14 2023

@author: Conor McGuire
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


import os
os.chdir('C://Users//Conor College//OneDrive - Dundalk Institute of Technology//Documents//4th Year//4th Year Semester 2//Data Analysis and Visualisation//data-master//comic-characters')

data = pd.read_csv('marvel-wikia-data.csv')

data.head()

data.info()
# #   Column            Non-Null Count  Dtype  
#---  ------            --------------  -----  
# 0   page_id           16376 non-null  int64  
# 1   name              16376 non-null  object 
# 2   urlslug           16376 non-null  object 
# 3   ID                12606 non-null  object 
# 4   ALIGN             13564 non-null  object 
# 5   EYE               6609 non-null   object 
# 6   HAIR              12112 non-null  object 
# 7   SEX               15522 non-null  object 
# 8   GSM               90 non-null     object 
# 9   ALIVE             16373 non-null  object 
# 10  APPEARANCES       15280 non-null  float64
# 11  FIRST APPEARANCE  15561 non-null  object 
# 12  Year              15561 non-null  float64

data.describe()
#             page_id   APPEARANCES          Year
#count   16376.000000  15280.000000  15561.000000
#mean   300232.082377     17.033377   1984.951803
#std    253460.403399     96.372959     19.663571
#min      1025.000000      1.000000   1939.000000
#25%     28309.500000      1.000000   1974.000000
#50%    282578.000000      3.000000   1990.000000
#75%    509077.000000      8.000000   2000.000000
#max    755278.000000   4043.000000   2013.000000

#Check data clean
data.isnull().sum() 
#page_id                 0
#name                    0
#urlslug                 0
#ID                   3770
#ALIGN                2812
#EYE                  9767
#HAIR                 4264
#SEX                   854
#GSM                 16286
#ALIVE                   3
#APPEARANCES          1096
#FIRST APPEARANCE      815
#Year                  815
#dtype: int64

#data['id'].unique()

# We can see from this that there are no data for name, page id or urlslug.
# ID has 3770 null values. This column refers to whether a character has a public or secret identity. If they have no aliases or other identities they get the no dual identity value. I think for the purposes of this data we can replace all blanks here with "no dual identity" as it is effectively a "default" value.
# Similarly for ALIGN I think it is ok to default to neutral for missing values.
# eye and hair colour are not important for now so we can ignore them
# sex is one of the key values I plan use to compare on. For this reason I will remove all null values.
# gsm = gender or sexual minority. All non-gsm rows contain an empty cell here. 

#2. DATA CLEANING ########################################################

data.drop('urlslug', axis = 1, inplace = True)
data.drop('page_id', axis = 1, inplace = True)

data = data.rename(columns={
    'ID':'id',
    'ALIGN':'alignment',
    'EYE':'eye',
    'HAIR':'hair',
    'SEX':'sex',
    'GSM':'gsm',
    'ALIVE':'alive',
    'APPEARANCES':'appearances',
    'FIRST APPEARANCE':'first_appearance',
    'Year':'year'})


data.drop('first_appearance', axis = 1, inplace = True)

# 50% of the remaining rows do not have an eye colour. Trying to remove these rows would cut the dataset by too much and the eye colour is not as important 

data.drop('eye', axis = 1, inplace = True)
data.drop('hair', axis = 1, inplace = True)

# delete rows with more than 4 missing values
missing_half = data.isnull().sum(axis=1) > 4

data.drop(data[missing_half].index, inplace = True)

#Drop null values in sex column
data = data.drop(data[data.sex.isnull()].index)
data['sex'].isnull().sum()  # == 0 so we know the previous line worked

data = data.drop(data[data.year.isnull()].index)
data['year'].isnull().sum()  # == 0 so we know the previous line worked

data = data.drop(data[data.appearances.isnull()].index)
data['appearances'].isnull().sum()  # == 0 so we know the previous line worked

# Many of the gsm rows are empty for characters without any minority status so I will replace the values with a 1 if the character has minority status and a 0 if they do not

data['gsm'].fillna("none", inplace = True)
data['gsm']=np.where(data.gsm =="none",0,1)

data.isnull().sum() 


data['gsm'].unique()
data['id'].describe()
data['id'].value_counts()

# Total 11019 rows remaining


figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(data.corr(), annot=True, cmap = 'Reds')
plt.show()














# =============================================================================
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# 
# 
# # split the data into training and testing sets based on null values in the 'identity' or 'alignment' columns
# test = data[(data['id'].isnull()) | (data['alignment'].isnull())]
# train = data[(~data['id'].isnull()) & (~data['alignment'].isnull())]
# 
# # preprocess the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(train.drop(['id', 'alignment'], axis=1))
# y_train = train[['id', 'alignment']].values
# X_test = scaler.transform(test.drop(['id', 'alignment'], axis=1))
# 
# # train the model
# model = LogisticRegression()
# model.fit(X_train, y_train)
# 
# # predict the missing values
# y_pred = model.predict(X_test)
# 
# # replace the null values with the predicted values
# test[['identity', 'alignment']] = pd.DataFrame(y_pred, columns=['identity', 'alignment'])
# data_imputed = pd.concat([train, test])
# 
# # save the imputed dataset
# data_imputed.to_csv('dataset_imputed.csv', index=False)
# =============================================================================

###############CLASSIFICATION MODELLING######################################

#########Modelling - Step 1: Split Data into Train and Test

# =============================================================================
# #Set the Response and the predictor variables
# 
# x = data[['sex', 'gsm', 'alive', 'appearances', 'year']] #pandas dataframe
# y = data['id'] #Pandas series
# 
# #Splitting the Data Set into Training Data and Test Data
# from sklearn.model_selection import train_test_split
# 
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# 
# 
# 
# data['sex'].unique()
# data['sex_male']=np.where(data.sex =="Male Characters",1,0)
# data['sex_female']=np.where(data.sex =="Female Characters",1,0)
# 
# data['alive'].unique()
# data['living']=np.where(data.sex =="Living Characters",1,0)
# 
# 
# test = data[(data['id'].isnull()) | (data['alignment'].isnull())]
# train = data[(~data['id'].isnull()) & (~data['alignment'].isnull())]
# 
# test.drop('name', axis = 1, inplace = True)
# train.drop('name', axis = 1, inplace = True)
# test.drop('sex', axis = 1, inplace = True)
# train.drop('sex', axis = 1, inplace = True)
# test.drop('alive', axis = 1, inplace = True)
# train.drop('alive', axis = 1, inplace = True)
# 
# test_id = test[(test['id'].isnull())]
# test_alignment = test[(test['alignment'].isnull())]
# 
# 
# # print the shapes of the training and testing sets
# print("Training set shape:", train.shape)
# print("Testing set shape:", test.shape)
# 
# test.isnull().sum() 
# train.isnull().sum() 
# 
# 
# scaler = StandardScaler()
# X_train = scaler.fit_transform(train.drop(['id', 'alignment'], axis=1))
# y_id_train = train['id'].values
# y_alignment_train = train['alignment'].values
# X_test = scaler.transform(test.drop(['id', 'alignment'], axis=1))
# 
# # train the model
# model = LogisticRegression()
# model.fit(X_train, y_id_train)
# 
# # test the model
# y_pred = model.predict(X_test)
# 
# y_pred['0'].unique()
# 
# 
# model.fit(X_train, y_alignment_train)
# 
# y_pred = model.predict(X_test)
# 
# 
# 
# # replace the null values with the predicted values
# test['id'] = pd.DataFrame(y_pred, columns=['id'])
# 
# # concatenate the training and testing sets back together
# data_imputed = pd.concat([train, test])
# 
# # save the imputed dataset
# data_imputed.to_csv('dataset_imputed.csv', index=False)
# 
# 
# 
# 
# 
# 
# 
# 
# data.to_csv('data_halfclean.csv', index=False)
# =============================================================================



# Unfortunately  I could not get the classification model working in time. It was predicting the values but would not merge backwith the main dataset
# This meant I had to drop the missing data.

data = data.drop(data[data.alignment.isnull()].index)
data['alignment'].isnull().sum()  # == 0 so we know the previous line worked

data = data.drop(data[data.id.isnull()].index)
data['id'].isnull().sum()  # == 0 so we know the previous line worked




data.to_csv('comic_marvel_final.csv', index=False)































































