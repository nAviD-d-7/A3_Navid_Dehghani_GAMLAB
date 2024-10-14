# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 20:06:13 2024

@author: Navid Dehghani

APM:
baratoon taghirate lazem ro anjam dadsam , mesle moratab kardan
taghire hypeparameter ha
kafi hast done done run konid va result begirid va sepas dar enteha dar ghesmate report aval begdi data chi bode , x chi bode y chi bode va hadaf chi bode
har model ch scori dare va behtrin kodome
bad az etmam baraye bande ersal konid
moafagh bashid



"""
#-----------Import Libs----------------------
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

#-----------Import Data----------------------
data = fetch_california_housing()

#-----------STEP1: X , Y----------------------
x = data.data
y = data.target

#-----------STEP1: normalization of data----------------------
scaler = MinMaxScaler() 
x_scaled = scaler.fit_transform(x)


#-----------STEP2:KFOLD CROSS VALIDATION----------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)




#----------MODEL 1 : LR----------------------

linear_model = LinearRegression()
linear_param = {}

GS_linear = GridSearchCV(linear_model, linear_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)

GS_linear.fit(x_scaled, y)

(GS_linear.best_score_)
(GS_linear.best_params_)



#----------MODEL 2 : KNN----------------------
knn_model = KNeighborsRegressor()
knn_param = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

GS_knn = GridSearchCV(knn_model, knn_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)

GS_knn.fit(x_scaled, y)

(GS_knn.best_score_)
(GS_knn.best_params_)



#----------MODEL 3 : DT----------------------

dt_model = DecisionTreeRegressor()
#dt_param = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 10, 20]}
dt_param = {'max_depth': [1,2,3,4,5,6,7,8,9], 'min_samples_split': [2, 10, 20]}
GS_dt = GridSearchCV(dt_model, dt_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)

GS_dt.fit(x_scaled, y)

GS_dt.best_score_
GS_dt.best_params_



#----------MODEL 4 : RF----------------------
rf_model = RandomForestRegressor()
#i CHanged the rf_params and increased max_dsepth ranges
rf_param = {'n_estimators': [50, 100, 200], 'max_depth': [3,4,5,6,7,8,9]}

GS_rf = GridSearchCV(rf_model, rf_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)

GS_rf.fit(x_scaled, y)

GS_rf.best_score_
GS_rf.best_params_


#----------MODEL 4 : SVR----------------------
svr_model = SVR()
#svr_param = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svr_param = {'C': [0.001,0.01,0.1, 1, 10], 'kernel': ['linear', 'rbf','poly'],'degree':[2,3,4]}


GS_svr = GridSearchCV(svr_model, svr_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)

GS_svr.fit(x_scaled, y)

GS_svr.best_score_
GS_svr.best_params_


#==================================
'''
REPORT:









'''
