# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 20:06:13 2024

@author: Navid Dehghani
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler


data = fetch_california_housing()
x = data.data
y = data.target


scaler = MinMaxScaler() 
x_scaled = scaler.fit_transform(x)


kf = KFold(n_splits=5, shuffle=True, random_state=42)





linear_model = LinearRegression()
linear_param = {}

GS_linear = GridSearchCV(linear_model, linear_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)

GS_linear.fit(x_scaled, y)

(GS_linear.best_score_)
(GS_linear.best_params_)




knn_model = KNeighborsRegressor()
knn_param = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

GS_knn = GridSearchCV(knn_model, knn_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)

GS_knn.fit(x_scaled, y)

(GS_knn.best_score_)
(GS_knn.best_params_)




dt_model = DecisionTreeRegressor()
dt_param = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 10, 20]}

GS_dt = GridSearchCV(dt_model, dt_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)

GS_dt.fit(x_scaled, y)

GS_dt.best_score_
GS_dt.best_params_




rf_model = RandomForestRegressor()
rf_param = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}

GS_rf = GridSearchCV(rf_model, rf_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)

GS_rf.fit(x_scaled, y)

GS_rf.best_score_
GS_rf.best_params_



svr_model = SVR()
svr_param = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

GS_svr = GridSearchCV(svr_model, svr_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)

GS_svr.fit(x_scaled, y)

GS_svr.best_score_
GS_svr.best_params_

