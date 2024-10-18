
"""
Created on Sun Oct 13 20:06:13 2024
@author: Navid Dehghani

"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import pandas as pd  
import matplotlib.pyplot as plt

data = fetch_california_housing()

x = data.data
y = data.target

scaler = MinMaxScaler() 
x_scaled = scaler.fit_transform(x)

df = pd.DataFrame(data.data, columns=data.feature_names)
 
df['Price'] = y

sorted_df = df.sort_values(by='Price', ascending=False)

kf = KFold(n_splits=15, shuffle=True, random_state=42)


linear_model = LinearRegression()
linear_param = {
}

GS_linear = GridSearchCV(linear_model, linear_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
GS_linear.fit(x_scaled, y)


print('Linear Regression best score:', GS_linear.best_score_)
print('Linear Regression best params:', GS_linear.best_params_)


knn_model = KNeighborsRegressor()
knn_param = {
    'n_neighbors': [10, 15, 20, 25],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
    'p': [1, 2] 
}

GS_knn = GridSearchCV(knn_model, knn_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
GS_knn.fit(x_scaled, y)


print('KNN best score:', GS_knn.best_score_)
print('KNN best params:', GS_knn.best_params_)


dt_model = DecisionTreeRegressor()
dt_param = {
    'max_depth': [5, 10, 25, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 6, 10, 16]
}

GS_dt = GridSearchCV(dt_model, dt_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
GS_dt.fit(x_scaled, y)


print('Decision Tree best score:', GS_dt.best_score_)
print('Decision Tree best params:', GS_dt.best_params_)


rf_model = RandomForestRegressor()
rf_param = {'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5, 6, 7, 8, 9],
            'min_samples_split': [2, 5, 10]
}

GS_rf = GridSearchCV(rf_model, rf_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
GS_rf.fit(x_scaled, y)


print('Random Forest best score:', GS_rf.best_score_)
print('Random Forest best params:', GS_rf.best_params_)


svr_model = SVR()
svr_param = {'C': [0.01, 0.1, 1, 10],
             'kernel': ['linear', 'rbf', 'poly'],
             'degree': [2, 3, 4]
}

GS_svr = GridSearchCV(svr_model, svr_param, cv=kf, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
GS_svr.fit(x_scaled, y)


 
print('SVR best score:', GS_svr.best_score_)
print('SVR best params:', GS_svr.best_params_)


scores = {
    'Linear Regression': float(GS_linear.best_score_),
    'KNN': float(GS_knn.best_score_),
    'Decision Tree': float(GS_dt.best_score_),
    'Random Forest': float(GS_rf.best_score_),
    'SVR': float(GS_svr.best_score_)
}


accuracies = {model: (100 + score * 100) for model, score in scores.items()}
for model, accuracy in accuracies.items():
   print(model,'accuracy :', round(accuracy,2),'%')
   
   
best_model = max(scores, key=lambda k:(scores[k]))
print('Best Mmodel is :' , best_model , 'whit score :' , scores[best_model])


if best_model == 'Linear Regression':
    model = GS_linear
elif best_model == 'KNN':
    model = GS_knn
elif best_model == 'Decision Tree':
    model = GS_dt
elif best_model == 'Random Forest':
    model = GS_rf
elif best_model == 'SVR':
    model = GS_svr

y_pred = model.predict(x_scaled)

plt.figure(figsize=(10, 6))
plt.plot(y, label='Actual values' ,color='red', linestyle='--', alpha=0.6)  # مقادیر واقعی
plt.plot(y_pred, label='Model predictions', color='blue', alpha=0.6)  # مقادیر پیش‌بینی شده
plt.legend()
plt.title('Comparison of actual values ​​and model predictions (best model)')
plt.xlabel('Samples')
plt.ylabel('target value')
plt.show()

'''
REPORT:
man ba estefadeh az Pandas dadeh hay koneh hay california ro be x ha va y  moratab kardam :
    
x hai ma shamele 8 ta az parametr hai hast ke az 2000 ta koneh dar california jam avari kardan 
x ha :
    MedInc
    HouseAge
    AveRoomes
    AveBedrms
    Population
    AveOccup
    Latitude
    Longitude


y ma gheymate on khoneha hast 
y :
    Price
....................................................................
    
In barnameh 5 model regrasion :
    
    Linear Regression
    K-Nearest Neighbors (KNN) 
    Decision Tree
    Random Forest
    Support Vector Regressor (SVR)
    
 roy e dadeha koneh haye california anjam midahad .  
 
 Ba  estefadeh az GridSearchCV va tanzim parametr hai mokhtalf barayeh har model , deghat model ro balayeh 80% beresonim.
 
 va dar enteha ba mohasebeh khata, mohasebeh deghat model , behtarin model barayeh regrasion ra miyabad va rasm mi konad.
 
 BA TASHAKOR.

'''
