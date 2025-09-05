import pandas as pd 
import numpy as np 
import matplotlib as plt 
import seaborn as sb
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score 

df = pd.read_csv(r'D:\nyc_temperature.csv') #Link to the dataset in README
df.head() 

X = df[["tmax", "tmin", "departure",  "HDD", "CDD" ]]
y = df["tavg"] 

X_train ,X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 42 ) 
model = RandomForestRegressor(n_estimators = 100 , random_state = 42 ) 
model.fit(X_train, y_train ) 

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

