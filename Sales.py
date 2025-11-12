import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pickle

df = pd.read_csv(r'C:\Users\alokk\Documents\GPU_PROJECT\Sales_prediction\salesPrediction\sales data file.csv')
df 
print(df) 
df.isna().sum()
df.dropna(inplace=True)
df.hist(figsize=(10,10), bins =50 )

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True)
# plot the pairplot
sns.pairplot(df)
plt.show()

X = df.drop('Sales', axis=1)
y = df["Sales"]
y = pd.DataFrame(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

regression_Model = RandomForestRegressor(n_estimators=250,random_state=0)
regression_Model.fit(X_train, y_train)

y_pred = regression_Model.predict(X_test)
model = regression_Model.fit(X_train, y_train)
print("Predictions:", y_pred)

print('Random Forest Regressor Train Score is : ' ,  regression_Model.score(X_train, y_train))
print('Random Forest Regressor Test Score is : ' , regression_Model.score(X_test, y_test))

print('Random Forest Regression Model')
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))

# Save model using pickle
with open('sales_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as sales_model.pkl")