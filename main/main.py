import pandas as pd

df = pd.read_csv("car-price-prediction.csv")
print(df.head())
print(df.columns)
print(df.info())
df = df.dropna()

import matplotlib.pyplot as plt

plt.scatter(df.iloc[:,0], df.iloc[:,-1])
plt.xlabel("Feature")
plt.ylabel("Price")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df.drop(df.columns[-1], axis=1)
y = df[df.columns[-1]]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))