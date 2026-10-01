import pandas as pd
from sklearn.model_selection import (train_test_split)

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

print("Primeiras linhas: ")
print(df.head())

print("\nDimensões: ")
print(df.shape)

print("\nValores ausentes por coluna: ")
print(df.isnull().sum())

X = df.drop(columns="Survived")
y = df["Survived"]

#tipos de colunas
num_cols =["Age", "Fare", "SibSp", "Parch"]
cat_cols = ["Sex", "Embarked", "Pclass"]

X = X[num_cols + cat_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

print(X_train.shape, X_test.shape)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler())
])

#Transformando categoricas em numeros
from sklearn.preprocessing import OneHotEncoder

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

#Juntar os caminhos
from sklearn.compose import ColumnTransformer

prep = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("prep", prep), 
    ("model", LogisticRegression(max_iter = 1000))
])

print(pipe.fit(X_train, y_train))