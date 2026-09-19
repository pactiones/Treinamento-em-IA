import pandas as pd
from sklearn.datasets import fetch_openml

df = fetch_openml("mushroom", version=1, as_frame=True).frame
print(df.shape)

from sklearn.model_selection import train_test_split

X = df.drop(columns="class")
y = df["class"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print(X_tr.shape, X_te.shape, y_tr. shape, y_te.shape)
print("Split finalizado com sucesso!")

print("Valores nulos por coluna: ")
print(X_tr.isna().sum().sort_values(ascending=False))

#Declarar os dois grupos
num_cols = []
cat_cols = X.columns.tolist()

assert set(num_cols + cat_cols) == set(X_tr.columns)
print(set(num_cols + cat_cols) == set(X_tr.columns))

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')), 
    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
#Os valores ausentes de stalk-root foram preenchidos com a moda

from sklearn.compose import ColumnTransformer

prep = ColumnTransformer([
    ("cat", cat_pipe, cat_cols)
])

print(prep.fit_transform(X_tr).shape)

#Modelo inteiro em um objeto
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("prep", prep),
    ("clf", LogisticRegression(max_iter=1000))
])

scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="accuracy")
print(scores.mean().round(4), scores.std().round(4))

#Abrir o conjunto trancado
pipe.fit(X_tr, y_tr)
print(pipe.score(X_te, y_te))

#Ler os coeficientes com os nomes
names = pipe.named_steps["prep"].get_feature_names_out()
coefs = pipe.named_steps["clf"].coef_[0]
print(pd.Series(coefs, index=names).sort_values())

#Ajustar pré-processamento por busca
from sklearn.model_selection import GridSearchCV

grid = {
    "prep__cat__imputer__strategy": ["most_frequent"],
    "clf__C": [0.1, 1.0, 10.0]
}

gs = GridSearchCV(pipe, grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_tr, y_tr)
print(gs.best_params_, round(gs.best_score_, 4))

#Transformação sem estado
from sklearn.preprocessing import FunctionTransformer

#Salvar objeto
import joblib

joblib.dump(pipe, "mushroom_pipeline_joblib")

modelo = joblib.load("mushroom_pipeline_joblib")

nova = pd.DataFrame([{
    "cap-shape": "x",
    "cap-surface": "s",
    "cap-color": "n",
    "bruises%3F": "t",
    "odor": "n",
    "gill-attachment": "f",
    "gill-spacing": "c",
    "gill-size": "b",
    "gill-color": "k",
    "stalk-shape": "e",
    "stalk-root": float("nan"),
    "stalk-surface-above-ring": "s",
    "stalk-surface-below-ring": "s",
    "stalk-color-above-ring": "w",
    "stalk-color-below-ring": "w",
    "veil-type": "p",
    "veil-color": "w",
    "ring-number": "o",
    "ring-type": "p",
    "spore-print-color": "k",
    "population": "s",
    "habitat": "u"
}])

print(modelo.predict_proba(nova)[0, 1].round(3))


