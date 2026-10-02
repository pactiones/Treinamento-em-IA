import sys
import subprocess
import importlib


def instalar_bibliotecas():

    bibliotecas = {
        "pandas": "pandas",
        "numpy": "numpy",
        "seaborn": "seaborn",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "IPython": "ipython"
    }

    print("=" * 60)
    print("VERIFICANDO BIBLIOTECAS NECESSÁRIAS")
    print("=" * 60)

    for modulo, pacote in bibliotecas.items():

        try:
            importlib.import_module(modulo)
            print(f"[OK] {pacote} já está instalado.")

        except ImportError:

            print(f"[INSTALANDO] {pacote}...")

            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                pacote
            ])

            print(f"[OK] {pacote} instalado com sucesso.")

    print("\nTodas as bibliotecas estão prontas!")
    print("=" * 60)


instalar_bibliotecas()


# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. CARREGAMENTO DO DATASET
# ============================================================

df = sns.load_dataset("penguins")


print("\n ===== Análise dos dados do dataset: ===")

print("\nPrimeiras linhas: ")
display(df.head())

print("\nDimensão do dataset: ")
display(df.shape)

print("\nValores ausentes: ")
display(df.isna().sum())

print("\nExistem valores duplicados: ")
if(df.duplicated().sum() > 0):
    print("Sim.")
else:
    print("Não.")

X = df.drop(columns="species")
y = df["species"]

num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
cat_cols = ["sex", "island"]

X = X[num_cols + cat_cols]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = 0.3, stratify=y, random_state=42)

print("\nDimensão do treino e do teste: ")
display(X_tr.shape, X_te.shape)

print("\nCodificação das variaveis categóricas e padronização das variaveis numéricas: ")

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

prep = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

pipe = Pipeline([
    ("prep", prep),
    ("model", LogisticRegression(max_iter=1000))
])

print("\nTreinamento do modelo...")
print(pipe.fit(X_tr, y_tr))

print("\nRealização das previsões...")
y_pred = pipe.predict(X_te)

print("\nAcurácia do modelo: ")
print(accuracy_score(y_te, y_pred))

print("\nMatriz de confusão: ")
ConfusionMatrixDisplay.from_predictions(y_te, y_pred)
plt.title("Matriz de Confusão")
plt.show()

'''
Insights:
1. O dataset "penguins" contém informações sobre espécies de pinguins, sendo elas: Ilhas em que o pinguim foi observado, 
Comprimento e Profundidade do bico, Comprimento da nadadeira, Massa Corporal e Sexo. O objetivo do dataset é prever a 
espécie do pinguim baseado nessas características.

2. O dataset possui 344 linhas e 7 colunas. Existem valores ausentes: 2 na coluna "bill_length_mm", 2 na coluna "bill_depth_mm", 
2 na coluna "flipper_length_mm", 2 na coluna "body_mass_g" e 11 na coluna "sex". Não existem valores duplicados.

3. A divisão do dataset entre treino e teste, foi de 70% para treino e os 30% restantes para teste.

4. Para tratar os valores das variáeis numéricas, foi utilizado a mediana. E para tratar os valores das variáveis categóricas,
utilizou-se a moda.

5. A acurácia final, usando regressão logística, foi de aproximadamente 0.99.
'''