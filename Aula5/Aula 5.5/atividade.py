import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

rng = np.random.default_rng(42)
for i in range(8):
    X[f"ruido_{i}"] = rng.normal(size=len(X))

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Número de features antes da seleção:", X_tr.shape[1])

from functools import partial
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

mi = partial(mutual_info_regression, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
modelo = HistGradientBoostingRegressor(random_state=42)

for k in [2, 4, 6, 8, 12, 16]:
    pipe = Pipeline([
        ("sel", SelectKBest(mi, k=k)),
        ("modelo", modelo),
    ])

    scores = cross_val_score(
        pipe, X_tr, y_tr,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    print(f"k={k:2} | R² médio={scores.mean():.4f} | desvio={scores.std():.4f}")

import pandas as pd

seletor = SelectKBest(mi, k=6)
seletor.fit(X_tr, y_tr)

features_mantidas = X_tr.columns[seletor.get_support()].tolist()
features_removidas = X_tr.columns[~seletor.get_support()].tolist()

print("\nMantidas:", features_mantidas)
print("Removidas:", features_removidas)
print("\nPontuação por feature:")
print(
    pd.Series(seletor.scores_, index=X_tr.columns)
      .sort_values(ascending=False)
      .round(4)
)

from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# Modelo com todas as 16 features
modelo_todas = clone(modelo)
modelo_todas.fit(X_tr, y_tr)
r2_todas = r2_score(y_te, modelo_todas.predict(X_te))

# Pipeline: seleciona 6 features no treino e ajusta o mesmo algoritmo
pipe_selecionado = Pipeline([
    ("sel", SelectKBest(mi, k=6)),
    ("modelo", clone(modelo)),
])
pipe_selecionado.fit(X_tr, y_tr)
r2_selecionadas = r2_score(y_te, pipe_selecionado.predict(X_te))

print(f"Número de features antes: {X_tr.shape[1]}")
print(f"Número de features depois: {len(features_mantidas)}")
print(f"R² no teste com todas: {r2_todas:.4f}")
print(f"R² no teste com selecionadas: {r2_selecionadas:.4f}")

'''
1. O numero de features antes da seleção é 16, e depois dela é 6.
2. O método de seleção utilizado foi o SelectKBest com a métrica de informação mútua.
3. As features mantidas foram: ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']
4. As features removidas foram: ['AveBedrms', 'Population', 'ruido_0', 'ruido_1', 'ruido_2', 'ruido_3', 'ruido_4', 'ruido_5', 'ruido_6', 'ruido_7']
5. As features com melhor pontuação foram selecionadas para manter, e as com menores pontuações foram removidas. 
O SelectKBest pontuou cada feature pela informação mútua com o alvo, e selecionou as 6 melhores.
6. O modelo, antes da seleção, obteve uma métrica de 0.8308, e depois da seleção, obteve uma métrica de 0.8373.
7. A seleção de features foi vantajosa nesse caso, tanto por diminuir o numero de features quanto por ter um 
desempenho melhor. A seleção de features é importante para reduzir a complexidade do modelo, melhorar a 
interpretabilidade, mantendo apenas as variáveis mais relevantes para a predição.
'''