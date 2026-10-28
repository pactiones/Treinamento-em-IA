#Imports, dataset e exploração

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import (train_test_split, KFold, cross_val_score, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.metrics import mean_squared_error, r2_score

SEED = 42
print("sklearn", sklearn.__version__)
X, y = load_diabetes(return_X_y=True, as_frame=True)
print(X.shape)  #(442, 10)
print(X.corrwith(y).sort_values())  #correlação com o alvo

#Divisão, dobras e baseline
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = "neg_root_mean_squared_error"

def make_pipe(model):
    return Pipeline([("scaler", StandardScaler()),
                     ("model", model)])

baseline = make_pipe(LinearRegression())
scores = cross_val_score(baseline, X_train, y_train, cv = cv, scoring=SCORING)
print(f"MQO RMSE-CV = {-scores.mean():.2f} ± {scores.std():.2f}")

#Ridge e Lasso com busca de alpha

ridge_search = GridSearchCV(
    make_pipe(Ridge()),
    {"model__alpha": np.logspace(-3, 3, 61)},
    cv=cv, scoring=SCORING)
ridge_search.fit(X_train, y_train)

lasso_search = GridSearchCV(
    make_pipe(Lasso(max_iter=50_000)),
    {"model__alpha" : np.logspace(-3, 1.5, 46)},
    cv = cv, scoring=SCORING)
    
lasso_search.fit(X_train, y_train)

print(ridge_search.best_params_, ridge_search.best_score_)
print(lasso_search.best_params_, lasso_search.best_score_)

#ElasticNet e comparação por CV
enet_search = GridSearchCV(
    make_pipe(ElasticNet(max_iter=50_000)),
    {"model__alpha": np.logspace(-3, 1.5, 31),
     "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
     cv = cv, scoring=SCORING)
enet_search.fit(X_train, y_train)

resumo = pd.DataFrame({
    "modelo": ["MQO", "Ridge", "Lasso", "ElasticNet"],
    "RMSE_CV": [-scores.mean(), -ridge_search.best_score_,
                -lasso_search.best_score_,
                -enet_search.best_score_]})
print(resumo.sort_values("RMSE_CV"))

#Coeficientes e caminho do Lasso
baseline.fit(X_train, y_train)
def get(pipe):
    return pipe.named_steps["model"].coef_

coefs = pd.DataFrame({
    "MQO": get(baseline),
    "Ridge": get(ridge_search.best_estimator_),
    "Lasso": get(lasso_search.best_estimator_),
    "ElasticNet": get(enet_search.best_estimator_)},
    index=X.columns)
print(coefs.round(1)); print((coefs == 0).sum())

Xs = StandardScaler().fit_transform(X_train)
alphas = np.logspace(-3, 1.5, 60)
path = [Lasso(alpha=a, max_iter=50_000).fit(Xs, y_train).coef_
        for a in alphas]
plt.semilogx(alphas, path); plt.xlabel("alpha")
plt.ylabel("coeficiente"); plt.show()

#Avaliação final no teste

candidatos = {"MQO": baseline, "Ridge": ridge_search,
              "Lasso": lasso_search, "ElasticNet": enet_search}
escolhido = resumo.loc[resumo["RMSE_CV"].idxmin(), "modelo"]
modelo_final = candidatos[escolhido]

y_pred = modelo_final.predict(X_test) #teste: unica vez
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

alphas = np.logspace(-3, 1.5, 46)

modelos = {
    "Ridge": lambda a: Ridge(alpha=a),
    "Lasso": lambda a: Lasso(alpha=a, max_iter=50_000),
    "ElasticNet": lambda a: ElasticNet(
        alpha=a, l1_ratio=0.5, max_iter=50_000
    ),
}

resultados = {}

for nome, criar_modelo in modelos.items():
    medias, desvios = [], []

    for alpha in alphas:
        pipe = make_pipe(criar_modelo(alpha))
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring=SCORING
        )
        rmse_dobras = -scores
        medias.append(rmse_dobras.mean())
        desvios.append(rmse_dobras.std())

    resultados[nome] = {
        "media": np.array(medias),
        "desvio": np.array(desvios),
    }

for nome, r in resultados.items():
    plt.semilogx(alphas, r["media"], label=nome)

plt.xlabel("alpha")
plt.ylabel("RMSE médio na validação cruzada")
plt.legend()
plt.grid(True)
plt.show()

alphas_exemplo = np.logspace(-3, 1.5, 5)
linhas = []

for alpha in alphas_exemplo:
    pipe = make_pipe(Lasso(alpha=alpha, max_iter=50_000))
    pipe.fit(X_train, y_train)

    coef = pipe.named_steps["model"].coef_
    linhas.append({
        "alpha": alpha,
        "coeficientes_zerados": np.isclose(coef, 0).sum(),
    })

tabela_zeros = pd.DataFrame(linhas)

r = resultados["Lasso"]
indice_min = r["media"].argmin()
limite = r["media"][indice_min] + r["desvio"][indice_min]

elegiveis = np.where(r["media"] <= limite)[0]
indice_escolhido = elegiveis[-1]  # alphas está em ordem crescente
alpha_1se = alphas[indice_escolhido]

tabela_zeros["alpha"] = tabela_zeros["alpha"].map(lambda x: f"{x:.4f}")
print(tabela_zeros)

rmse_min = r["media"][indice_min]
desvio_no_min = r["desvio"][indice_min]
rmse_1se = r["media"][indice_escolhido]

lasso_1se = make_pipe(Lasso(alpha=alpha_1se, max_iter=50_000))
lasso_1se.fit(X_train, y_train)

print(f"RMSE-CV mínimo: {rmse_min:.3f}")
print(f"Desvio padrão nesse mínimo: {desvio_no_min:.3f}")
print(f"Limite da regra de 1 DP: {limite:.3f}")
print(f"alpha escolhido pela regra: {alpha_1se:.4f}")
print(f"RMSE-CV nesse alpha: {rmse_1se:.3f}")
print(f"Coeficientes zerados nesse alpha: "
      f"{np.isclose(lasso_1se.named_steps['model'].coef_, 0).sum()}")

sementes = [7, 21, 42]
linhas_sementes = []
indicadores_zero = []

for semente in sementes:
    cv_semente = KFold(n_splits=5, shuffle=True, random_state=semente)
    medias_seed, desvios_seed = [], []

    for alpha in alphas:
        pipe = make_pipe(Lasso(alpha=alpha, max_iter=50_000))
        scores_seed = cross_val_score(
            pipe, X_train, y_train,
            cv=cv_semente, scoring=SCORING
        )
        rmse_seed = -scores_seed
        medias_seed.append(rmse_seed.mean())
        desvios_seed.append(rmse_seed.std())

    medias_seed = np.array(medias_seed)
    desvios_seed = np.array(desvios_seed)

    idx_min_seed = medias_seed.argmin()
    limite_seed = medias_seed[idx_min_seed] + desvios_seed[idx_min_seed]
    elegiveis_seed = np.where(medias_seed <= limite_seed)[0]
    idx_1se_seed = elegiveis_seed[-1]
    alpha_seed = alphas[idx_1se_seed]

    modelo_seed = make_pipe(
        Lasso(alpha=alpha_seed, max_iter=50_000)
    )
    modelo_seed.fit(X_train, y_train)
    coef_seed = modelo_seed.named_steps["model"].coef_
    zeros_seed = np.isclose(coef_seed, 0)

    indicadores_zero.append(zeros_seed)
    linhas_sementes.append({
        "semente": semente,
        "alpha_1se": alpha_seed,
        "RMSE_CV": medias_seed[idx_1se_seed],
        "zeros": zeros_seed.sum(),
    })

tabela_sementes = pd.DataFrame(linhas_sementes)
frequencia_zeros = pd.DataFrame(
    indicadores_zero,
    index=[f"semente_{s}" for s in sementes],
    columns=X.columns
).sum().rename("vezes zerado")

print(tabela_sementes.round(3))
print(frequencia_zeros)

y_pred = lasso_1se.predict(X_test)
rmse_teste = mean_squared_error(y_test, y_pred) ** 0.5
r2_teste = r2_score(y_test, y_pred)

print(f"Lasso (regra de 1 DP): RMSE teste = "
      f"{rmse_teste:.2f} | R² = {r2_teste:.3f}")