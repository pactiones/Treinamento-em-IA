"""Aula 5.5 — Seleção de features · Exercício guiado (California Housing)."""

# ======================================================================
# Passo 1 — Dados, ruído, teste trancado e baseline
# ======================================================================

import numpy as np, pandas as pd, time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
rng = np.random.default_rng(42)
for i in range(8):
    X[f"ruido_{i}"] = rng.normal(size=len(X))      # 8 colunas que devem sair

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
modelo = HistGradientBoostingRegressor(random_state=42)

def avaliar(est):
    t = time.perf_counter()
    s = cross_val_score(est, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1)
    return round(s.mean(), 4), round(s.std(), 4), round(time.perf_counter() - t, 1)

print(X.shape, "| treino:", X_tr.shape, "| teste trancado:", X_te.shape)
base = avaliar(modelo)                              # baseline: 16 colunas
print("Baseline (16 colunas) -> R² = %.4f ± %.4f  (%.1f s)" % base)

# Tabela que vai sendo preenchida ao longo dos passos (Passo 5)
resultados = {"Baseline": dict(colunas=X_tr.shape[1], r2=base[0], desvio=base[1], tempo=base[2],
                               selecionadas=list(X_tr.columns))}
limite = base[0] - base[1]      # critério da aula: R² dentro de 1 desvio padrão do baseline
print(f"Faixa aceitável: R² >= {limite:.4f}")


# ======================================================================
# Passo 2 — Filtros: ANOVA F e informação mútua
# ======================================================================

from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

mi = partial(mutual_info_regression, random_state=42)

curvas = {"F": {}, "MI": {}}
for nome, f in [("F", f_regression), ("MI", mi)]:
    for k in [2, 4, 6, 8, 12, 16]:
        pipe = Pipeline([
            ("sel", SelectKBest(f, k=k)),
            ("m", modelo),
        ])
        r = avaliar(pipe)
        curvas[nome][k] = r
        print(nome, k, r)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
for nome, cor in [("F", "tab:blue"), ("MI", "tab:orange")]:
    ks = list(curvas[nome]); m = [curvas[nome][k][0] for k in ks]; s = [curvas[nome][k][1] for k in ks]
    ax.errorbar(ks, m, yerr=s, marker="o", capsize=3, label=f"SelectKBest {nome}", color=cor)
ax.axhline(base[0], color="k", ls="--", lw=1, label="baseline (16 col.)")
ax.axhspan(base[0] - base[1], base[0] + base[1], color="k", alpha=.08, label="± 1 desvio")
ax.set_xlabel("k (colunas mantidas)"); ax.set_ylabel("R² (CV 5 folds)")
ax.set_title("Filtros: R² de validação vs. k"); ax.legend(); ax.grid(alpha=.3)
plt.show()

for nome in curvas:
    k_ok = min(k for k, r in curvas[nome].items() if r[0] >= limite)
    print(f"{nome}: menor k dentro de 1 desvio do baseline = {k_ok}  ->  R² {curvas[nome][k_ok][0]:.4f}")
    r = curvas[nome][k_ok]
    resultados[f"SelectKBest {nome}"] = dict(colunas=k_ok, r2=r[0], desvio=r[1], tempo=r[2])

escolhas = {}
for nome, f in [("F", f_regression), ("MI", mi)]:
    for k in [resultados[f"SelectKBest {nome}"]["colunas"], 8]:
        pipe = Pipeline([("sel", SelectKBest(f, k=k)), ("m", modelo)]).fit(X_tr, y_tr)
        cols = pipe[:-1].get_feature_names_out().tolist()
        escolhas[(nome, k)] = cols
        print(f"{nome:>2} k={k:<2}: {cols}")
    resultados[f"SelectKBest {nome}"]["selecionadas"] = escolhas[(nome, resultados[f"SelectKBest {nome}"]["colunas"])]

# Os dois rankings completos, lado a lado (scores no treino inteiro, só para leitura)
sel_f  = SelectKBest(f_regression, k="all").fit(X_tr, y_tr)
sel_mi = SelectKBest(mi, k="all").fit(X_tr, y_tr)
rank = pd.DataFrame({"F": sel_f.scores_, "MI": sel_mi.scores_}, index=X_tr.columns)
rank["pos_F"]  = rank["F"].rank(ascending=False).astype(int)
rank["pos_MI"] = rank["MI"].rank(ascending=False).astype(int)
print(rank.sort_values("pos_MI").round(4))

for nome in ["F", "MI"]:
    ruidos = [c for c in escolhas[(nome, 8)] if c.startswith("ruido")]
    reais_fora = [c for c in X.columns[:8] if c not in escolhas[(nome, 8)]]
    print(f"{nome}, k=8 -> ruídos dentro: {ruidos or 'nenhum'} | colunas reais que ficaram de fora: {reais_fora or 'nenhuma'}")


# ======================================================================
# Passo 3 — Wrapper com RFECV
# ======================================================================

from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

pipe_rfe = Pipeline([
    ("sc", StandardScaler()),
    ("sel", RFECV(Ridge(), step=1, cv=3, scoring="r2")),
    ("m", modelo),
])
r = avaliar(pipe_rfe)                    # validação aninhada
print("RFECV + Ridge -> R² = %.4f ± %.4f  (%.1f s)" % r)

pipe_rfe.fit(X_tr, y_tr)
rfe = pipe_rfe.named_steps["sel"]
cols_rfe = X_tr.columns[rfe.support_].tolist()
print(rfe.n_features_, cols_rfe)
resultados["RFECV + Ridge"] = dict(colunas=rfe.n_features_, r2=r[0], desvio=r[1], tempo=r[2], selecionadas=cols_rfe)

res = rfe.cv_results_
n_feat = res["n_features"] if "n_features" in res else np.arange(1, len(res["mean_test_score"]) + 1)
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(n_feat, res["mean_test_score"], yerr=res["std_test_score"], marker="o", capsize=3)
ax.axvline(rfe.n_features_, color="tab:red", ls="--", label=f"escolhido: {rfe.n_features_}")
ax.set_xlabel("número de colunas"); ax.set_ylabel("R² do Ridge (CV interno, 3 folds)")
ax.set_title("RFECV: score do ranqueador vs. nº de colunas"); ax.legend(); ax.grid(alpha=.3)
plt.show()

print("Ordem de eliminação (1 = sobreviveu):")
print(pd.Series(rfe.ranking_, index=X_tr.columns).sort_values().to_string())


# ======================================================================
# Passo 4 — Embedded com Lasso e Random Forest
# ======================================================================

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

pipe_lasso = Pipeline([
    ("sc", StandardScaler()),
    ("sel", SelectFromModel(LassoCV(cv=3, random_state=42))),   # fica quem não zerou
    ("m", modelo),
])
r = avaliar(pipe_lasso)
print("Lasso -> R² = %.4f ± %.4f  (%.1f s)" % r)

pipe_lasso.fit(X_tr, y_tr)
lasso = pipe_lasso.named_steps["sel"].estimator_
cols_lasso = pipe_lasso[:-1].get_feature_names_out().tolist()
print(f"alpha escolhido = {lasso.alpha_:.5f} | {len(cols_lasso)} colunas: {cols_lasso}")
print(pd.Series(lasso.coef_, index=X_tr.columns).round(4).to_string())
resultados["Lasso"] = dict(colunas=len(cols_lasso), r2=r[0], desvio=r[1], tempo=r[2], selecionadas=cols_lasso)

pipe_lasso_t = Pipeline([
    ("sc", StandardScaler()),
    ("sel", SelectFromModel(LassoCV(cv=3, random_state=42), threshold="0.1*mean")),
    ("m", modelo),
])
r = avaliar(pipe_lasso_t)
pipe_lasso_t.fit(X_tr, y_tr)
cols_lasso_t = pipe_lasso_t[:-1].get_feature_names_out().tolist()
print("Lasso (limiar 0.1*mean) -> R² = %.4f ± %.4f  (%.1f s)" % r)
print(f"limiar = {pipe_lasso_t.named_steps['sel'].threshold_:.4f} | {len(cols_lasso_t)} colunas: {cols_lasso_t}")
resultados["Lasso (limiar 0.1·mean)"] = dict(colunas=len(cols_lasso_t), r2=r[0], desvio=r[1], tempo=r[2],
                                              selecionadas=cols_lasso_t)

X_a, X_v, y_a, y_v = train_test_split(X_tr, y_tr, test_size=0.25, random_state=42)
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42).fit(X_a, y_a)
perm = permutation_importance(rf, X_v, y_v, n_repeats=10, random_state=42, n_jobs=-1)
imp = pd.DataFrame({"MDI": rf.feature_importances_, "perm": perm.importances_mean},
                   index=X.columns).sort_values("perm", ascending=False)
print(imp.round(4))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
cores = ["tab:gray" if c.startswith("ruido") else "tab:green" for c in imp.index]
axes[0].barh(imp.index, imp["MDI"], color=cores); axes[0].set_title("MDI (treino)")
axes[1].barh(imp.index, imp["perm"], color=cores); axes[1].set_title("Permutação (validação)")
axes[0].invert_yaxis()
for a in axes: a.grid(axis="x", alpha=.3)
plt.suptitle("Random Forest — verde: colunas reais · cinza: ruído"); plt.tight_layout(); plt.show()

print("Soma da MDI dada ao ruído: %.3f" % imp.loc[imp.index.str.startswith("ruido"), "MDI"].sum())
print("Soma da permutação do ruído: %.4f" % imp.loc[imp.index.str.startswith("ruido"), "perm"].sum())


# ======================================================================
# Passo 5 — Comparar e abrir o teste uma vez
# ======================================================================

tab = pd.DataFrame(resultados).T[["colunas", "r2", "desvio", "tempo"]]
tab.columns = ["Colunas", "R² CV", "Desvio", "Tempo (s)"]
tab["Dentro de 1 desvio?"] = tab["R² CV"] >= limite
print(tab)

candidatos = tab[tab["Dentro de 1 desvio?"]].sort_values(["Colunas", "R² CV"], ascending=[True, False])
escolhido = candidatos.index[0]
print("Pipeline escolhido:", escolhido)
print("Colunas:", resultados[escolhido]["selecionadas"])

# Reconstrói o pipeline escolhido
k_F, k_MI = resultados["SelectKBest F"]["colunas"], resultados["SelectKBest MI"]["colunas"]
pipes = {
    "Baseline":         modelo,
    "SelectKBest F":    Pipeline([("sel", SelectKBest(f_regression, k=k_F)), ("m", modelo)]),
    "SelectKBest MI":   Pipeline([("sel", SelectKBest(mi, k=k_MI)), ("m", modelo)]),
    "RFECV + Ridge":    pipe_rfe,
    "Lasso":            pipe_lasso,
    "Lasso (limiar 0.1·mean)": pipe_lasso_t,
}
pipe_escolhido = pipes[escolhido]

final = pipe_escolhido.fit(X_tr, y_tr)
r2_teste = final.score(X_te, y_te)                  # uma única vez
print(f"{escolhido}: R² no teste = {r2_teste:.4f}  (R² CV = {resultados[escolhido]['r2']:.4f})")