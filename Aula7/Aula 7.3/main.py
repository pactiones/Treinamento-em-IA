# Etapa 01 - Imports
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

# Etapas 02 e 03 - Criar e visualizar dataset
X, y = make_moons(n_samples=600, noise=0.28, random_state=42)
print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Classes:", np.unique(y))
print("Por classe:", np.bincount(y))

plt.figure(figsize=(7, 5.5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="#352364", label="Classe 0", edgecolors="#15121D", linewidths=0.4)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="#D7FF56", label="Classe 1", edgecolors="#15121D", linewidths=0.4)
plt.title("Dispersão das duas luas")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()

# Etapa 04 - Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("Treino e Teste:", X_train.shape, X_test.shape)
print("Distribuição treino e teste:", np.bincount(y_train), np.bincount(y_test))

# Etapas 05, 06 e 07 - Treinar, prever e avaliar uma árvore
arvore = DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

print("Profundidade:", arvore.get_depth())
print("Folhas:", arvore.get_n_leaves())

pred_arvore = arvore.predict(X_test)
print("Previsto:", pred_arvore[:10])
print("Real:    ", y_test[:10])

acc_treino_arvore = accuracy_score(y_train, arvore.predict(X_train))
acc_teste_arvore = accuracy_score(y_test, pred_arvore)
cm_arvore = confusion_matrix(y_test, pred_arvore)

print("Acc. treino:", acc_treino_arvore)
print("Acc. teste :", acc_teste_arvore)
print("Matriz de confusão:\n", cm_arvore)

# Etapa 08 - Função e plotagem da região de decisão (Árvore)
COR_CLASSE = ["#352364", "#D7FF56"]
COR_REGIAO = ["#C9BEF5", "#F1FFC2"]
LIMITES = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

def plotar_fronteira(modelo, X_plot, y_plot, ax, titulo, resolucao=300):
    x_min, x_max, y_min, y_max = LIMITES
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolucao), np.linspace(y_min, y_max, resolucao))
    grade = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(grade).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=COR_REGIAO)
    for classe in (0, 1):
        m = y_plot == classe
        ax.scatter(X_plot[m, 0], X_plot[m, 1], s=18, c=COR_CLASSE[classe], edgecolors="#15121D", linewidths=0.4, label=f"Classe {classe}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(titulo)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right")

fig, ax = plt.subplots(figsize=(7, 5.5))
plotar_fronteira(arvore, X_test, y_test, ax, "Árvore de Decisão")
plt.show()

# Etapas 09, 10, 11 e 12 - Criar, treinar, prever e avaliar a floresta
floresta = RandomForestClassifier(
    n_estimators=500,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

t0 = time.perf_counter()
floresta.fit(X_train, y_train)
tempo_floresta = time.perf_counter() - t0

pred_floresta = floresta.predict(X_test)
print("Discordâncias entre Árvore e Floresta:", (pred_floresta != pred_arvore).sum())

acc_treino_floresta = accuracy_score(y_train, floresta.predict(X_train))
acc_teste_floresta = accuracy_score(y_test, pred_floresta)
cm_floresta = confusion_matrix(y_test, pred_floresta)

print("Acc. treino:", acc_treino_floresta)
print("Acc. teste :", acc_teste_floresta)
print("Tempo (s)  :", tempo_floresta)
print("Matriz de confusão:\n", cm_floresta)

# Etapa 13 - Fronteiras lado a lado
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
plotar_fronteira(arvore, X_test, y_test, axes[0], "Árvore de Decisão (1 árvore)")
plotar_fronteira(floresta, X_test, y_test, axes[1], "Random Forest (500 árvores)")
plt.show()

# Etapa 14 - OOB e acurácia de teste
print("OOB Score:", floresta.oob_score_)
print("Acc. teste:", acc_teste_floresta)
print("Diferença :", floresta.oob_score_ - acc_teste_floresta)

# Etapas 15 e 16 - Importância MDI e Permutação
nomes = ["x1", "x2"]
imp_mdi = floresta.feature_importances_
print("MDI:", dict(zip(nomes, imp_mdi)), "Soma:", imp_mdi.sum())

perm = permutation_importance(
    floresta,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)
print("Permutação Média:", perm.importances_mean)
print("Permutação Desvio:", perm.importances_std)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(nomes, imp_mdi, color="#7255E8")
axes[0].set_title("Importância MDI (Impureza)")
axes[1].bar(nomes, perm.importances_mean, yerr=perm.importances_std, color="#352364")
axes[1].set_title("Importância por Permutação")
plt.show()

# Etapa 17 - Quantidade de árvores
quantidades = [1, 5, 10, 25, 50, 100, 200, 500]
resultados_n = []

for n in quantidades:
    usa_oob = n >= 25
    rf_n = RandomForestClassifier(n_estimators=n, oob_score=usa_oob, random_state=42, n_jobs=-1)
    rf_n.fit(X_train, y_train)
    resultados_n.append({
        "n_estimators": n,
        "acc_treino": accuracy_score(y_train, rf_n.predict(X_train)),
        "acc_teste": accuracy_score(y_test, rf_n.predict(X_test)),
        "oob": rf_n.oob_score_ if usa_oob else np.nan
    })

tabela_n = pd.DataFrame(resultados_n)
print(tabela_n)

plt.figure(figsize=(7, 4))
plt.plot(tabela_n["n_estimators"], tabela_n["acc_teste"], marker="o", label="Teste")
plt.plot(tabela_n["n_estimators"], tabela_n["oob"], marker="s", label="OOB")
plt.xscale("log")
plt.xlabel("n_estimators (escala log)")
plt.ylabel("Acurácia")
plt.title("Acurácia por número de árvores")
plt.legend()
plt.show()

# Etapa 18 - Profundidade das árvores
profundidades = [1, 2, 3, 4, 6, 8, 10, None]
resultados_d = []

for d in profundidades:
    a = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    f = RandomForestClassifier(n_estimators=500, max_depth=d, random_state=42, n_jobs=-1).fit(X_train, y_train)
    for nome, m in [("Árvore", a), ("Floresta", f)]:
        resultados_d.append({
            "modelo": nome,
            "max_depth": str(d),
            "acc_treino": m.score(X_train, y_train),
            "acc_teste": m.score(X_test, y_test)
        })

tabela_d = pd.DataFrame(resultados_d)
print(tabela_d)