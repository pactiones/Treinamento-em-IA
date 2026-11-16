# 01 Criar ou carregar outro dataset
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = dataset.target

# 02 Explorar as características
print("Dimensões de X:", X.shape)
print("Classes alvo:", dataset.target_names)
print("Distribuição das classes:", np.bincount(y))
print("\nPrimeiras 5 amostras:")
print(X.head())
print("\nValores ausentes por coluna:", X.isnull().sum().sum())

# 03 Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("Treino:", X_train.shape, "| Teste:", X_test.shape)

# 04 Treinar uma árvore
t0 = time.perf_counter()
arvore = DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)
tempo_arvore = time.perf_counter() - t0

pred_arvore = arvore.predict(X_test)
#Árvore treinada em {tempo_arvore:.4f}s | Profundidade: {arvore.get_depth()} | Folhas: {arvore.get_n_leaves()}")

# 05 Treinar uma Random Forest
t0 = time.perf_counter()
floresta = RandomForestClassifier(
    n_estimators=500,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
floresta.fit(X_train, y_train)
tempo_floresta = time.perf_counter() - t0

pred_floresta = floresta.predict(X_test)
#Floresta treinada em {tempo_floresta:.4f}s")

# 06 Comparar quantidades de árvores
quantidades = [1, 5, 10, 25, 50, 100, 200, 500]
res_n = []

for n in quantidades:
    usa_oob = n >= 25
    rf = RandomForestClassifier(n_estimators=n, oob_score=usa_oob, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    res_n.append({
        "n_estimators": n,
        "acc_treino": rf.score(X_train, y_train),
        "acc_teste": rf.score(X_test, y_test),
        "oob": rf.oob_score_ if usa_oob else np.nan
    })

df_n = pd.DataFrame(res_n)
print(df_n.to_string(index=False))

# 07 Investigar profundidades
profundidades = [1, 2, 3, 4, 6, 8, 10, None]
res_d = []

for d in profundidades:
    arv = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    flo = RandomForestClassifier(n_estimators=500, max_depth=d, random_state=42, n_jobs=-1).fit(X_train, y_train)
    res_d.append({
        "max_depth": str(d),
        "arvore_treino": arv.score(X_train, y_train),
        "arvore_teste": arv.score(X_test, y_test),
        "floresta_treino": flo.score(X_train, y_train),
        "floresta_teste": flo.score(X_test, y_test)
    })

df_d = pd.DataFrame(res_d)
print(df_d.to_string(index=False))

# 08 Utilizar OOB
acc_teste_floresta = accuracy_score(y_test, pred_floresta)
#OOB Score (estimativa interna no treino): {floresta.oob_score_:.4f}")
#Acurácia no conjunto de Teste:            {acc_teste_floresta:.4f}")
#Diferença (OOB - Teste):                  {floresta.oob_score_ - acc_teste_floresta:+.4f}")

# 09 Examinar importâncias
imp_mdi = pd.Series(floresta.feature_importances_, index=X.columns).sort_values(ascending=False)

perm = permutation_importance(floresta, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
imp_perm = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

print("Top 5 Características - Impureza (MDI):")
print(imp_mdi.head(5))
print("\nTop 5 Características - Permutação (Teste):")
print(imp_perm.head(5))

# 10 Comparar métricas
acc_treino_arv = accuracy_score(y_train, arvore.predict(X_train))
acc_teste_arv = accuracy_score(y_test, pred_arvore)
cm_arv = confusion_matrix(y_test, pred_arvore)

acc_treino_flo = accuracy_score(y_train, floresta.predict(X_train))
cm_flo = confusion_matrix(y_test, pred_floresta)

tabela_comparativa = pd.DataFrame([
    {
        "Modelo": "Árvore de Decisão",
        "Árvores": 1,
        "Acc Treino": f"{acc_treino_arv:.4f}",
        "Acc Teste": f"{acc_teste_arv:.4f}",
        "OOB": "n/a",
        "Matriz Confusão": cm_arv.tolist()
    },
    {
        "Modelo": "Random Forest",
        "Árvores": 500,
        "Acc Treino": f"{acc_treino_flo:.4f}",
        "Acc Teste": f"{acc_teste_floresta:.4f}",
        "OOB": f"{floresta.oob_score_:.4f}",
        "Matriz Confusão": cm_flo.tolist()
    }
])

print(tabela_comparativa.to_string(index=False))

# 11 Criar gráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Número de árvores
axes[0].plot(df_n["n_estimators"], df_n["acc_teste"], marker="o", label="Teste")
axes[0].plot(df_n["n_estimators"], df_n["oob"], marker="s", label="OOB")
axes[0].set_xscale("log")
axes[0].set_title("Acurácia vs. Nº de Árvores")
axes[0].set_xlabel("n_estimators (escala log)")
axes[0].set_ylabel("Acurácia")
axes[0].legend()

# Gráfico 2: Profundidade
axes[1].plot(df_d["max_depth"], df_d["arvore_teste"], marker="o", linestyle="--", label="Árvore (Teste)")
axes[1].plot(df_d["max_depth"], df_d["floresta_teste"], marker="s", label="Floresta (Teste)")
axes[1].set_title("Acurácia no Teste vs. Profundidade")
axes[1].set_xlabel("max_depth")
axes[1].set_ylabel("Acurácia")
axes[1].legend()

# Gráfico 3: Top 8 Características por Permutação
top8 = imp_perm.head(8)[::-1]
axes[2].barh(top8.index, top8.values, color="#352364")
axes[2].set_title("Top 8 Características (Permutação)")
axes[2].set_xlabel("Queda média na acurácia")

plt.tight_layout()
plt.show()

# 1. Ganho de generalização: A Random Forest supera a árvore de decisão única no conjunto de teste ao combinar 
# 500 árvores descorrelacionadas.
# 2. Controle de overfitting: Enquanto a árvore única sem limite de profundidade atinge 100% no treino 
# (decorando ruídos e apresentando alta variância), a floresta mantém a capacidade de generalização.
# 3. Confiabilidade do OOB: O oob_score_ fornece uma estimativa interna consistente e muito próxima da 
# acurácia real de teste, validando o modelo com as 36,8% amostras deixadas de fora de cada bootstrap.
# 4. Importância das características: A análise por permutação no teste confirma quais variáveis realmente 
# sustentam a acurácia do modelo, evitando o viés da importância por impureza (MDI) calculada apenas no treino.
# 5. Custo computacional vs. Benefício: Treinar e prever com 500 árvores exige mais tempo e processamento 
# do que uma árvore isolada, mas o paralelismo (n_jobs=-1) torna esse custo baixo frente ao ganho de 
# estabilidade e precisão.