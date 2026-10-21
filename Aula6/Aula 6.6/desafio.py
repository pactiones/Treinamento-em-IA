import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestCentroid


# ============================================================
# ETAPAS 1 A 3 · carregar e conhecer os dados
# ============================================================

dados = load_breast_cancer()

X = dados.data
y = dados.target

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)

classes, contagens = np.unique(y, return_counts=True)

print("\nDistribuição das classes:")
for classe, quantidade in zip(classes, contagens):
    nome_classe = dados.target_names[classe]
    print(f"Classe {classe} ({nome_classe}): {quantidade} amostras")


# ============================================================
# ETAPA 4 · K-Fold feito manualmente
# ============================================================

def meu_kfold(n_amostras, k=5, shuffle=False, random_state=None):

    indices = np.arange(n_amostras)

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(indices)

    tamanhos = np.full(k, n_amostras // k)
    tamanhos[:n_amostras % k] += 1

    inicio = 0

    for tamanho in tamanhos:
        fim = inicio + tamanho

        val_idx = indices[inicio:fim]
        train_idx = np.concatenate(
            [indices[:inicio], indices[fim:]]
        )

        yield train_idx, val_idx
        inicio = fim

print("\nExemplo de K-Fold com 10 amostras e 3 folds:")

for numero, (train_idx, val_idx) in enumerate(
    meu_kfold(10, k=3),
    start=1
):
    print(f"Fold {numero} | treino: {train_idx} | validação: {val_idx}")

val_indices_exemplo = np.concatenate([
    val_idx
    for _, val_idx in meu_kfold(10, k=3)
])

print(
    "Cada índice aparece uma vez na validação:",
    np.array_equal(np.sort(val_indices_exemplo), np.arange(10))
)


# ============================================================
# ETAPA 6 · Stratified K-Fold feito manualmente
# ============================================================

def meu_stratified_kfold(y, k=5, shuffle=False, random_state=None):

    y = np.asarray(y)
    classes, contagens = np.unique(y, return_counts=True)

    rng = np.random.default_rng(random_state)
    partes_por_fold = [[] for _ in range(k)]

    for classe in classes:
        indices_classe = np.flatnonzero(y == classe)

        if shuffle:
            indices_classe = rng.permutation(indices_classe)

        partes = np.array_split(indices_classe, k)

        for numero_fold, parte in enumerate(partes):
            partes_por_fold[numero_fold].append(parte)

    todos_indices = np.arange(len(y))

    for partes in partes_por_fold:
        val_idx = np.sort(np.concatenate(partes))
        train_idx = np.setdiff1d(todos_indices, val_idx)

        yield train_idx, val_idx


# ============================================================
# ETAPA 7 · mostrar índices e distribuição de cada fold
# ============================================================

folds_stratified = list(
    meu_stratified_kfold(
        y,
        k=5,
        shuffle=True,
        random_state=42
    )
)

print("\nÍndices e distribuição das classes por fold:")

for numero, (train_idx, val_idx) in enumerate(
    folds_stratified,
    start=1
):
    distribuicao_validacao = np.bincount(
        y[val_idx],
        minlength=len(classes)
    )

    print(f"\nFold {numero}")
    print("Índices de treino:", train_idx)
    print("Índices de validação:", val_idx)
    print("Amostras de treino:", len(train_idx))
    print("Amostras de validação:", len(val_idx))
    print("Classes na validação:", distribuicao_validacao)


# Verificar que cada amostra foi validação exatamente uma vez
todos_indices_validacao = np.concatenate([
    val_idx
    for _, val_idx in folds_stratified
])

print(
    "\nCada amostra aparece uma vez na validação:",
    np.array_equal(
        np.sort(todos_indices_validacao),
        np.arange(len(y))
    )
)

# Verificar que treino e validação não compartilham índices
sem_sobreposicao = all(
    len(np.intersect1d(train_idx, val_idx)) == 0
    for train_idx, val_idx in folds_stratified
)

print("Treino e validação não têm índices em comum:", sem_sobreposicao)


# ============================================================
# ETAPA 8 · modelo de referência
# ============================================================

class CentroideMaisProximo:

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        self.centroides_ = np.array([
            X[y == classe].mean(axis=0)
            for classe in self.classes_
        ])

        return self

    def predict(self, X):
        distancias = np.stack([
            np.linalg.norm(X - centroide, axis=1)
            for centroide in self.centroides_
        ], axis=1)

        indices_classes = np.argmin(distancias, axis=1)

        return self.classes_[indices_classes]


def acuracia(y_true, y_pred):
    """Calcula a proporção de previsões corretas."""
    return np.mean(y_true == y_pred)


# ============================================================
# ETAPA 11 · cross-validation próprio
# ============================================================

def meu_cross_val_score(modelo_factory, X, y, folds, metrica):
    scores = []

    for numero, (train_idx, val_idx) in enumerate(folds, start=1):
        X_tr = X[train_idx]
        y_tr = y[train_idx]

        X_val = X[val_idx]
        y_val = y[val_idx]

        # Um modelo novo é criado para cada rodada.
        modelo = modelo_factory()

        modelo.fit(X_tr, y_tr)

        y_pred = modelo.predict(X_val)
        score = metrica(y_val, y_pred)

        scores.append(score)

        print(
            f"Fold {numero}: "
            f"treino={len(train_idx)}, "
            f"validação={len(val_idx)}, "
            f"acurácia={score:.4f}"
        )

    return np.array(scores)

print("\nAvaliação com Stratified K-Fold:")

scores_stratified = meu_cross_val_score(
    CentroideMaisProximo,
    X,
    y,
    folds_stratified,
    acuracia
)


# ============================================================
# ETAPA 10 · média e desvio-padrão dos scores
# ============================================================

media = np.mean(scores_stratified)
desvio_padrao = np.std(scores_stratified, ddof=0)

print("\nScores individuais:", scores_stratified)
print(f"Média: {media:.4f}")
print(f"Desvio-padrão populacional (ddof=0): {desvio_padrao:.4f}")


# ============================================================
# Avaliar também outras formas de divisão
# ============================================================

formas_de_divisao = {
    "K-Fold sem shuffle": list(
        meu_kfold(len(y), k=5)
    ),
    "K-Fold com shuffle": list(
        meu_kfold(
            len(y),
            k=5,
            shuffle=True,
            random_state=42
        )
    ),
    "Stratified K-Fold": folds_stratified
}

resultados = {
    "Stratified K-Fold": scores_stratified
}

print("\nComparação entre estratégias:")

for nome, folds in formas_de_divisao.items():
    if nome == "Stratified K-Fold":
        scores = scores_stratified
    else:
        print(f"\n{nome}:")
        scores = meu_cross_val_score(
            CentroideMaisProximo,
            X,
            y,
            folds,
            acuracia
        )

    resultados[nome] = scores

    print(
        f"{nome}: scores={scores} | "
        f"média={scores.mean():.4f} | "
        f"desvio-padrão={scores.std(ddof=0):.4f}"
    )

classe_mais_frequente = classes[np.argmax(contagens)]
previsoes_baseline = np.full_like(y, classe_mais_frequente)
score_baseline = acuracia(y, previsoes_baseline)

print(
    "\nBaseline da classe mais frequente:",
    f"{score_baseline:.4f}"
)


# ============================================================
# Gráfico dos scores por fold
# ============================================================

plt.figure(figsize=(9, 5))

for nome, scores in resultados.items():
    plt.plot(
        range(1, len(scores) + 1),
        scores,
        marker="o",
        label=f"{nome} ({scores.mean():.3f} ± {scores.std(ddof=0):.3f})"
    )

plt.axhline(
    score_baseline,
    linestyle="--",
    color="gray",
    label=f"Baseline ({score_baseline:.3f})"
)

plt.xticks(range(1, 6))
plt.ylim(0, 1.05)
plt.xlabel("Fold")
plt.ylabel("Acurácia")
plt.title("Scores por fold")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# ETAPA 12 · comparação com scikit-learn
# ============================================================

print("\nComparação com scikit-learn:")

# Comparar os índices de validação do K-Fold sem shuffle
meus_folds_kfold = list(
    meu_kfold(len(y), k=5)
)

folds_kfold_sklearn = list(
    KFold(n_splits=5).split(X)
)

indices_kfold_iguais = all(
    np.array_equal(val_idx_meu, val_idx_sklearn)
    for (_, val_idx_meu), (_, val_idx_sklearn)
    in zip(meus_folds_kfold, folds_kfold_sklearn)
)

print(
    "Meu K-Fold e KFold do scikit-learn usam "
    "os mesmos índices de validação:",
    indices_kfold_iguais
)

scores_sklearn_mesmos_folds = cross_val_score(
    NearestCentroid(),
    X,
    y,
    cv=folds_stratified
)

print("\nComparação usando os mesmos folds:")
print("Minha implementação:", scores_stratified)
print("scikit-learn:", scores_sklearn_mesmos_folds)
print(
    "Scores equivalentes:",
    np.allclose(scores_stratified, scores_sklearn_mesmos_folds)
)


cv_sklearn = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores_sklearn_folds_proprios = cross_val_score(
    NearestCentroid(),
    X,
    y,
    cv=cv_sklearn
)

print("\nStratifiedKFold do scikit-learn:")
print("Scores:", scores_sklearn_folds_proprios)
print(f"Média: {scores_sklearn_folds_proprios.mean():.4f}")
print(
    "Desvio-padrão:",
    f"{scores_sklearn_folds_proprios.std(ddof=0):.4f}"
)