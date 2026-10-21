import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import warnings
 
# Só usados na etapa 12 (comparação com o scikit-learn)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestCentroid
 
np.set_printoptions(precision=3, suppress=True)
 
 
# %% [markdown]
# ## DADOS · etapas 1 a 3
 
# %% ---------------------------------------------------------------------------
# ETAPA 1 · criar o dataset
# Duas nuvens gaussianas 2D: 90 amostras da classe 0 em torno de (0, 0)
# e 10 amostras da classe 1 em torno de (2, 2). Base desbalanceada (90/10).
# -----------------------------------------------------------------------------
rng = np.random.default_rng(0)
X0 = rng.normal(loc=[0, 0], scale=1.0, size=(90, 2))   # classe 0
X1 = rng.normal(loc=[2, 2], scale=1.0, size=(10, 2))   # classe 1
 
# %% ---------------------------------------------------------------------------
# ETAPA 2 · identificar X e y
# X: uma linha por amostra, uma coluna por característica  -> (100, 2)
# y: um rótulo por amostra                                  -> (100,)
# Atenção: o dataset está ORDENADO por classe (90 zeros primeiro, depois 10 uns).
# -----------------------------------------------------------------------------
X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(90, dtype=int), np.ones(10, dtype=int)])
 
print("Etapa 2 ·", X.shape, y.shape)
 
# %% ---------------------------------------------------------------------------
# ETAPA 3 · distribuição das classes
# -----------------------------------------------------------------------------
classes, contagens = np.unique(y, return_counts=True)
print("Etapa 3 · classes:", classes, "| contagens:", contagens,
      "| proporção de positivos:", contagens[1] / len(y))
 
plt.figure(figsize=(5, 4))
plt.scatter(X0[:, 0], X0[:, 1], s=18, alpha=0.7, label="classe 0 (90)")
plt.scatter(X1[:, 0], X1[:, 1], s=40, marker="^", label="classe 1 (10)")
plt.xlabel("X[:, 0]"); plt.ylabel("X[:, 1]")
plt.title("Etapa 3 · dataset 90 / 10")
plt.legend(); plt.tight_layout(); plt.show()
 
 
# %% [markdown]
# ## FOLDS · etapas 4 a 7
 
# %% ---------------------------------------------------------------------------
# ETAPA 4 · índices do K-Fold
# Gerador: cada `yield` entrega um par (train_idx, val_idx) e pausa.
# -----------------------------------------------------------------------------
def meu_kfold(n_amostras, k=5, shuffle=False, random_state=None):
    """Gera K pares (train_idx, val_idx) cortando posições seguidas."""
    # 1 validar K
    if not 2 <= k <= n_amostras:
        raise ValueError("k precisa estar entre 2 e n_amostras")
    # 2 criar a sequência de índices
    indices = np.arange(n_amostras)
    # 3 embaralhar (opcional) — embaralhamos ÍNDICES, nunca X e y separadamente
    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(indices)
    # 4 tamanho de cada fold: base n // k; os primeiros n % k folds ganham +1
    tamanhos = np.full(k, n_amostras // k)
    tamanhos[: n_amostras % k] += 1
 
    inicio = 0
    # 5 percorrer os folds
    for tamanho in tamanhos:
        fim = inicio + tamanho
        # 6 fold de validação
        val_idx = indices[inicio:fim]
        # 7 treino = todo o resto
        train_idx = np.concatenate([indices[:inicio], indices[fim:]])
        # 8 entregar o par e pausar
        yield train_idx, val_idx
        inicio = fim   # sem isso, o primeiro fold se repetiria K vezes
 
 
# Exemplo pequeno dos slides (n = 10, K = 3 -> tamanhos [4 3 3])
print("\nEtapa 4 · meu_kfold(10, k=3)")
for fold, (train_idx, val_idx) in enumerate(meu_kfold(10, k=3), start=1):
    print(f"Fold {fold} | treino: {train_idx} | val: {val_idx}")
 
# Verificações
vals = np.concatenate([v for _, v in meu_kfold(10, k=3)])
print("Cada índice é validação uma vez:", np.array_equal(np.sort(vals), np.arange(10)))
print("Treino e validação não se cruzam:",
      all(len(np.intersect1d(t, v)) == 0 for t, v in meu_kfold(10, k=3)))
 
 
# %% ---------------------------------------------------------------------------
# ETAPA 6 · Stratified K-Fold
# (vem antes da etapa 5 no código porque a visualização compara as três formas)
# Reparte os índices de CADA classe entre os K folds -> cada fold herda a
# proporção do dataset.
# -----------------------------------------------------------------------------
def meu_stratified_kfold(y, k=5, shuffle=False, random_state=None):
    """Gera K pares (train_idx, val_idx) preservando a proporção das classes."""
    y = np.asarray(y)
    classes, contagens = np.unique(y, return_counts=True)        # 1
    if k < 2 or contagens.min() < k:
        raise ValueError("cada classe precisa ter pelo menos k amostras")
    rng = np.random.default_rng(random_state)
    partes_por_fold = [[] for _ in range(k)]                     # 2
    for c in classes:
        idx_c = np.flatnonzero(y == c)                           # 3 posições da classe c
        if shuffle:
            idx_c = rng.permutation(idx_c)                       #   embaralha dentro da classe
        for f, parte in enumerate(np.array_split(idx_c, k)):    # 4 parte f -> fold f
            partes_por_fold[f].append(parte)
    todos = np.arange(len(y))
    for partes in partes_por_fold:
        val_idx = np.sort(np.concatenate(partes))                # 5 validação
        train_idx = np.setdiff1d(todos, val_idx)                 # 6 treino = complemento
        yield train_idx, val_idx                                 # 7
 
 
# Exemplo dos slides: y = [0 0 0 0 0 0 0 0 1 1 1 1], K = 4
y_mini = np.array([0] * 8 + [1] * 4)
print("\nEtapa 6 · meu_stratified_kfold(y_mini, k=4)")
for fold, (train_idx, val_idx) in enumerate(meu_stratified_kfold(y_mini, k=4), start=1):
    print(f"Fold {fold} | treino: {train_idx} | val: {val_idx}")
 
 
# %% ---------------------------------------------------------------------------
# ETAPA 5 · visualizar os folds
# Cada linha = uma rodada; cada coluna = uma amostra (índice 0..99).
# Verde = treino, branco = validação. Faixa de baixo = classe de cada amostra.
# -----------------------------------------------------------------------------
def matriz_folds(gerador, n):
    """Matriz (K x n): 1 onde a amostra é validação na rodada, 0 onde é treino."""
    linhas = []
    for _, val_idx in gerador:
        linha = np.zeros(n, dtype=int)
        linha[val_idx] = 1
        linhas.append(linha)
    return np.array(linhas)
 
 
estrategias = {
    "meu_kfold · shuffle=False": meu_kfold(len(y), k=5),
    "meu_kfold · shuffle=True, rs=42": meu_kfold(len(y), k=5, shuffle=True, random_state=42),
    "meu_stratified_kfold · shuffle=True, rs=42":
        meu_stratified_kfold(y, k=5, shuffle=True, random_state=42),
}
 
cmap_folds = ListedColormap(["#1f9e5f", "#f2f5f2"])   # treino, validação
cmap_classe = ListedColormap(["#3b4a44", "#3cff8f"])  # classe 0, classe 1
 
fig, axes = plt.subplots(3, 1, figsize=(11, 7))
for ax, (nome, gerador) in zip(axes, estrategias.items()):
    M = matriz_folds(gerador, len(y))
    ax.imshow(M, cmap=cmap_folds, aspect="auto", interpolation="nearest")
    # faixa com a classe de cada amostra logo abaixo
    ax.imshow(y[None, :], cmap=cmap_classe, aspect="auto", interpolation="nearest",
              extent=(-0.5, len(y) - 0.5, 5.4, 4.6))
    ax.set_ylim(5.5, -0.5)
    ax.set_yticks(list(range(5)) + [5])
    ax.set_yticklabels([f"R{i}" for i in range(1, 6)] + ["classe"])
    ax.set_title(nome, loc="left", fontsize=10)
    ax.set_xlabel("índice da amostra")
fig.legend(handles=[Patch(color="#1f9e5f", label="treino"),
                    Patch(color="#f2f5f2", label="validação", ec="gray"),
                    Patch(color="#3cff8f", label="classe 1")],
           loc="upper right", ncol=3)
fig.suptitle("Etapa 5 · visualização dos folds (n = 100, K = 5)", x=0.01, ha="left")
plt.tight_layout(); plt.show()
 
 
# %% ---------------------------------------------------------------------------
# ETAPA 7 · classes por fold
# np.bincount(y[val_idx], minlength=2) -> [negativos, positivos] na validação
# -----------------------------------------------------------------------------
def contar_classes(gerador):
    for fold, (train_idx, val_idx) in enumerate(gerador, start=1):
        print(fold, np.bincount(y[val_idx], minlength=2))
 
 
print("\nEtapa 7 · K-Fold sem shuffle")
contar_classes(meu_kfold(100, k=5))
print("Etapa 7 · K-Fold com shuffle (42)")
contar_classes(meu_kfold(100, k=5, shuffle=True, random_state=42))
print("Etapa 7 · Stratified K-Fold com shuffle (42)")
contar_classes(meu_stratified_kfold(y, k=5, shuffle=True, random_state=42))
 
 
# %% [markdown]
# ## AVALIAÇÃO · etapas 8 a 11
 
# %% ---------------------------------------------------------------------------
# ETAPA 8 · modelo de referência (modelo novo por rodada)
# fit:     centroide de cada classe = média das linhas daquela classe
# predict: distância a cada centroide; vence o mais próximo (argmin)
# -----------------------------------------------------------------------------
class CentroideMaisProximo:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroides_ = np.array(
            [X[y == c].mean(axis=0) for c in self.classes_])
        return self
 
    def predict(self, X):
        dist = np.stack([np.linalg.norm(X - c, axis=1)
                         for c in self.centroides_], axis=1)
        return self.classes_[np.argmin(dist, axis=1)]
 
 
def acuracia(y_true, y_pred):
    """Fração de acertos (aula 6.5)."""
    return np.mean(y_true == y_pred)
 
 
# Fábrica: sem parênteses passamos a classe; com parênteses criamos um modelo.
fabrica = CentroideMaisProximo
m1, m2 = fabrica(), fabrica()
print("\nEtapa 8 · m1 is m2 ->", m1 is m2)   # False: cada chamada = objeto novo
 
 
# %% ---------------------------------------------------------------------------
# ETAPA 9 · accuracy por fold (laço escrito à mão)
# -----------------------------------------------------------------------------
scores = []
for fold, (train_idx, val_idx) in enumerate(
        meu_stratified_kfold(y, k=5, shuffle=True, random_state=42), start=1):
    modelo = CentroideMaisProximo()                  # 1 modelo novo
    modelo.fit(X[train_idx], y[train_idx])           # 2 treina só com o treino do fold
    y_pred = modelo.predict(X[val_idx])              # 3 prevê na validação
    s = acuracia(y[val_idx], y_pred)                 # 4 compara com y[val_idx]
    scores.append(s)                                 # 5 guarda
    print(f"Etapa 9 · fold {fold}: treino={len(train_idx)} val={len(val_idx)} acc={s:.2f}")
scores = np.array(scores)
 
 
# %% ---------------------------------------------------------------------------
# ETAPA 10 · média e desvio-padrão
# média  s̄ = (1/K) Σ s_k
# desvio σ = sqrt( (1/K) Σ (s_k − s̄)² )   -> populacional, ddof=0 (padrão do np.std)
# -----------------------------------------------------------------------------
K = len(scores)
media_manual = scores.sum() / K
desvio_manual = np.sqrt(((scores - media_manual) ** 2).sum() / K)
 
print("\nEtapa 10 · scores:", scores)
print(f"média manual = {media_manual:.3f} | np.mean = {np.mean(scores):.3f}")
print(f"desvio manual (ddof=0) = {desvio_manual:.3f} | np.std = {np.std(scores):.3f}")
print(f"desvio amostral (ddof=1) = {np.std(scores, ddof=1):.3f}")
 
# Conferindo com os valores ilustrativos dos slides
ilustr = np.array([0.80, 0.85, 0.90, 0.75, 0.85])
print(f"slides: média = {ilustr.mean():.2f} | σ = {ilustr.std():.3f} "
      f"| σ (ddof=1) = {ilustr.std(ddof=1):.3f}")
 
 
# %% ---------------------------------------------------------------------------
# ETAPA 11 · meu_cross_val_score
# Devolve um score por fold; média e desvio ficam com quem chamou.
# -----------------------------------------------------------------------------
def meu_cross_val_score(modelo_factory, X, y, folds, metrica):
    scores = []
    for train_idx, val_idx in folds:
        X_treino, y_treino = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        modelo = modelo_factory()            # novo a cada rodada
        modelo.fit(X_treino, y_treino)
        y_pred = modelo.predict(X_val)
        scores.append(metrica(y_val, y_pred))
    return np.array(scores)
 
 
# Mesmo modelo, três formas de dividir
formas = {
    "K-Fold sem shuffle": meu_kfold(len(y), k=5),
    "K-Fold, shuffle (42)": meu_kfold(len(y), k=5, shuffle=True, random_state=42),
    "Stratified, shuffle (42)": meu_stratified_kfold(y, k=5, shuffle=True, random_state=42),
}
 
print("\nEtapa 11 · meu_cross_val_score")
resultados = {}
for nome, folds in formas.items():
    sc = meu_cross_val_score(CentroideMaisProximo, X, y, folds, acuracia)
    resultados[nome] = sc
    print(f"{nome:26s} {sc}  {sc.mean():.3f} ± {sc.std():.3f}")
 
# Referência: quem responde sempre 0 acerta 90/100
print(f"Baseline 'sempre 0': {acuracia(y, np.zeros_like(y)):.2f}")
 
# Gráfico dos scores por fold
plt.figure(figsize=(7, 4))
for nome, sc in resultados.items():
    plt.plot(range(1, 6), sc, marker="o", label=f"{nome} · {sc.mean():.3f} ± {sc.std():.3f}")
plt.axhline(0.90, ls="--", c="gray", lw=1, label="baseline 'sempre 0' (0,90)")
plt.xticks(range(1, 6)); plt.ylim(0.4, 1.02)
plt.xlabel("fold"); plt.ylabel("accuracy")
plt.title("Etapa 11 · scores por fold")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()
 
 
# %% [markdown]
# ## COMPARAÇÃO · etapa 12
 
# %% ---------------------------------------------------------------------------
# ETAPA 12 · conferir com o scikit-learn
# -----------------------------------------------------------------------------
print("\nEtapa 12 · comparação com o scikit-learn")
 
# 1 mesmos índices sem shuffle?
iguais = all(np.array_equal(v1, v2) for (_, v1), (_, v2)
             in zip(meu_kfold(100, k=5), KFold(5).split(X)))
print("1 · meu_kfold == KFold (sem shuffle):", iguais)
 
# 2 nossas divisões no cross_val_score oficial
meus_folds = list(meu_stratified_kfold(y, k=5, shuffle=True, random_state=42))
sc2 = cross_val_score(NearestCentroid(), X, y, cv=meus_folds)
print("2 · nossos folds no cross_val_score:", sc2,
      "| idêntico ao nosso:", np.allclose(sc2, resultados["Stratified, shuffle (42)"]))
 
# 3 divisões do próprio sklearn (outro gerador aleatório e outra regra de alocação)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
sc3 = cross_val_score(NearestCentroid(), X, y, cv=cv)
print(f"3 · StratifiedKFold do sklearn: {sc3}  {sc3.mean():.3f} ± {sc3.std():.3f}")
for fold, (_, v) in enumerate(cv.split(X, y), start=1):
    print("   fold", fold, np.bincount(y[v], minlength=2))
 
# Extra · KFold sem shuffle no sklearn: o fold 5 treina só com a classe 0
with warnings.catch_warnings():          # o sklearn avisa (com traceback) que 1 fit falhou
    warnings.simplefilter("ignore")
    sc4 = cross_val_score(NearestCentroid(), X, y, cv=KFold(5), error_score=np.nan)
print("Extra · KFold sem shuffle no sklearn:", sc4,
      "(nan = NearestCentroid recusa treinar com uma única classe)")
 