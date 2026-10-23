# Distância euclidiana em NumPy

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split)
from sklearn.neighbors import KNeighborsClassifier

a = np.array([1, 2])  # ponto A
b = np.array([4, 6])  # ponto B

distancia = np.sqrt(np.sum((a - b) ** 2)) #euclidiana

# Distância de Manhattan

distancia = np.sum(np.abs(a - b))  # 7
# a - b  -> [-3, -4]
# np.abs -> [ 3,  4]
# np.sum -> 7

# Votação majoritária

vizinhos = np.array([0, 1, 1, 0, 1])

votos = np.bincount(vizinhos)  # [2, 3]
classe = votos.argmax()  # 1

# Alternativa para rótulos quaisquer
classes, contagens = np.unique(
    vizinhos, return_counts=True
)
# classes = [0, 1]  contagens = [2, 3]
classe = classes[np.argmax(contagens)]  # 1

# K pequeno e K grande — escolha de K por validação cruzada

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

X_train = np.array([[1, 2], [4, 6], [2, 1], [5, 5], [3, 3]])
y_train = np.array([0, 1, 0, 1, 0])


# K é inteiro positivo: 1, 2, 3...
# K ≤ número de amostras de treino

# Escalonar sem vazamento de dados

X_train, X_test, y_train, y_test = \
    train_test_split(X_train, y_train, test_size=0.25,
                     random_state=42)

scaler = StandardScaler()
# fit: aprende μ e σ só no treino
X_train_esc = scaler.fit_transform(X_train)
# transform: reutiliza μ e σ do treino
X_test_esc = scaler.transform(X_test)

# Escalonar sem vazamento de dados

X_train, X_test, y_train, y_test = \
    train_test_split(X_train, y_train, test_size=0.25,
                     random_state=42)

scaler = StandardScaler()
# fit: aprende μ e σ só no treino
X_train_esc = scaler.fit_transform(X_train)
# transform: reutiliza μ e σ do treino
X_test_esc = scaler.transform(X_test)

# Versão manual com NumPy

mu = X_train.mean(axis=0)  # média por coluna
sigma = X_train.std(axis=0)  # desvio por coluna
X_train_esc = (X_train - mu) / sigma
X_test_esc = (X_test - mu) / sigma

# KNN com loops: distâncias de uma amostra

def distancias_loop(X_train, x):
    distancias = np.zeros(len(X_train))
    for i in range(len(X_train)):
        diff = X_train[i] - x
        distancias[i] = np.sqrt(np.sum(diff ** 2))
    return distancias

# Dados de treino e amostra de consulta

X_train = np.array([[1, 2], [4, 6], [2, 1],
                    [5, 5], [3, 3]])
y_train = np.array([0, 1, 0, 1, 0])
x_novo = np.array([4, 5])

# KNN com loops: vizinhos e votação

dist = distancias_loop(X_train, x_novo)
k = 3
indices = np.argsort(dist, kind='stable')[:k]
vizinhos = y_train[indices]
votos = np.bincount(vizinhos)
classe = votos.argmax()

# KNN com loops: várias amostras

def predict_loop(X_train, y_train, X_test, k=3):
    previsoes = []
    for x in X_test:  # loop externo
        distancias = np.zeros(len(X_train))
        for i in range(len(X_train)):  # loop interno
            dif = X_train[i] - x
            distancias[i] = np.sqrt(np.sum(dif ** 2))
        indices = np.argsort(distancias, kind='stable')[:k]
        votos = np.bincount(y_train[indices])
        previsoes.append(votos.argmax())
    return np.array(previsoes)


X_test = np.array([[1, 1], [4, 5], [3, 2]])
predict_loop(X_train, y_train, X_test)  # array([0, 1, 0])

# Dimensões das matrizes

X_train.shape  # (5, 2)
X_train[3]  # [5, 5] — linha 3
X_train[:, 0]  # [1, 4, 2, 5, 3] — coluna 0
X_train.mean(axis=0)  # [3. , 3.4] — média por coluna
x = np.array([4, 5])  # shape (2,)
x.reshape(1, 2)  # (1, 2): 1 linha, 2 colunas

# axis=0 percorre as linhas (amostras)
# axis=1 percorre as colunas (características)
# Uma operação com axis=a reduz (elimina) o eixo a do resultado.

# Loops Python

lista = [1.0, 2.0, 3.0, 4.0]
dobro = []
for x in lista:  # 4 iterações
    dobro.append(x * 2)  # interpretadas uma a uma

# Operação vetorizada

arr = np.array([1.0, 2.0, 3.0, 4.0])
dobro = arr * 2  # 1 chamada ao NumPy
# [2., 4., 6., 8.]

# Criando eixos com None

A = X_test[:, None, :]  # (3, 2) -> (3, 1, 2)
B = X_train[None, :, :]  # (5, 2) -> (1, 5, 2)
diferencas = A - B  # (3, 5, 2)

# np.newaxis é o mesmo que None
X_test[:, np.newaxis, :].shape  # (3, 1, 2)

# Matriz de distâncias

quadrados = diferencas ** 2  # (3, 5, 2)
somas = np.sum(quadrados, axis=2)  # (3, 5)
distancias = np.sqrt(somas)  # (3, 5)

# Manhattan
distancias_man = np.sum(np.abs(diferencas), axis=2)

# Selecionando os K menores valores de uma linha

d = np.array([4.5, 1.2, 3.8, 0.9])

np.sort(d)  # [0.9, 1.2, 3.8, 4.5]
np.argsort(d)  # [3, 1, 2, 0]
np.argsort(d)[:2]  # [3, 1]
np.argpartition(d, 1)[:2]  # [3, 1] ou [1, 3]

# sort devolve valores; argsort devolve posições.
# Só as posições levam aos rótulos em y_train.

# Seleção dos K menores valores em uma matriz de distâncias

indices = np.argsort(distancias, axis=1,
                     kind='stable')[:, :k]
# [[0, 2, 4],
#  [1, 3, 4],
#  [4, 2, 0]]

indices = np.argpartition(distancias, k - 1,
                          axis=1)[:, :k]

# axis=1 ordena cada linha (cada amostra de teste)
# [:, :k] seleciona as k primeiras colunas de todas as linhas
# argpartition separa os k menores sem ordenar tudo;
# a ordem interna não é garantida

# A classe KNNDoZero: estrutura

class KNNDoZero:
    def __init__(self, k=3, metrica='euclidiana'):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError('k deve ser inteiro positivo')
        if metrica not in ('euclidiana', 'manhattan'):
            raise ValueError('métrica inválida')
        self.k, self.metrica = k, metrica

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        if self.k > len(self.X_train):
            raise ValueError('k > n_train')
        self.classes, self.y_cod = np.unique(
            np.asarray(y), return_inverse=True)
        return self

# A classe KNNDoZero: predict e score

class KNNDoZero:  # continuação
    def _distancias(self, X):
        dif = X[:, None, :] - self.X_train[None, :, :]
        if self.metrica == 'euclidiana':
            return np.sqrt(np.sum(dif ** 2, axis=2))
        return np.sum(np.abs(dif), axis=2)

    def predict(self, X):
        d = self._distancias(np.asarray(X, dtype=float))
        indices = np.argsort(d, axis=1, kind='stable')
        indices = indices[:, :self.k]
        viz = self.y_cod[indices]
        venc = [np.bincount(v).argmax() for v in viz]
        return self.classes[venc]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))

# Complexidade e memória

def distancias_em_blocos(X_test, X_train, tam=100):
    blocos = []
    for ini in range(0, len(X_test), tam):
        lote = X_test[ini:ini + tam]
        dif = lote[:, None, :] - X_train[None, :, :]
        d = np.sqrt(np.sum(dif ** 2, axis=2))
        blocos.append(d)
    return np.vstack(blocos)  # (n_test, n_train)

# Cálculo das distâncias:
# O(n_test × n_train × n_features)
# A seleção dos vizinhos soma mais custo.
# Broadcasting pode ser rápido, mas usa memória para os arrays intermediários.

# Loop vs Broadcasting — A: Loops

def distancias_loops(X_test, X_train):
    n_test, n_train = len(X_test), len(X_train)
    D = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            D[i, j] = np.sqrt(np.sum((X_test[i] - X_train[j]) ** 2))
    return D

# Loop vs Broadcasting — B: Broadcasting

def distancias_broadcasting(X_test, X_train):
    dif = X_test[:, None, :] - X_train[None, :, :]
    return np.sqrt(np.sum(dif ** 2, axis=2))

# Verificação: mesmos vizinhos

D_a = distancias_loops(X_test, X_train)
D_b = distancias_broadcasting(X_test, X_train)
np.allclose(D_a, D_b)  # True

ia = np.argsort(D_a, axis=1, kind='stable')[:, :k]
ib = np.argsort(D_b, axis=1, kind='stable')[:, :k]
np.array_equal(ia, ib)  # True: mesmos vizinhos

# Medindo o tempo de execução

from time import perf_counter

def medir(funcao, *args, repeticoes=7):
    tempos = []
    for _ in range(repeticoes):
        inicio = perf_counter()
        funcao(*args)
        fim = perf_counter()
        tempos.append(fim - inicio)
    return np.median(tempos)

t_loops = medir(distancias_loops, X_test, X_train)
t_broad = medir(distancias_broadcasting, X_test, X_train)
speedup = t_loops / t_broad

# Speedup = tempo_loops / tempo_broadcasting
# Execute as medições com os mesmos dados nas duas versões.
# Os tempos dependem da máquina e são medidos durante a execução.

print("\n=== Distâncias ===")
print("Distância euclidiana:", np.sqrt(np.sum((a - b) ** 2)))
print("Distância de Manhattan:", np.sum(np.abs(a - b)))

print("\n=== Votação ===")
print("Rótulos dos vizinhos:", vizinhos)
print("Votos por classe:", votos)
print("Classe prevista:", classe)

print("\n=== Dados ===")
print("Formato de X_train:", X_train.shape)
print("Formato de X_test:", X_test.shape)
print("X_train:\n", X_train)
print("X_test:\n", X_test)

print("\n=== KNN com loops ===")
print("Amostra consultada:", x_novo)
print("Distâncias até as amostras de treino:", dist)
print(f"Índices dos {k} vizinhos mais próximos:", indices)
print("Rótulos dos vizinhos mais próximos:", y_train[indices])
print("Previsões para X_test:", predict_loop(X_train, y_train, X_test, k))

print("\n=== Matrizes de distâncias ===")
print("Distâncias calculadas com loops:\n", D_a)
print("Distâncias calculadas com broadcasting:\n", D_b)
print("As matrizes são iguais:", np.allclose(D_a, D_b))
print(f"Índices dos {k} vizinhos por amostra de teste:\n", ia)
print("Os vizinhos encontrados são iguais:", np.array_equal(ia, ib))

print("\n=== Tempo de execução ===")
print(f"Tempo mediano com loops: {t_loops:.8f} s")
print(f"Tempo mediano com broadcasting: {t_broad:.8f} s")
print(f"Speedup observado: {speedup:.2f}x")