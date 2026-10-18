#Reescrever loops como operações matriciais

import numpy as np
import timeit

rng = np.random.default_rng(42)
n, d = 10_000, 50
X = rng.normal(size=(n, d)) #(n, d)
w = rng.normal(size=d) #(d,)
y = X @ w + rng.normal(scale=0.1, size=n)

#Produto interno
#Com laço
x = X[0] #(5,)
total = 0.0
for j in range(d):
    total += x[j] * w[j]
    print(total)

#Vetorizado
total_v = X[0] @ w #()
print(total_v)
np.isclose(total, total_v) #True

#Todas as previsões

y_hat = np.zeros(n) # (n,)
 
# Com laço
for i in range(n):
    for j in range(d):
        y_hat[i] += X[i, j] * w[j]
 
# Vetorizado
y_hat_v = X @ w # (n,)
print(np.allclose(y_hat, y_hat_v)) # True
tempo = timeit.timeit(lambda: X @ w, number=1000)
print("Tempo:", tempo)

#Distancia entre pares de pontos

#Com laço
A, B = X[:200], X[:50] # (200, 5), (50, 5)
D = np.zeros((200, 50)) # (200, 50)
for i in range(200):
    for j in range(50):
        D[i, j] = np.linalg.norm(A[i] - B[j])

#Vetorizado
#(200, 1, 5) - (1, 50, 5)
diff = A[:, None, :] - B[None, :, :]
#(200, 50, 5) -> (200, 50)
D_v = np.sqrt((diff ** 2).sum(axis=2))
np.allclose(D, D_v)    #true


#MSE e gradiente

#Com laço
w0 = np.zeros(d)
mse, grad = 0.0, np.zeros(d)
for i in range(n):
    erro = X[i] @ w0 - y[i]
    mse += erro**2 / n
    for j in range(d):
        grad[j] += 2 * erro * X[i, j] / n


#Vetorizado
erro = X @ w0 - y #(n,)
mse_v = np.mean(erro**2) #()
grad_v = 2 / n * X.T @ erro #(d,)
np.allclose(grad, grad_v) # True

#Conferir o gradiente e cronometrar

def mse_de(w):
    return np.mean((X @ w - y)**2)

def grad_numerico(f, w, h=1e-6):
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w)
        e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)
    return g

print(np.allclose(grad_v, grad_numerico(mse_de, w0))) # True

#Norma
def norma(x, ord=2):
    if ord == 1:
        #Soma dos valores absolutos
        return np.sum(np.abs(x))

    elif ord == 2:
        #Raiz da soma dos quadrados
        return np.sqrt(np.sum(x ** 2))

    else:
        raise ValueError("Deve ser somente 1 ou 2")

x = X[0]

print(norma(x, ord = 1))
print(norma(x, ord = 2))

#Distancias
def distancias(A, B):
    diferencas = A[:, None, :] - B[None, :, :]  
    dist = np.sqrt(np.sum(diferencas ** 2, axis = 2))
    return dist

A, B = X[:200], X[:50]
D = distancias(A, B)

print(D.shape)

#MSE
def mse(X, y, w):
    y_hat = X @ w
    erros = y_hat - y
    return np.mean(erros ** 2)

print(mse(X, y, np.zeros(d)))

#Grad_MSE
def grad_mse(X, y, w):
    n = X.shape[0]
    erros = X @ w - y
    return (2/n) * (X.T @ erros)

w0 = np.zeros(d)
print(grad_mse(X, y, w0))

#grad_numerico
def grad_numerico(f, w, h = 1e-6):
    g = np.zeros_like(w)

    for j in range (len(w)):
        e = np.zeros_like(w)
        e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)

    return g

def mse_de(w):
    return mse(X, y, w)

w0 = np.zeros(d)

print(grad_mse(X, y, w0))
print(grad_numerico(mse_de, w0))
print(np.allclose(grad_mse(X, y, w0), grad_numerico(mse_de, w0)))

from scipy.spatial.distance import cdist

x = X[0]
print(np.allclose(norma(x, ord=1), np.linalg.norm(x, ord=1)))
print(np.allclose(norma(x, ord=2), np.linalg.norm(x, ord=2)))

A, B = X[:200], X[:50]
print(np.allclose(distancias(A, B), cdist(A, B)))

for seed in range(5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(10_000, 50))
    w = rng.normal(size=50)
    y = X @ w + rng.normal(scale=0.1, size=10_000)

    w0 = np.zeros(50)

    f = lambda pesos: mse(X, y, pesos)
    resultado = np.allclose(
        grad_mse(X, y, w0),
        grad_numerico(f, w0)
    )
    print(seed, resultado)