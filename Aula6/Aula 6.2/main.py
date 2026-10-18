import numpy as np
from sklearn.datasets import fetch_california_housing

dados = fetch_california_housing()
X, y = dados.data, dados.target

print(X.shape, y.shape) # (20640, 8) (20640,)
print(dados.feature_names) # ['MedInc', 'HouseAge', ...]

#Adicionar a colunas de ls
n = X.shape[0]
Xb = np.column_stack([np.ones(n), X])

print(Xb.shape) # (20640, 9)
print(Xb[0, :3]) # [1.  8.3252 41.0 ]

#Calcular β pelos três metódos  
A = Xb.T @ Xb #(9, 9)
b = Xb.T @ y #(9,)

beta_inv = np.linalg.inv(A) @ b  #Inversa da matriz A
beta_solve = np.linalg.solve(A, b) #Resolver o sistema linear A * beta = b
beta_pinv = np.linalg.pinv(Xb) @ y #Usando a pseudo-inversa para calcular beta

print(np.allclose(beta_solve, beta_pinv)) #Conferir se A^-1 * b é igual a beta_solve
print(np.abs(beta_inv - beta_pinv).max()) #Conferir a diferença máxima entre beta_inv e beta_pinv

y_hat = Xb @ beta_pinv #(20640,)

r2 = 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)
print("R²: ", r2)

#Conferir com o LinearRregression
from sklearn.linear_model import LinearRegression

lr = LinearRegression() #fit_intercept=True - Criar o modelo de regressão linear
lr.fit(X, y) #X SEM a coluna de 1s - Treinar o modelo com os dados
beta_sk = np.r_[lr.intercept_, lr.coef_] #Obter os coeficientes do modelo treinado

print(np.allclose(beta_pinv, beta_sk)) # True
print(lr.score(X, y)) #≈0.606

class MinimosQuadrados:
    def __init__(self, metodo="pinv"):
        if metodo not in ("inv", "solve", "pinv"):
            raise ValueError("metodo deve ser 'inv', 'solve' ou 'pinv'")
        self.metodo = metodo
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Adiciona uma coluna de 1s para calcular o intercepto
        n = X.shape[0]
        Xb = np.column_stack([np.ones(n), X])

        if self.metodo == "inv":
            A = Xb.T @ Xb
            b = Xb.T @ y
            self.coef_ = np.linalg.inv(A) @ b

        elif self.metodo == "solve":
            A = Xb.T @ Xb
            b = Xb.T @ y
            self.coef_ = np.linalg.solve(A, b)

        else:  # "pinv"
            self.coef_ = np.linalg.pinv(Xb) @ y

        return self

    def predict(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        Xb = np.column_stack([np.ones(n), X])
        return Xb @ self.coef_

    def score(self, X, y):
        y = np.asarray(y)
        y_hat = self.predict(X)

        soma_erros = np.sum((y - y_hat) ** 2)
        soma_total = np.sum((y - y.mean()) ** 2)

        return 1 - soma_erros / soma_total

modelo = MinimosQuadrados(metodo="pinv")
modelo.fit(X, y)

previsoes = modelo.predict(X)

print("Coeficientes do modelo:")
print(modelo.coef_)

print("Primeiras 5 previsões:")
print(previsoes[:5])

print("Pontuação do R²: ", modelo.score(X, y))

for metodo in ("inv", "solve", "pinv"):
    modelo = MinimosQuadrados(metodo=metodo).fit(X, y)
    print(metodo,"-", modelo.score(X, y))

print("Valores duplicados:")
X_duplicada = np.column_stack([X, X[:, 0]])

modelo_pinv = MinimosQuadrados(metodo="pinv").fit(X_duplicada, y)
print(modelo_pinv.score(X_duplicada, y))