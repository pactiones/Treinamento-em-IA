import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
m_por_classe = 50

X0 = rng.normal(loc=[-1, -1], scale=1.2, size=(m_por_classe, 2))
X1 = rng.normal(loc=[1, 1], scale=1.2, size=(m_por_classe, 2))

X_dados = np.vstack([X0, X1])
y = np.concatenate([
    np.zeros(m_por_classe),
    np.ones(m_por_classe)
]).astype(int)

idx = rng.permutation(len(y))
corte = int(0.8 * len(y))

treino, teste = idx[:corte], idx[corte:]

X_treino, y_treino = X_dados[treino], y[treino]
X_teste, y_teste = X_dados[teste], y[teste]

class RegressaoLogistica:
    def __init__(self, alpha=0.1, epocas=1000):
        self.alpha = alpha
        self.epocas = epocas
        self.beta = None
        self.historico = []

    def _sigmoide(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _com_intercepto(self, X):
        uns = np.ones((X.shape[0], 1))
        return np.hstack([uns, X])

    def _custo(self, p, y, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y):
        Xb = self._com_intercepto(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)

        if Xb.shape[0] != y.shape[0]:
            raise ValueError("X e y devem ter o mesmo numero de exemplos.")

        self.beta = np.zeros(Xb.shape[1])
        self.historico = []
        m = len(y)

        for _ in range(self.epocas):
            p = self._sigmoide(Xb @ self.beta)
            self.historico.append(self._custo(p, y))

            gradiente = Xb.T @ (p - y) / m
            self.beta -= self.alpha * gradiente

        return self

    def predict_proba(self, X):
        Xb = self._com_intercepto(np.asarray(X, dtype=float))
        return self._sigmoide(Xb @ self.beta)

    def predict(self, X, limiar=0.5):
        return (self.predict_proba(X) >= limiar).astype(int)

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)
        return np.mean(self.predict(X) == y)

# Visualizar os dados antes do treinamento
plt.scatter(X_dados[y == 0, 0], X_dados[y == 0, 1], label="classe 0")
plt.scatter(X_dados[y == 1, 0], X_dados[y == 1, 1], label="classe 1")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Dados gerados")
plt.legend()
plt.show()

# Criar e treinar o modelo
modelo = RegressaoLogistica(alpha=0.1, epocas=1000)
modelo.fit(X_treino, y_treino)

# Mostrar o que foi aprendido e como foi o resultado
print("Coeficientes [beta_0, beta_1, beta_2]:", modelo.beta.round(3))
print("Épocas registradas:", len(modelo.historico))
print("Primeiro custo:", round(modelo.historico[0], 4))
print("Último custo:", round(modelo.historico[-1], 4))
print(f"Acurácia no treino: {modelo.score(X_treino, y_treino):.3f}")
print(f"Acurácia no teste:  {modelo.score(X_teste, y_teste):.3f}")

# Ver algumas previsões
p_teste = modelo.predict_proba(X_teste)
y_pred = modelo.predict(X_teste)

print("Probabilidades previstas:", p_teste[:5].round(3))
print("Classes previstas:", y_pred[:5])
print("Classes corretas:", y_teste[:5])

# Desenhar a fronteira aprendida pelo modelo
plt.scatter(X_dados[y == 0, 0], X_dados[y == 0, 1], label="classe 0")
plt.scatter(X_dados[y == 1, 0], X_dados[y == 1, 1], label="classe 1")

beta_0, beta_1, beta_2 = modelo.beta
x1 = np.linspace(X_dados[:, 0].min(), X_dados[:, 0].max(), 100)
x2 = -(beta_0 + beta_1 * x1) / beta_2

plt.plot(x1, x2, "k--", label="fronteira p = 0,5")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Fronteira de decisão aprendida")
plt.legend()
plt.show()

# Desenhar a evolução do custo durante o treinamento
plt.plot(modelo.historico)
plt.xlabel("Época")
plt.ylabel("Custo J(beta)")
plt.title("Curva de custo")
plt.show()