import numpy as np
import matplotlib.pyplot as plt

#Passo 1: Gerar e visualziar os dados
rng = np.random.default_rng(42) #semente: os mesmos dados sempre
m_por_classe = 50
X0 = rng.normal(loc=[-1, -1], scale=1.2, size=(m_por_classe, 2)) #classe 0
X1 = rng.normal(loc=[1, 1], scale=1.2, size=(m_por_classe, 2)) #classe 1

X_dados = np.vstack([X0, X1]) #(100, 2) caracteristicas
y = np.concatenate([np.zeros(m_por_classe),
                    np.ones(m_por_classe)]) #(100,) rótulos 0/1

plt.scatter(X_dados[y == 0, 0], X_dados[y == 0, 1], label="classe 0")
plt.scatter(X_dados[y == 1, 0], X_dados[y == 1, 1], label="classe 1")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(); plt.show()

#Passo 2: Separar os dados e montar a matriz X
idx = rng.permutation(len(y)) #embaralha os índices
corte = int(0.8*len(y)) #80% treino, 20% teste
treino, teste = idx[:corte], idx[corte:]

uns = np.ones((X_dados.shape[0], 1)) #coluna de 1s: multiplica β0
X = np.hstack([uns, X_dados]) #(100, 3) = [1, x1, x2]

X_treino, y_treino = X[treino], y[treino]
X_teste, y_teste = X[teste], y[teste]
print(X_treino.shape, y_treino.shape)

beta = np.zeros(X.shape[1]) #[β0, β1, β2] = [0, 0, 0]
print(beta)

#Passo 3: Sigmoide, custo e gradiente
def sigmoide(z):
    return 1 / (1 + np.exp(-z)) #σ(z), aplicada elemento a elemento

def custo(p, y, eps = 1e-12):
    p = np.clip(p, eps, 1 - eps) #evita log(0)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def gradiente(X, p, y):
    m = len(y)
    return X.T @ (p - y) / m #∇ J(β): uma dervidada por β

#Passo 4: Treinar e guardar o histórico
alpha, epocas = 0.1, 1000
beta = np.zeros(X_treino.shape[1])
historico = []

for epoca in range(epocas):
    z = X_treino @ beta #combinação linear, (80,)
    p = sigmoide(z) #probabilidades, (80,)
    historico.append(custo(p, y_treino)) #guarda J(β) desta época
    beta = beta - alpha * gradiente(X_treino, p, y_treino)
    if epoca % 200 == 0:
        print(f"época {epoca:4d} custo {historico[-1]:.4f}")

print("beta:", beta.round(3))

#Passo 5: Probabilidades, classes e acurácia
p_teste = sigmoide(X_teste @ beta) #probabilidade da classe 1
y_pred = (p_teste >= 0.5).astype(int) #limiar 0,5 -> classes 0/1

acc_teste = np.mean(y_pred == y_teste) #fração de acertos
acc_treino = np.mean((sigmoide(X_treino @ beta) >= 0.5) == y_treino)

print(p_teste[:5].round(3), y_pred[:5], y_teste[:5].astype(int))
print(f"acurácia treino {acc_treino:.3f} | teste {acc_teste:.3f}")

#Passo 6: Desenhar a fronteira e a curva de custo
b0, b1, b2 = beta
x1 = np.linspace(X_dados[:, 0].min(), X_dados[:, 0].max(), 100)
x2 = -(b0 + b1 * x1) / b2 #reta onde z = 0(p = 0,5)

plt.scatter(X_dados[y == 0, 0], X_dados[y == 0, 1], label = "classe 0")
plt.scatter(X_dados[y == 1, 0], X_dados[y == 1, 1], label = "classe 1")
plt.plot(x1, x2, "k--", label="fronteira p = 0,5")
plt.legend(); plt.show()

plt.plot(historico) #curva de custo
plt.xlabel("época"); plt.ylabel("J(β)"); plt.show()