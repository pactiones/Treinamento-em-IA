# Etapa 1: Imports
import numpy as np
import matplotlib.pyplot as plt

# Etapa 2: Tabela XOR (Dados de Entrada X e Rótulos Y)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

Y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

print("Formato de X:", X.shape)  # Formato de X: (4, 2)
print("Formato de Y:", Y.shape)  # Formato de Y: (4, 1)

# Etapa 3: Visualização do XOR no Espaço Original
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap="coolwarm", s=300, edgecolors="black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("XOR: classes em diagonais opostas")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# Etapa 4 e 5: Perceptron Único (Função de Treinamento)
def treinar_perceptron(X, y, taxa_aprendizado=0.1, max_epocas=100):
    pesos = np.zeros(X.shape[1])  # np.zeros: pesos começam em zero
    bias = 0.0
    for epoca in range(max_epocas):
        erros = 0
        for xi, yi in zip(X, y):
            z = np.dot(xi, pesos) + bias  # z = w^T * x_i + b
            previsao = 1 if z >= 0 else -1  # Função sinal (degrau binário -1/+1)
            if previsao != yi:
                pesos += taxa_aprendizado * yi * xi  # w <- w + eta * y_i * x_i
                bias += taxa_aprendizado * yi        # b <- b + eta * y_i
                erros += 1
        if erros == 0:  # Época sem erros encerra o laço
            break
    return pesos, bias, epoca + 1

# Etapa 6: Perceptron tentando aprender a função OR (Linearmente separável)
Y_or = np.array([-1, 1, 1, 1])
pesos_or, bias_or, epocas_or = treinar_perceptron(X, Y_or)
z_or = X @ pesos_or + bias_or
prev_or = np.where(z_or >= 0, 1, -1)

print("\n--- TESTE PERCEPTRON NO OR ---")
print("Pesos, Bias, Épocas:", pesos_or, bias_or, epocas_or)  # Output ref: [0.1 0.1] -0.1 4
print("Previsto vs Esperado:", prev_or, Y_or)               # Output ref: [-1 1 1 1] [-1 1 1 1]

# Etapa 7: Perceptron tentando aprender o XOR (Não separável linearmente)
Y_xor_pm = np.array([-1, 1, 1, -1])
p_x, b_x, ep_x = treinar_perceptron(X, Y_xor_pm)
prev_xor = np.where(X @ p_x + b_x >= 0, 1, -1)

print("\n--- TESTE PERCEPTRON NO XOR ---")
print("Épocas executadas (max_epocas):", ep_x)               # Output ref: 100
print("Previsto vs Esperado:", prev_xor, Y_xor_pm)          # Output ref: [ 1  1 -1 -1] [-1  1  1 -1] (2 de 4 acertos)

# Etapa 8: Ativação Sigmoid
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Etapa 9: Inicialização dos Parâmetros da MLP (Arquitetura 2 -> 4 -> 1)
rng = np.random.default_rng(42)
n_entradas = 2
n_ocultos = 4
n_saidas = 1

params = {
    "W1": rng.normal(0, 0.8, size=(n_entradas, n_ocultos)),
    "b1": np.zeros((1, n_ocultos)),
    "W2": rng.normal(0, 0.8, size=(n_ocultos, n_saidas)),
    "b2": np.zeros((1, n_saidas))
}

print("\n--- FORMATO DOS PARÂMETROS DA MLP ---")
for k, v in params.items():
    print(k, v.shape)

# Etapa 10: Forward Propagation
def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = sigmoid(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)
    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache

# Etapa 11: Função de Perda (Binary Cross-Entropy)
def calcular_loss(Y, probabilidades):
    eps = 1e-12
    p = np.clip(probabilidades, eps, 1 - eps)
    loss = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    return float(loss)

A2_inicial, _ = forward(X, params)
loss_inicial = calcular_loss(Y, A2_inicial)
print("\nPrevisões iniciais A2:", A2_inicial.ravel())  # Ref: [0.578 0.613 0.660 0.682]
print("Perda inicial:", loss_inicial)                  # Ref: 0.7285 (acima de log(2) ≈ 0.693)

# Etapa 12 e 13: Backpropagation (Retropropagação)
def backward(Y, params, cache):
    m = Y.shape[0]
    X_in = cache["X"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = (A2 - Y) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * A1 * (1 - A1)  # Derivada da sigmoid em relação a Z1
    dW1 = X_in.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }

grads = backward(Y, params, forward(X, params)[1])
print("\n--- FORMATO DOS GRADIENTES ---")
for k in grads:
    print(k, grads[k].shape, "igual ao parâmetro:", params[k].shape)

# Etapa 14: Atualização de Parâmetros (Gradient Descent)
def update_parameters(params, gradients, lr):
    for key in params:
        params[key] -= lr * gradients[key]
    return params

# Etapas 15 e 16: Laço de Treinamento Full Batch (20.000 épocas)
learning_rate = 0.5
epochs = 20000
historico_loss = []

for epoca in range(epochs):
    A2, cache = forward(X, params)
    loss = calcular_loss(Y, A2)
    gradients = backward(Y, params, cache)
    params = update_parameters(params, gradients, learning_rate)
    if epoca % 100 == 0:
        historico_loss.append(loss)

print("\n--- RESULTADO DO TREINAMENTO ---")
print("Perda inicial:", historico_loss[0])                      # Ref: 0.7285
print("Perda final:", calcular_loss(Y, forward(X, params)[0]))  # Ref: 0.0007

# Etapa 17: Curva de Perda (Binary Cross-Entropy vs Época)
epocas_reg = np.arange(len(historico_loss)) * 100
plt.figure(figsize=(7, 4))
plt.plot(epocas_reg, historico_loss, color="#10b981", linewidth=2)
plt.xlabel("Época")
plt.ylabel("Binary Cross-Entropy")
plt.title("Curva de Aprendizado da MLP")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# Etapa 18: Previsões Finais da Rede
probabilidades, _ = forward(X, params)
previsoes_binarias = (probabilidades >= 0.5).astype(int)

print("\n--- TABELA DE VERIFICAÇÃO FINAL ---")
print("X (Entradas)   | A2 (Probabilidade) | Classe Prevista | Esperado")
print("-" * 62)
for i in range(len(X)):
    print(f"{X[i]}       | {probabilidades[i][0]:.4f}             | {previsoes_binarias[i][0]}               | {int(Y[i][0])}")

print("\nPrevisões binárias finais:", previsoes_binarias.ravel())  # Output ref: [0 1 1 0]

# Etapa 19: Região de Decisão da Rede Treinada (Fronteira Curva de Decisão)
x1_grid = np.linspace(-0.5, 1.5, 300)
x2_grid = np.linspace(-0.5, 1.5, 300)
G1, G2 = np.meshgrid(x1_grid, x2_grid)                      # (300, 300)
grade = np.column_stack((G1.ravel(), G2.ravel()))           # (90000, 2)

saida_grade, _ = forward(grade, params)                     # (90000, 1)
saida_grid = saida_grade.reshape(G1.shape)                  # (300, 300)

plt.figure(figsize=(6, 5))
plt.contourf(G1, G2, saida_grid, levels=20, cmap="coolwarm", alpha=0.8)
plt.contour(G1, G2, saida_grid, levels=[0.5], colors="black", linewidths=2)
plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap="coolwarm", s=200, edgecolors="black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Fronteira de Decisão Curva Aprendida para o XOR")
plt.show()