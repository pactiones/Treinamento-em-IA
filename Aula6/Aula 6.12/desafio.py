#1: Perceptron para AND
import numpy as np
import matplotlib.pyplot as plt

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_and = np.array([-1, -1, -1, 1])

def treinar_perceptron(X, y, taxa_aprendizado=0.1, max_epocas=100):
    pesos = np.zeros(X.shape[1])
    bias = 0.0
    for epoca in range(max_epocas):
        erros = 0
        for xi, yi in zip(X, y):
            z = np.dot(xi, pesos) + bias
            previsao = 1 if z >= 0 else -1
            if previsao != yi:
                pesos += taxa_aprendizado * yi * xi
                bias += taxa_aprendizado * yi
                erros += 1
        if erros == 0:
            break
    return pesos, bias, epoca + 1

pesos_and, bias_and, epocas_and = treinar_perceptron(X_and, Y_and)
print("Perceptron no and: ")
print("Pesos:", pesos_and, "Bias:", bias_and, "Convergiu em épocas:", epocas_and)

#2: Mostrar por que ele não resolve XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_xor_perceptron = np.array([-1, 1, 1, -1])

pesos_xor, bias_xor, epocas_xor = treinar_perceptron(X_xor, Y_xor_perceptron)
z_xor = X_xor @ pesos_xor + bias_xor
previsoes_xor_p = np.where(z_xor >= 0, 1, -1)

print("\nPerceptron no XOR: ")
print("Épocas atingidas (máximo):", epocas_xor)
print("Previsto pelo Perceptron:", previsoes_xor_p)
print("Esperado do XOR:         ", Y_xor_perceptron)

#3: Construir uma MLP em NumPy
rng = np.random.default_rng(42)

def inicializar_mlp(n_entradas=2, n_ocultos=4, n_saidas=1):
    params = {
        "W1": rng.normal(0, 0.8, size=(n_entradas, n_ocultos)),
        "b1": np.zeros((1, n_ocultos)),
        "W2": rng.normal(0, 0.8, size=(n_ocultos, n_saidas)),
        "b2": np.zeros((1, n_saidas))
    }
    return params

params_mlp = inicializar_mlp(n_entradas=2, n_ocultos=4, n_saidas=1)

print("\nMLP: ")
for k, v in params_mlp.items():
    print(f"Matriz {k}: dimensão {v.shape}")

#4: Implementar forward
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

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

print("\nForward: ")
Y_xor_mlp = np.array([[0], [1], [1], [0]], dtype=float)
A2_teste, cache_teste = forward(X_xor, params_mlp)
print("Saída A2 antes do treino:\n", A2_teste.ravel())

#5: Implementar a função de perda
def calcular_loss(Y, probabilidades):
    eps = 1e-12
    p = np.clip(probabilidades, eps, 1 - eps)
    loss = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    return float(loss)

loss_teste = calcular_loss(Y_xor_mlp, A2_teste)
print("\nFunção de perda: ")
print("Erro inicial (Binary Cross-Entropy):", loss_teste)

#6: Implementar Backpropagation
def backward(Y, params, cache):
    m = Y.shape[0]
    X_in = cache["X"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = (A2 - Y) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = X_in.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }

grads_teste = backward(Y_xor_mlp, params_mlp, cache_teste)
print("\nBackpropagation: ")
for k in grads_teste:
    print(f"Gradiente {k}: dimensão {grads_teste[k].shape}")

#7: Treinar a rede
def update_parameters(params, gradients, lr):
    for key in params:
        params[key] -= lr * gradients[key]
    return params

def treinar_mlp(X, Y, n_ocultos=4, lr=0.5, epochs=20000):
    params = inicializar_mlp(n_entradas=X.shape[1], n_ocultos=n_ocultos, n_saidas=Y.shape[1])
    historico_loss = []
    
    for epoca in range(epochs):
        A2, cache = forward(X, params)
        loss = calcular_loss(Y, A2)
        grads = backward(Y, params, cache)
        params = update_parameters(params, grads, lr)
        
        if epoca % 100 == 0:
            historico_loss.append(loss)
            
    return params, historico_loss

params_treinados, historico_loss = treinar_mlp(X_xor, Y_xor_mlp, n_ocultos=4, lr=0.5, epochs=20000)
print("\nPerda inicial:", historico_loss[0])
print("Perda final:  ", historico_loss[-1])

#8: Visualizar a curva de perda
epocas_eixo = np.arange(len(historico_loss)) * 100

plt.figure(figsize=(7, 4))
plt.plot(epocas_eixo, historico_loss, color="#2563eb", linewidth=2, label="Binary Cross-Entropy")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Curva de Aprendizado da MLP no XOR")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

#9: Verificar as previsões
probabilidades_finais, _ = forward(X_xor, params_treinados)
previsoes_binarias = (probabilidades_finais >= 0.5).astype(int)

print("\nPrevisôes:")
print("Entrada (X) | Probabilidade (A2) | Classe Prevista | Classe Esperada")
for i in range(len(X_xor)):
    print(f"{X_xor[i]}     | {probabilidades_finais[i][0]:.4f}             | {previsoes_binarias[i][0]}               | {int(Y_xor_mlp[i][0])}")

#10: Variar os neurônios ocultos
plt.figure(figsize=(8, 4))
for n_h in [2, 4, 8]:
    _, loss_h = treinar_mlp(X_xor, Y_xor_mlp, n_ocultos=n_h, lr=0.5, epochs=10000)
    plt.plot(np.arange(len(loss_h))*100, loss_h, label=f"{n_h} neurônios ocultos")

plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Impacto do Número de Neurônios Ocultos")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

#11: Variar a taxa de aprendizado
plt.figure(figsize=(8, 4))
for lr_test in [0.05, 0.5, 2.5]:
    _, loss_lr = treinar_mlp(X_xor, Y_xor_mlp, n_ocultos=4, lr=lr_test, epochs=10000)
    plt.plot(np.arange(len(loss_lr))*100, loss_lr, label=f"taxa (lr) = {lr_test}")

plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Impacto da Taxa de Aprendizado")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

# O Perceptron comum falha porque só sabe fazer uma "régua reta" em 2D, e o XOR tem os pontos cruzados 
# em diagonais. A Camada Oculta da MLP resolve isso. Nesse novo espaço elevado, os pontos ficam fáceis 
# de separar com um plano reto.