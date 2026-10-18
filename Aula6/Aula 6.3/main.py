import numpy as np

#Montando X, β e y em NumPy
x = np.array([1.0, 2.0, 3.0])   #(3,)
y = np.array([2.0, 4.0, 6.0])   #(3,)
m = len(y)  #3 amostras
X = np.column_stack([np.ones(m), x])    #(3,2)
beta = np.zeros(X.shape[1])    #(2,): [b0, b1]

#Versao 1: o algoritmo com laço explícito
x = np.array([1.0, 2.0, 3.0])  
y = np.array([2.0, 4.0, 6.0]) 
m = len(x)  #3 amostras
b0, b1 = 0.0, 0.0   #β0 e β1 iniciais
alpha = 0.1     

for t in range(1000):
    y_hat = b0 + b1 * x
    erro = y_hat - y
    grad_b0 = erro.sum() / m
    grad_b1 = (erro * x).sum() / m
    b0 = b0 - alpha * grad_b0
    b1 = b1 - alpha * grad_b1


#Versao 2: o algoritmo vetorizado
alpha, epocas, tol = 0.1, 1000, 1e-8
historico = []

for t in range(epocas):
    y_hat = X @ beta
    erro = y_hat - y
    custo = (erro @ erro) / (2 * m)
    historico.append(custo)

    gradiente = (X.T @ erro) / m
    beta = beta - alpha * gradiente

    if t > 0 and abs(historico[-2] - historico[-1]) < tol:
        break
print(epocas)

#Calculando J em uma grade para desenhar o mapa
trajetoria = [beta.copy()]  #Antes do laço
trajetoria.append(beta.copy())  #No laço, após atualizar
traj = np.array(trajetoria) #(epocas + 1, 2)

B0, B1 = np.meshgrid(np.linspace(-1, 1.5, 200),
                     np.linspace(-0.5, 3, 200)) #200, 200
E = B0[..., None] + B1[..., None] * x - y   #(200, 200, m)
J_grade = (E ** 2).sum(axis=-1) / (2*m)    #(200, 200)

import matplotlib.pyplot as plt

plt.contour(B0, B1, J_grade, levels=20)
plt.plot(traj[:, 0], traj[:, 1], 'o-')
plt.show()

#Animando a trajetoria com FuncAnimation
fig3d = plt.figure()
ax3d = fig3d.add_subplot(projection='3d')
ax3d.plot_surface(B0, B1, J_grade, cmap='viridis', alpha=0.8)
ax3d.set_xlabel('β₀'); ax3d.set_ylabel('β₁'); ax3d.set_zlabel('J')
plt.show()

class gradienteDescendente:
    def __init__(self, alpha = 0.1, epocas = 1000, tol = 1e-8):
        self.alpha = alpha
        self.epocas = epocas
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        X_com_intercepto = np.column_stack([np.ones(len(y)), X])

        self.beta_ = np.zeros(X_com_intercepto.shape[1])
        self.historico_ = []

        m = len(y)
        self.trajetoria_ = [self.beta_.copy()]

        for t in range(self.epocas):
            y_hat = X_com_intercepto @ self.beta_
            erro = y_hat - y

            custo = (erro @ erro) / (2 * m)
            self.historico_.append(custo)

            gradiente = (X_com_intercepto.T @ erro) / m
            self.beta_ = self.beta_ - self.alpha * gradiente
            self.trajetoria_.append(self.beta_.copy())

            if t > 0 and abs(self.historico_[-2] - self.historico_[-1]) < self.tol:
                break

    def predict(self, X):
        X = np.asarray(X)
        X_com_intercepto = np.column_stack([np.ones(len(X)), X])
        return X_com_intercepto @ self.beta_

    def score(self, X, y):
        y = np.asarray(y)
        y_hat = self.predict(X)

        media_y = np.mean(y)
        soma_erros_modelo = np.sum((y - y_hat) ** 2)
        soma_erros_media = np.sum((y - media_y) ** 2)

        return 1 - soma_erros_modelo / soma_erros_media

modelo = gradienteDescendente()
modelo.fit(x, y)

print(modelo.beta_)
print(modelo.historico_[-1])

print(modelo.predict(x))

print(modelo.score(x, y))

from matplotlib.animation import FuncAnimation

# Grade de valores possíveis para β0 e β1
B0, B1 = np.meshgrid(
    np.linspace(-1, 1.5, 200),
    np.linspace(-0.5, 3, 200)
)

# Calcula o custo em cada ponto da grade
E = B0[..., None] + B1[..., None] * x - y
J_grade = (E ** 2).sum(axis=-1) / (2 * len(y))

traj = np.array(modelo.trajetoria_)

fig, ax = plt.subplots()
ax.contour(B0, B1, J_grade, levels=20)
ax.set_xlabel("β₀")
ax.set_ylabel("β₁")
ax.set_title("Trajetória do gradiente descendente")

# Ponto que será movido pela animação
ponto, = ax.plot([], [], "ro")

def atualizar(quadro):
    ponto.set_data([traj[quadro, 0]], [traj[quadro, 1]])
    return ponto,

animacao = FuncAnimation(
    fig,
    atualizar,
    frames=len(traj),
    interval=30,
    blit=False
)

plt.show()

"""
β: os coeficientes que o modelo aprende. β0 é a previsão de y quando x = 0, e β1 mostra quanto a previsão de y 
muda quando x muda 1 valor.
α: taxa de aprendizado. Define o tamanho do passo.
Gradiente: mostra como o custo muda em relação a cada coeficiente
Custo: mede o erro do modelo no treino.
"""