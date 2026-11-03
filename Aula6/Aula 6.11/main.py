# Etapas 1-3: Imports, carregar dados e explorar
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Carregar o dataset Iris (150 amostras, 4 características)
iris = load_iris()
X = iris.data
y = iris.target  # Rótulos (usados apenas para a cor no gráfico final)

print("Formato de X:", X.shape)
print("Características:", iris.feature_names)
print("Classes:", iris.target_names)
print("Contagem por classe:", np.bincount(y))

# Etapas 5-6: Média por coluna e centralização dos dados (Xc = X - mu)
media = X.mean(axis=0)
X_centralizado = X - media

print("Média original de cada característica:\n", np.round(media, 4))
print("\nMédia após centralizar (deve ser ~0):\n", np.round(X_centralizado.mean(axis=0), 12))

# Etapa 7: Matriz de Covariância (p x p, no caso 4x4)
n = len(X)
cov_manual = (X_centralizado.T @ X_centralizado) / (n - 1)

# Comparando nosso cálculo manual com a função np.cov do NumPy
cov_numpy = np.cov(X_centralizado, rowvar=False)

print("Matriz de Covariância (4x4):\n", np.round(cov_manual, 4))
print("\nO cálculo manual é igual ao np.cov?", np.allclose(cov_manual, cov_numpy))

# Etapas 8-9: Calcular autovalores/autovetores e ordenar em ordem decrescente
autovalores, autovetores = np.linalg.eigh(cov_manual)

# O eigh devolve em ordem crescente, então invertemos os índices ([::-1])
indices = np.argsort(autovalores)[::-1]

autovalores = autovalores[indices]
autovetores = autovetores[:, indices]  # Importante: reordenar as COLUNAS!

print("Autovalores ordenados (Variâncias):\n", np.round(autovalores, 4))
print("\nVariância Total (Soma dos autovalores / Traço):", np.round(autovalores.sum(), 4))

# Etapas 10-11: Variância Explicada Proporcional (EVR) e Acumulada
evr = autovalores / autovalores.sum()
evr_acumulada = np.cumsum(evr)

for i in range(len(evr)):
    print(f"PC{i+1}: EVR = {evr[i]*100:.2f}% | Acumulada = {evr_acumulada[i]*100:.2f}%")

# Etapas 12-13: Matriz de Projeção W (4x2) e cálculo dos Scores Z (150x2)
k = 2
W = autovetores[:, :k]  # Seleciona as 2 primeiras colunas

# Projeção: (150x4) @ (4x2) -> (150x2)
Z = X_centralizado @ W

print("Formato da matriz de pesos W:", W.shape)
print("Formato dos dados reduzidos Z:", Z.shape)
print("\nPrimeiras 3 amostras projetadas em 2D:\n", np.round(Z[:3], 4))

# Etapa 14: Plot da projeção em 2D com as classes coloridas
plt.figure(figsize=(8, 5))

# Plot de cada classe para colocar a legenda
for classe_id, nome_classe in enumerate(iris.target_names):
    plt.scatter(
        Z[y == classe_id, 0], 
        Z[y == classe_id, 1], 
        label=nome_classe, 
        alpha=0.8
    )

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel(f"PC1 ({evr[0]*100:.2f}% da variância)")
plt.ylabel(f"PC2 ({evr[1]*100:.2f}% da variância)")
plt.title("Iris Dataset reduzido de 4D para 2D via PCA (NumPy)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()

# Etapa 15: Reconstrução dos dados originais e erro quadrático médio (MSE)
X_reconstruido = Z @ W.T + media

# Cálculo do Erro Quadrático Médio
mse = np.mean((X - X_reconstruido) ** 2)

print("Formato da reconstrução:", X_reconstruido.shape)
print(f"Erro Quadrático Médio (MSE com k=2): {mse:.5f}")

# Etapas 18-19: Comparação com a biblioteca Scikit-Learn
pca_sk = PCA(n_components=2)
Z_sk = pca_sk.fit_transform(X)

print("EVR Scikit-Learn:", np.round(pca_sk.explained_variance_ratio_, 4))
print("Nossa EVR (NumPy): ", np.round(evr[:2], 4))

# Verificação do Subespaço (compara as projeções ignorando possíveis inversões de sinal)
P_manual = W @ W.T
P_sk = pca_sk.components_.T @ pca_sk.components_

print("\nO subespaço gerado na mão é idêntico ao do Scikit-Learn?", np.allclose(P_manual, P_sk))