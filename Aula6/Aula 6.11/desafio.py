# Etapas 1-3: Carregar, explorar e centralizar

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Carregar os dados
wine = load_wine()
X = wine.data
y = wine.target

# 2. Explorar o formato
print("Formato de X:", X.shape)
print("Número de características (p):", X.shape[1])
print("Classes de vinho:", wine.target_names)

# 3. Centralizar os dados
media = X.mean(axis=0)
X_centralizado = X - media

# Etapas 4-7: covariância, autovalores, EVR, dois componentes

# 4. Matriz de covariância (13x13)
cov = (X_centralizado.T @ X_centralizado) / (len(X) - 1)

# 5. Autovalores
autovalores, autovetores = np.linalg.eigh(cov)

# 6. Ordenação decrescente (do maior autovalor para o menor)
indices = np.argsort(autovalores)[::-1]
autovalores = autovalores[indices]
autovetores = autovetores[:, indices]

# 7. EVR
evr = autovalores / autovalores.sum()
evr_acumulada = np.cumsum(evr)

print(f"EVR do PC1 (sem padronização): {evr[0]*100:.2f}%")
print(f"EVR do PC2 (sem padronização): {evr[1]*100:.2f}%")
print(f"Variância acumulada em 2D: {evr_acumulada[1]*100:.2f}%")

# Etapas 8-10: Projetar, gráfico 2D, comparar com Scikit-Learn
k = 2
W = autovetores[:, :k] 

# 8. Projeção
Z = X_centralizado @ W

# 9. Plot 2D
plt.figure(figsize=(8, 5))
for classe_id, nome_classe in enumerate(wine.target_names):
    plt.scatter(Z[y == classe_id, 0], Z[y == classe_id, 1], label=nome_classe, alpha=0.8)

plt.xlabel(f"PC1 ({evr[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({evr[1]*100:.2f}%)")
plt.title("Wine Dataset 2D via PCA (Sem Padronização)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()

# 10. Validação com Scikit-Learn
pca_sk = PCA(n_components=2)
Z_sk = pca_sk.fit_transform(X)

# Comparar subespaços projetados
P_manual = W @ W.T
P_sk = pca_sk.components_.T @ pca_sk.components_
print("Está idêntico ao Scikit-Learn?", np.allclose(P_manual, P_sk))

# Etapas 11-12: Padronizado (StandardScaler) vs. Não Padronizado

# 11. Padronizado (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ajustar PCA nos dados padronizados
pca_scaled = PCA(n_components=2)
Z_scaled = pca_scaled.fit_transform(X_scaled)
evr_scaled = pca_scaled.explained_variance_ratio_

print("\nComparação de Variância: \n")
print(f"Sem Padronização -> PC1: {evr[0]*100:.2f}%, PC2: {evr[1]*100:.2f}% | Total: {evr_acumulada[1]*100:.2f}%")
print(f"Com StandardScaler -> PC1: {evr_scaled[0]*100:.2f}%, PC2: {evr_scaled[1]*100:.2f}% | Total: {(evr_scaled[0]+evr_scaled[1])*100:.2f}%")

# Grafico dos dados padronizados
plt.figure(figsize=(8, 5))
for classe_id, nome_classe in enumerate(wine.target_names):
    plt.scatter(Z_scaled[y == classe_id, 0], Z_scaled[y == classe_id, 1], label=nome_classe, alpha=0.8)

plt.xlabel(f"PC1 ({evr_scaled[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({evr_scaled[1]*100:.2f}%)")
plt.title("Wine Dataset 2D via PCA (Com StandardScaler = Padronizado)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()