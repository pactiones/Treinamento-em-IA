# 01 Criar dataset com make_moons
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = make_moons(
    n_samples=300,
    noise=0.20,
    random_state=42
)
print(X.shape, y.shape)
print(np.unique(y, return_counts=True))

# 02 Visualizar os dados
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1],
            c="#B2B9C1", marker="s",
            edgecolor="k", label="Classe 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1],
            c="#FF6B35", marker="o",
            edgecolor="k", label="Classe 1")
plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
plt.legend(loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, -0.1), frameon=False)
plt.axis("equal"); plt.show()

# 03 Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
print(X_train.shape, X_test.shape)

# 04 Pipeline com StandardScaler
svm_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="linear", C=1.0))
])

svm_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

# 05 Treinar SVM Linear
svm_linear.fit(X_train, y_train)

# 06 Treinar SVM RBF
svm_rbf.fit(X_train, y_train)

# 07 Comparar os modelos
pred_linear = svm_linear.predict(X_test)
pred_rbf = svm_rbf.predict(X_test)

# 08 Calcular acurácia
acc_linear_tr = accuracy_score(y_train, svm_linear.predict(X_train))
acc_linear_te = accuracy_score(y_test, pred_linear)

acc_rbf_tr = accuracy_score(y_train, svm_rbf.predict(X_train))
acc_rbf_te = accuracy_score(y_test, pred_rbf)

print(f"SVM Linear | Treino: {acc_linear_tr:.3f} | Teste: {acc_linear_te:.3f}")
print(f"SVM RBF    | Treino: {acc_rbf_tr:.3f} | Teste: {acc_rbf_te:.3f}")

# 09 Matrizes de confusão
cm_linear = confusion_matrix(y_test, pred_linear)
cm_rbf = confusion_matrix(y_test, pred_rbf)

print("\nMatriz de Confusão - SVM Linear:")
print(cm_linear)
print("\nMatriz de Confusão - SVM RBF:")
print(cm_rbf)

# 10 Visualizar fronteiras
def plot_decision_boundary(model, X, y, ax, title, x_lim, y_lim, resolution=300):
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], resolution),
        np.linspace(y_lim[0], y_lim[1], resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    D = model.decision_function(grid).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#E7E5E1", "#F8DED1"])
    ax.contour(xx, yy, D, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
    
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#B2B9C1", marker="s", edgecolor="k", s=30, label="Classe 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#FF6B35", marker="o", edgecolor="k", s=30, label="Classe 1")
    
    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))

x_lim = (X[:, 0].min() - 0.3, X[:, 0].max() + 0.3)
y_lim = (X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
plot_decision_boundary(svm_linear, X_train, y_train, axes[0], f"SVM Linear | teste {acc_linear_te:.3f}", x_lim, y_lim)
plot_decision_boundary(svm_rbf, X_train, y_train, axes[1], f"SVM RBF | teste {acc_rbf_te:.3f}", x_lim, y_lim)
plt.tight_layout(); plt.show()

# 11 Testar valores de C
valores_C = [0.01, 1, 100]
modelos_C = {}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, C in zip(axes, valores_C):
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=C, gamma=1.0))
    ])
    modelo.fit(X_train, y_train)
    modelos_C[C] = modelo
    acc_tr = accuracy_score(y_train, modelo.predict(X_train))
    acc_te = accuracy_score(y_test, modelo.predict(X_test))
    plot_decision_boundary(modelo, X_train, y_train, ax, f"C={C} | treino {acc_tr:.3f} | teste {acc_te:.3f}", x_lim, y_lim)

plt.tight_layout(); plt.show()

# 12 Testar valores de gamma
valores_gamma = [0.1, 1, 10]
modelos_gamma = {}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, g in zip(axes, valores_gamma):
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma=g))
    ])
    modelo.fit(X_train, y_train)
    modelos_gamma[g] = modelo
    acc_tr = accuracy_score(y_train, modelo.predict(X_train))
    acc_te = accuracy_score(y_test, modelo.predict(X_test))
    plot_decision_boundary(modelo, X_train, y_train, ax, f"gamma={g} | treino {acc_tr:.3f} | teste {acc_te:.3f}", x_lim, y_lim)

plt.tight_layout(); plt.show()

# 13 Interpretar os resultados

# 1. SVM Linear vs RBF: O modelo Linear não consegue separar as meias-luas (make_moons), 
#    atingindo uma acurácia limitada. O kernel RBF cria uma fronteira curva que se fica perfeitamente no formato.
# 2. Efeito de C: Valores muito baixos (C=0.01) criam margens largas demais e subajustam os dados (underfitting). 
#    Valores muito altos (C=100) penalizam qualquer erro, tornando a margem rígida e sensível a ruídos(overfitting).
# 3. Efeito de Gamma: Gamma muito baixo (gamma=0.1) gera uma fronteira excessivamente suave. 
#    Gamma muito alto (gamma=10) restringe a influência aos pontos mais próximos, gerando ilhas e sobreajuste (overfitting).
# 4. Melhores Hiperparâmetros: O modelo RBF com C=1.0 e gamma=1.0 (ou gamma='scale') oferece o melhor equilíbrio 
#    entre acurácia no treino e capacidade de generalização no teste.