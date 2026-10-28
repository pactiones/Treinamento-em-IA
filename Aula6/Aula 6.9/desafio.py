import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#TÓPICO 1: Receber X e y
#-------------------------
# [idade em anos, visitas à loja no mês]
X = np.array([
    [19, 1], [25, 6], [22, 6], [28, 8],
    [24, 2], [31, 3], [27, 3], [34, 9],
    [35, 2], [36, 5], [41, 3], [39, 8],
    [47, 1], [42, 6], [55, 4], [45, 10],
    [58, 8], [29, 4], [61, 6], [48, 7],
    [64, 9], [33, 11], [38, 4], [52, 9]
])

#0 não comprou, 1 comprou
y = np.array([
    0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1
])

print("Shape de X: ", X.shape)
print("Shape de y: ", y.shape)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Formato de X_treino:", X_treino.shape)
print("Formato de X_teste:", X_teste.shape)
print("Classes no treino:", np.bincount(y_treino))
print("Classes no teste:", np.bincount(y_teste))

#TÓPICO 2: Impureza de Gini
#-------------------------

def gini(y):
    if len(y) == 0:
        return 0.0

    _, contagens = np.unique(y, return_counts=True)
    proporcoes = contagens / len(y)

    return 1.0 - np.sum(proporcoes**2)

print("Misturado:", gini(np.array([0, 0, 1, 1])))
print("Puro:", gini(np.array([0, 0, 0, 0])))
print("Exemplo do material:", gini(np.array([0, 0, 1, 1, 1])))

#TÓPICO 3: Buscar o melhor corte
#-------------------------

def gini_ponderado(y_esquerda, y_direita):
    total = len(y_esquerda) + len(y_direita)

    if total == 0:
        return 0.0

    peso_esquerda = len(y_esquerda) / total
    peso_direita = len(y_direita) / total

    return (
        peso_esquerda * gini(y_esquerda)
        + peso_direita * gini(y_direita)
    )


def melhor_corte(X, y):
    melhor_coluna = None
    melhor_limiar = None
    menor_gini = float("inf")

    # Testa idade e visitas
    for coluna in range(X.shape[1]):
        valores = np.unique(X[:, coluna])

        # Os cortes candidatos ficam entre valores consecutivos
        limiares = (valores[:-1] + valores[1:]) / 2

        for limiar in limiares:
            mascara = X[:, coluna] <= limiar

            y_esquerda = y[mascara]
            y_direita = y[~mascara]

            gini_corte = gini_ponderado(y_esquerda, y_direita)

            if gini_corte < menor_gini:
                melhor_coluna = coluna
                melhor_limiar = limiar
                menor_gini = gini_corte

    return melhor_coluna, melhor_limiar, menor_gini


coluna, limiar, impureza = melhor_corte(X_treino, y_treino)

nomes_caracteristicas = ["Idade", "Visitas"]

print("Melhor característica:", nomes_caracteristicas[coluna])
print("Melhor limiar:", limiar)
print("Gini ponderado do melhor corte:", impureza)


#TÓPICO 4: Criar nós internos
#-------------------------

class No:
    def __init__(
        self,
        coluna=None,
        limiar=None,
        esquerda=None,
        direita=None,
        classe=None,
        amostras=0,
        impureza=0.0
    ):
        self.coluna = coluna       
        self.limiar = limiar      
        self.esquerda = esquerda    # próximo nó se valor <= limiar
        self.direita = direita      # próximo nó se valor > limiar
        self.classe = classe        # será usado pelas folhas
        self.amostras = amostras    # quantidade de exemplos no nó
        self.impureza = impureza    # Gini deste nó

    def eh_folha(self):
        return self.esquerda is None and self.direita is None

raiz = No(
    coluna=coluna,
    limiar=limiar,
    impureza=impureza)

print("Pergunta da raiz:", nomes_caracteristicas[raiz.coluna], "<=", raiz.limiar)
print("É folha?\n", raiz.eh_folha())

#TÓPICO 5: Criar folhas
#-------------------------

def criar_folha(y):
    classes, contagens = np.unique(y, return_counts=True)
    classe_majoritaria = classes[np.argmax(contagens)]

    return No(
        classe=classe_majoritaria,
        amostras=len(y),
        impureza=gini(y)
    )

folha_exemplo = criar_folha(np.array([0, 0, 1]))

print("Classe prevista pela folha:", folha_exemplo.classe)
print("Amostras na folha:", folha_exemplo.amostras)
print("Impureza da folha:", folha_exemplo.impureza)
print("É folha?", folha_exemplo.eh_folha())

#TÓPICO 6: Construção Recursiva
#-------------------------

def construir_arvore(X, y):
    no = criar_folha(y)

    if no.impureza == 0:
        return no

    coluna, limiar, gini_corte = melhor_corte(X, y)

    if coluna is None or gini_corte >= no.impureza:
        return no

    mascara = X[:, coluna] <= limiar

    # Transformamos a folha em nó interno e construímos seus dois ramos
    no.coluna = coluna
    no.limiar = limiar
    no.esquerda = construir_arvore(X[mascara], y[mascara])
    no.direita = construir_arvore(X[~mascara], y[~mascara])

    return no

#TÓPICO 7: Critérios de parada
#-------------------------

def construir_arvore(
    X,
    y,
    profundidade=0,
    max_depth=3,
    min_samples_split=2
):
    no = criar_folha(y)

    # Para se o grupo já for puro
    if no.impureza == 0:
        return no

    # Para se atingir a profundidade máxima
    if profundidade >= max_depth:
        return no

    if len(y) < min_samples_split:
        return no

    coluna, limiar, gini_corte = melhor_corte(X, y)

    # Para se não há corte válido ou ele não melhora a impureza
    if coluna is None or gini_corte >= no.impureza:
        return no

    mascara = X[:, coluna] <= limiar

    no.coluna = coluna
    no.limiar = limiar
    no.esquerda = construir_arvore(
        X[mascara], y[mascara],
        profundidade + 1, max_depth, min_samples_split
    )
    no.direita = construir_arvore(
        X[~mascara], y[~mascara],
        profundidade + 1, max_depth, min_samples_split
    )

    return no

arvore = construir_arvore(X_treino, y_treino, max_depth=3)

# TÓPICO 8: Fazer previsões
# ------------------------

def prever_amostra(amostra, no):
    if no.eh_folha():
        return no.classe

    if amostra[no.coluna] <= no.limiar:
        return prever_amostra(amostra, no.esquerda)

    return prever_amostra(amostra, no.direita)

cliente = X[0]
previsao = prever_amostra(cliente, arvore)

print("Dados do cliente:", cliente)
print("Classe prevista:", previsao)

def prever(X, arvore):
    return np.array([prever_amostra(amostra, arvore) for amostra in X])

y_previsto = prever(X_teste, arvore)

print("Rótulos reais:   ", y_teste)
print("Rótulos previstos:", y_previsto)

# TÓPICO 9: Calcular acurácia
# --------------------------

def acuracia(y_real, y_previsto):
    return np.mean(y_real == y_previsto)

resultado = acuracia(y_teste, y_previsto)

print("Acurácia no teste:", resultado)
print("Acurácia em porcentagem:", resultado * 100, "%")

# TÓPICO 10: Imprimir a árvore
# ----------------------------

def imprimir_arvore(no, nomes_caracteristicas, nivel=0, caminho="[Raiz]"):
    recuo = "    " * nivel

    if no.eh_folha():
        print(f"{recuo}{caminho} -> Classe: {no.classe} "
              f"(amostras: {no.amostras}, Gini: {no.impureza:.3f})")
        return

    pergunta = (
        f"{nomes_caracteristicas[no.coluna]} <= {no.limiar:g}?"
    )
    print(f"{recuo}{caminho} {pergunta}")

    imprimir_arvore(
        no.esquerda, nomes_caracteristicas, nivel + 1, "[Sim]"
    )
    imprimir_arvore(
        no.direita, nomes_caracteristicas, nivel + 1, "[Não]"
    )

nomes_caracteristicas = ["Idade", "Visitas"]

imprimir_arvore(arvore, nomes_caracteristicas)

# TÓPICO 11: Comparar com scikit-learn
# ------------------------------------

arvore_sk = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    min_samples_split=2,
    random_state=42
)

arvore_sk.fit(X_treino, y_treino)
y_previsto_sk = arvore_sk.predict(X_teste)

print("Acurácia da árvore:", acuracia(y_teste, y_previsto))
print("Acurácia do scikit-learn:", accuracy_score(y_teste, y_previsto_sk))
print("Mesmas previsões:", np.array_equal(y_previsto, y_previsto_sk))

import matplotlib.pyplot as plt

# Gráfico das regiões de decisão
# ------------------------------

# Cria uma grade que cobre a faixa de idades e visitas dos dados
idades = np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 200)
visitas = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)

grade_idades, grade_visitas = np.meshgrid(idades, visitas)

# Cada ponto da grade é um cliente hipotético
pontos_grade = np.c_[
    grade_idades.ravel(),
    grade_visitas.ravel()
]

# Prevê a classe de cada ponto e reorganiza para desenhar o fundo
classes_grade = prever(pontos_grade, arvore)
regioes = classes_grade.reshape(grade_idades.shape)

plt.contourf(
    grade_idades,
    grade_visitas,
    regioes,
    levels=[-0.5, 0.5, 1.5],
    colors=["#F4BA64", "#35D49A"],
    alpha=0.3
)

# Mostra os clientes usados no treino
plt.scatter(
    X_treino[y_treino == 0, 0],
    X_treino[y_treino == 0, 1],
    color="#C88724",
    label="Treino: não comprou"
)
plt.scatter(
    X_treino[y_treino == 1, 0],
    X_treino[y_treino == 1, 1],
    color="#15966A",
    label="Treino: comprou"
)

# Mostra os clientes reservados para teste com círculos vazados
plt.scatter(
    X_teste[:, 0],
    X_teste[:, 1],
    facecolors="none",
    edgecolors="black",
    marker="o",
    s=100,
    label="Teste"
)

plt.xlabel("Idade (anos)")
plt.ylabel("Visitas à loja no mês")
plt.title("Regiões de decisão da nossa árvore")
plt.legend()
plt.show()