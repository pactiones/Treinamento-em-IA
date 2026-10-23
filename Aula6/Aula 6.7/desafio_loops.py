# Preparação dos dados
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Amostras de treino:", len(X_train))
print("Amostras de teste:", len(X_test))
print("Características por amostra:", X_train.shape[1])

# KNN com loops e distância Euclidiana

def prever_knn_euclidiano_loops(X_train, y_train, X_test, k):

    previsoes = []

    # O loop externo percorre cada amostra que queremos classificar.
    for amostra_teste in X_test:
        distancias = []

        # O loop interno compara essa amostra com cada amostra de treino.
        for amostra_treino in X_train:
            diferencas = amostra_teste - amostra_treino

            # Distância Euclidiana:
            distancia = np.sqrt(np.sum(diferencas ** 2))
            distancias.append(distancia)

        # Ordena os índices das distâncias, da menor para a maior.
        # kind="stable" faz o resultado ficar previsível em caso de distâncias iguais.
        indices_ordenados = np.argsort(distancias, kind="stable")

        # Pega os índices dos k vizinhos mais próximos.
        indices_vizinhos = indices_ordenados[:k]

        # Usa esses índices para buscar os rótulos dos vizinhos.
        rotulos_vizinhos = y_train[indices_vizinhos]

        # Conta quantos votos cada classe recebeu.
        classes, contagens = np.unique(
            rotulos_vizinhos,
            return_counts=True
        )

        # np.unique ordena as classes.
        # Se houver empate, argmax escolhe a primeira classe nessa ordem.
        classe_prevista = classes[np.argmax(contagens)]
        previsoes.append(classe_prevista)

    return np.array(previsoes)


# Calcula a proporção de previsões corretas.
def calcular_acuracia(y_real, y_previsto):
    return np.mean(y_real == y_previsto)


# Testa o KNN com k = 5.
k = 5
previsoes = prever_knn_euclidiano_loops(
    X_train,
    y_train,
    X_test,
    k
)

acuracia = calcular_acuracia(y_test, previsoes)

print("Previsões:", previsoes)
print("Classes corretas:", y_test)
print("Acurácia:", acuracia)

# Escalonamento dos dados

from sklearn.preprocessing import StandardScaler

escalonador = StandardScaler()

# Aprende média e desvio padrão somente com os dados de treino.
X_train_escalado = escalonador.fit_transform(X_train)

# Aplica ao teste os parâmetros aprendidos no treino.
X_test_escalado = escalonador.transform(X_test)

# Executa a versão com dados escalonados

previsoes_loops_escaladas = prever_knn_euclidiano_loops(
    X_train_escalado,
    y_train,
    X_test_escalado,
    k
)

acuracia_loops_escalada = calcular_acuracia(
    y_test,
    previsoes_loops_escaladas
)

print("Acurácia dos loops sem escalonamento:", acuracia)
print("Acurácia dos loops com escalonamento:", acuracia_loops_escalada)

# Medição de tempo

from time import perf_counter

def medir_tempo(funcao, repeticoes=30):
    funcao()  # aquecimento, fora da medição
    tempos = []

    for _ in range(repeticoes):
        inicio = perf_counter()
        funcao()
        fim = perf_counter()
        tempos.append(fim - inicio)

    return np.median(tempos)

# Mede o tempo das previsões com dados originais e escalonados

tempo_loops_sem = medir_tempo(
    lambda: prever_knn_euclidiano_loops(
        X_train, y_train, X_test, k
    )
)

tempo_loops_escalado = medir_tempo(
    lambda: prever_knn_euclidiano_loops(
        X_train_escalado, y_train, X_test_escalado, k
    )
)

print("Loops — sem escalonamento:", tempo_loops_sem, "segundos")
print("Loops — com escalonamento:", tempo_loops_escalado, "segundos")

# Gráfico comparativo dos tempos

import matplotlib.pyplot as plt
import numpy as np

categorias = ["Sem escalonamento", "Com escalonamento"]

tempos_loops_ms = np.array([
    tempo_loops_sem,
    tempo_loops_escalado
]) * 1000

posicoes = np.arange(len(categorias))
largura = 0.35

plt.bar(
    posicoes - largura / 2,
    tempos_loops_ms,
    largura,
    label="Loops — Euclidiana"
)

plt.xticks(posicoes, categorias)
plt.ylabel("Tempo mediano (milissegundos)")
plt.title("Tempo de execução das versões do KNN")
plt.legend()
plt.tight_layout()
plt.show()

'''
Sem escalonamento, se obteve uma acurácia de 68,42% - 26 acertos em 38 amostras. 
Com escalonamento, a acurácia foi de 63,16% para a distância Euclidiana com loops. 
Neste experimento, portanto, o escalonamento não aumentou a acurácia.

Nos tempos medidos, a versão com loops levou aproximadamente 9,57 ms sem escalonamento e 10,33 ms com escalonamento



O KNN pode ser implementado calculando as distâncias com loops ou com operações vetorizadas do NumPy. 
Os loops deixam o cálculo explícito e facilitam acompanhar a lógica, enquanto o broadcasting calcula 
várias distâncias de uma vez e foi mais rápido neste experimento.
'''