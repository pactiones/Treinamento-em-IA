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

# KNN com broadcasting e distância Manhattan

def prever_knn_manhattan_broadcasting(X_train, y_train, X_test, k):

    # Cria diferenças entre todos os pares:
    # cada amostra de teste e cada amostra de treino.
    diferencas = X_test[:, None, :] - X_train[None, :, :]

    # A distância Manhattan soma os valores absolutos das diferenças.
    # axis=2 indica que somamos ao longo das características.
    distancias = np.sum(np.abs(diferencas), axis=2)

    # Para cada amostra de teste, ordena os índices dos treinos
    # pela distância e seleciona os k primeiros.
    indices_vizinhos = np.argsort(
        distancias,
        axis=1,
        kind="stable"
    )[:, :k]

    # Busca os rótulos dos vizinhos.
    # Cada linha corresponde aos vizinhos de uma amostra de teste.
    rotulos_vizinhos = y_train[indices_vizinhos]

    previsoes = []

    # Faz uma votação para cada amostra de teste.
    for rotulos in rotulos_vizinhos:
        classes, contagens = np.unique(
            rotulos,
            return_counts=True
        )

        # Em caso de empate, escolhe a menor classe.
        classe_prevista = classes[np.argmax(contagens)]
        previsoes.append(classe_prevista)

    return np.array(previsoes)


# Calcula a proporção de previsões corretas.
def calcular_acuracia(y_real, y_previsto):
    return np.mean(y_real == y_previsto)

# Usa o mesmo k e os mesmos conjuntos de treino e teste da primeira versão.
k = 5
previsoes_manhattan = prever_knn_manhattan_broadcasting(
    X_train,
    y_train,
    X_test,
    k
)

acuracia_manhattan = calcular_acuracia(y_test, previsoes_manhattan)

print("Previsões:", previsoes_manhattan)
print("Classes corretas:", y_test)
print("Acurácia:", acuracia_manhattan)

# Escalonamento dos dados

from sklearn.preprocessing import StandardScaler

escalonador = StandardScaler()

# Aprende média e desvio padrão somente com os dados de treino.
X_train_escalado = escalonador.fit_transform(X_train)

# Aplica ao teste os parâmetros aprendidos no treino.
X_test_escalado = escalonador.transform(X_test)

# Executa a versão com dados escalonados

previsoes_manhattan_escaladas = prever_knn_manhattan_broadcasting(
    X_train_escalado,
    y_train,
    X_test_escalado,
    k
)

acuracia_manhattan_escalada = calcular_acuracia(
    y_test,
    previsoes_manhattan_escaladas
)

print("Acurácia de Manhattan sem escalonamento:", acuracia_manhattan)
print("Acurácia de Manhattan com escalonamento:", acuracia_manhattan_escalada)

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

tempo_broadcasting_sem = medir_tempo(
    lambda: prever_knn_manhattan_broadcasting(
        X_train, y_train, X_test, k
    )
)

tempo_broadcasting_escalado = medir_tempo(
    lambda: prever_knn_manhattan_broadcasting(
        X_train_escalado, y_train, X_test_escalado, k
    )
)

print("Broadcasting — sem escalonamento:", tempo_broadcasting_sem, "segundos")
print("Broadcasting — com escalonamento:", tempo_broadcasting_escalado, "segundos")

# Gráfico comparativo dos tempos

import matplotlib.pyplot as plt
import numpy as np

categorias = ["Sem escalonamento", "Com escalonamento"]

tempos_broadcasting_ms = np.array([
    tempo_broadcasting_sem,
    tempo_broadcasting_escalado
]) * 1000

posicoes = np.arange(len(categorias))
largura = 0.35


plt.bar(
    posicoes + largura / 2,
    tempos_broadcasting_ms,
    largura,
    label="Broadcasting — Manhattan"
)

plt.xticks(posicoes, categorias)
plt.ylabel("Tempo mediano (milissegundos)")
plt.title("Tempo de execução das versões do KNN")
plt.legend()
plt.tight_layout()
plt.show()

'''
Sem escalonamento, se obteve uma acurácia de 68,42% - 26 acertos em 38 amostras. 
Com escalonamento, a acurácia foi de 60,53% para a distância Manhattan com broadcasting. 
Neste experimento, portanto, o escalonamento não aumentou a acurácia.

Nos tempos medidos, a versão com broadcasting levou cerca de 0,41 ms e 0,46 ms, respectivamente


O KNN pode ser implementado calculando as distâncias com loops ou com operações vetorizadas do NumPy. 
Os loops deixam o cálculo explícito e facilitam acompanhar a lógica, enquanto o broadcasting calcula 
várias distâncias de uma vez e foi mais rápido neste experimento.
'''