# 01 Carregar o dataset
import time
import numpy as np
import pandas as pd
import optuna
from scipy.stats import randint, loguniform
from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

wine = load_wine()
X, y = wine.data, wine.target

# 02 Explorar os dados
df = pd.DataFrame(X, columns=wine.feature_names)
df["classe"] = y

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Nomes das classes:", wine.target_names)
print("Distribuição das classes:", np.unique(y, return_counts=True))
print("Valores ausentes:", df.isnull().sum().sum())
print(df.head())
print(df.describe().round(2))

# 03 Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(X_train, y_train))

print(f"Amostras de treino: {len(X_train)} | Amostras de teste: {len(X_test)}")

# 04 Definir o espaço de busca
ORCAMENTO = 12
modelo_base = GradientBoostingClassifier(n_estimators=100, random_state=42)

# 3 x 2 x 2 = 12 combinações exatas para o Grid Search
param_grid = {
    "max_depth": [1, 2, 3],
    "learning_rate": [0.03, 0.10],
    "min_samples_leaf": [1, 5]
}

# Distribuições equivalentes para o Random Search
param_distributions = {
    "max_depth": randint(1, 4),
    "learning_rate": loguniform(0.02, 0.15),
    "min_samples_leaf": randint(1, 6)
}

# 05 Executar Grid Search
grid = GridSearchCV(
    estimator=modelo_base,
    param_grid=param_grid,
    scoring="accuracy",
    cv=splits,
    n_jobs=1,
    refit=False
)

inicio = time.perf_counter()
grid.fit(X_train, y_train)
tempo_grid = time.perf_counter() - inicio

print("Grid - Melhores parâmetros:", grid.best_params_)
print(f"Grid - Melhor Acurácia CV: {grid.best_score_:.4f}")

# 06 Executar Random Search
random_search = RandomizedSearchCV(
    estimator=modelo_base,
    param_distributions=param_distributions,
    n_iter=ORCAMENTO,
    scoring="accuracy",
    cv=splits,
    random_state=42,
    n_jobs=1,
    refit=False
)

inicio = time.perf_counter()
random_search.fit(X_train, y_train)
tempo_random = time.perf_counter() - inicio

print("Random - Melhores parâmetros:", random_search.best_params_)
print(f"Random - Melhor Acurácia CV: {random_search.best_score_:.4f}")

# 07 Executar Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5)
    }
    modelo = GradientBoostingClassifier(n_estimators=100, random_state=42, **params)
    return cross_val_score(modelo, X_train, y_train, cv=splits, scoring="accuracy").mean()

sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=5)
study = optuna.create_study(direction="maximize", sampler=sampler)

inicio = time.perf_counter()
study.optimize(objective, n_trials=ORCAMENTO, n_jobs=1)
tempo_optuna = time.perf_counter() - inicio

print("Optuna - Melhores parâmetros:", study.best_params)
print(f"Optuna - Melhor Acurácia CV: {study.best_value:.4f}")

# 08 Comparar os resultados
tabela_cv = pd.DataFrame({
    "Estratégia": ["Grid Search", "Random Search", "Optuna (TPE)"],
    "Acurácia CV": [grid.best_score_, random_search.best_score_, study.best_value],
    "Melhores Parâmetros": [
        str(grid.best_params_),
        str(random_search.best_params_),
        str(study.best_params)
    ]
})
print(tabela_cv)

# 09 Avaliar os melhores modelos
melhores_configs = {
    "Grid Search": (grid.best_params_, grid.best_score_),
    "Random Search": (random_search.best_params_, random_search.best_score_),
    "Optuna (TPE)": (study.best_params, study.best_value)
}

avaliacoes = []
for nome, (params, acc_cv) in melhores_configs.items():
    modelo_final = GradientBoostingClassifier(n_estimators=100, random_state=42, **params)
    modelo_final.fit(X_train, y_train)
    y_pred = modelo_final.predict(X_test)
    
    avaliacoes.append({
        "Estratégia": nome,
        "Acurácia CV": round(acc_cv, 4),
        "Acurácia Teste": round(accuracy_score(y_test, y_pred), 4),
        "F1 Macro Teste": round(f1_score(y_test, y_pred, average="macro"), 4)
    })
    print(f"\nMatriz de Confusão - {nome}:\n{confusion_matrix(y_test, y_pred)}")

tabela_teste = pd.DataFrame(avaliacoes)
print(tabela_teste)

# 10 Comparar os tempos
n_folds = len(splits)
tabela_tempos = pd.DataFrame({
    "Estratégia": ["Grid Search", "Random Search", "Optuna (TPE)"],
    "Configurações": [ORCAMENTO, ORCAMENTO, ORCAMENTO],
    "Treinos (fits)": [ORCAMENTO * n_folds] * 3,
    "Tempo Total (s)": [round(tempo_grid, 3), round(tempo_random, 3), round(tempo_optuna, 3)],
    "Tempo/Config (s)": [
        round(tempo_grid / ORCAMENTO, 4),
        round(tempo_random / ORCAMENTO, 4),
        round(tempo_optuna / ORCAMENTO, 4)
    ]
})
print(tabela_tempos)

# - Diferença entre Validação (CV) e Teste: a Acurácia CV (best_score_) costuma ser
#   otimista por representar o máximo entre várias tentativas.
# - Cobertura do espaço de busca: o Grid Search fica preso a pontos fixos da grade e
#   pode pular regiões ótimas intermediárias. Já o Random Search e o Optuna exploram
#   valores contínuos, avaliando mais valores distintos nos hiperparâmetros que realmente impactam o modelo.
# - Aprendizado e tempo de execução: Grid e Random ignoram os resultados anteriores. O Optuna (TPE) usa os 5 primeiros
#   trials aleatórios como base e depois orienta a busca pelo histórico, adicionando
#   um pequeno overhead matemático para calcular as densidades l(x) e g(x).

# - Nenhuma estratégia é sempre superior: o melhor método depende do tamanho do
#   espaço de busca, do orçamento de tentativas e da complexidade do problema.
# - Neste desafio com o dataset Wine (poucas amostras, treino rápido e orçamento
#   curto de 12 configurações), o Random Search e o Grid Search se mostram muito
#   competitivos e velozes, com vantagem conceitual para o Random Search por explorar
#   o learning_rate de forma contínua na escala logarítmica sem overhead.
# - O Optuna (TPE) torna-se a escolha mais interessante em cenários mais pesados:
#   datasets maiores, espaços com muitos hiperparâmetros e modelos demorados, onde
#   a busca adaptativa e o Pruning economizam bastante tempo computacional.