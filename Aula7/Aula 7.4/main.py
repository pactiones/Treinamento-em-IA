# Bibliotecas e versões
import time, importlib.metadata
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

for p in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost"]:
    print(p, importlib.metadata.version(p))


# Criação do dataset
X, y = make_classification(
    n_samples=2400,
    n_features=16,
    n_informative=8,
    n_redundant=4,
    n_repeated=0,
    n_clusters_per_class=2,
    weights=[0.55, 0.45],
    flip_y=0.03,
    class_sep=1.0,
    random_state=42
)
print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Classes:", np.unique(y, return_counts=True))

# Exploração dos dados
colunas = [f"feature_{i:02d}" for i in range(1, 17)]
df = pd.DataFrame(X, columns=colunas)
df["target"] = y

print(df.head())
print(df.shape)
print(df.dtypes.value_counts())
print(df["target"].value_counts(normalize=True))
print(df.describe().T)

df["target"].value_counts().sort_index().plot(kind="bar", color=["#75BCE8", "#62CDB5"])

# Treino, validação e teste
X_df = df[colunas]
X_temp, X_test, y_temp, y_test = train_test_split(
    X_df, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# Baseline
baseline = DummyClassifier(strategy="prior")
baseline.fit(X_train, y_train)
pred_base = baseline.predict(X_test)
acc_base = accuracy_score(y_test, pred_base)
print(f"Acurácia do baseline: {acc_base:.4f}")

# Configuração e treino do XGBoost
modelo_xgb = XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.05,
    objective="binary:logistic",
    eval_metric="auc",
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
    early_stopping_rounds=40
)

inicio = time.perf_counter()
modelo_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
tempo_xgb = time.perf_counter() - inicio
arvores_xgb = modelo_xgb.best_iteration + 1

# Configuração e treino do LightGBM
modelo_lgbm = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary",
    metric="auc",
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
inicio = time.perf_counter()
modelo_lgbm.fit(
    X_train, y_train,
    eval_X=X_val, 
    eval_y=y_val, 
    callbacks=[
        early_stopping(40, verbose=False),
        log_evaluation(0)
    ]
)

tempo_lgbm = time.perf_counter() - inicio
arvores_lgbm = modelo_lgbm.best_iteration_

# Configuração e treino do CatBoost
modelo_cat = CatBoostClassifier(
    iterations=800,
    depth=5,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1
)
inicio = time.perf_counter()
modelo_cat.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=40,
    verbose=False
)
tempo_cat = time.perf_counter() - inicio
arvores_cat = modelo_cat.tree_count_

# Probabilidades e limiar comum
def probabilidade_classe_positiva(modelo, X):
    probabilidades = modelo.predict_proba(X)
    indice_positivo = list(modelo.classes_).index(1)
    return probabilidades[:, indice_positivo]

LIMIAR = 0.5
modelos = {
    "XGBoost": modelo_xgb,
    "LightGBM": modelo_lgbm,
    "CatBoost": modelo_cat
}
proba_teste = {}
pred_teste = {}

for nome, m in modelos.items():
    p = probabilidade_classe_positiva(m, X_test)
    proba_teste[nome] = p
    pred_teste[nome] = (p >= LIMIAR).astype(int)

# Métricas comuns
def calcular_metricas(nome, y_true, proba, tempo=np.nan):
    pred = (proba >= LIMIAR).astype(int)
    return {
        "Modelo": nome,
        "Acurácia": accuracy_score(y_true, pred),
        "Precisão": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, proba),
        "Log Loss": log_loss(y_true, proba),
        "Tempo (s)": tempo
    }

# Matrizes de confusão
for nome in modelos.keys():
    cm = confusion_matrix(y_test, pred_teste[nome], labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot()

# Tempo de treino e importância por permutação
from sklearn.inspection import permutation_importance

def auc_validacao(modelo, X, y):
    proba = probabilidade_classe_positiva(modelo, X)
    return roc_auc_score(y, proba)

importancias = {}
for nome, modelo in modelos.items():
    resultado = permutation_importance(
        modelo, X_val, y_val,
        scoring=auc_validacao,
        n_repeats=5, random_state=42
    )
    importancias[nome] = resultado.importances_mean

# Resultados Finais (Tabela Comparativa e Tempos)
resultados = []

# Adicionando o Baseline à lista
resultados.append({
    "Modelo": "Baseline",
    "Acurácia": acc_base,
    "Precisão": "-",
    "Recall": "-",
    "F1": "-",
    "ROC AUC": "-",
    "Log Loss": "-",
    "Tempo (s)": "-"
})

# Calculando as métricas para os 3 modelos usando a função da Etapa 15
resultados.append(calcular_metricas("XGBoost", y_test, proba_teste["XGBoost"], tempo_xgb))
resultados.append(calcular_metricas("LightGBM", y_test, proba_teste["LightGBM"], tempo_lgbm))
resultados.append(calcular_metricas("CatBoost", y_test, proba_teste["CatBoost"], tempo_cat))

# Criando e exibindo a tabela (Etapa 17)
df_resultados = pd.DataFrame(resultados)
print("\nTabela Comparativa no Teste:")
print(df_resultados) 

print("\nÁrvores Efetivas e Tempos (Etapas 18 e 19)")
print(f"XGBoost  - Árvores após early stopping: {arvores_xgb} | Tempo: {tempo_xgb:.3f}s")
print(f"LightGBM - Árvores após early stopping: {arvores_lgbm} | Tempo: {tempo_lgbm:.3f}s")
print(f"CatBoost - Árvores após early stopping: {arvores_cat} | Tempo: {tempo_cat:.3f}s")