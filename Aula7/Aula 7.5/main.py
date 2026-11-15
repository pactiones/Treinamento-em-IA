# Etapa 1. Imports
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
from scipy.stats import randint, loguniform
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    log_loss, confusion_matrix
)

# Etapa 2. Criar o dataset
X, y = make_classification(
    n_samples=1800,
    n_features=16,
    n_informative=8,
    n_redundant=4,
    n_repeated=0,
    n_classes=2,
    weights=[0.55, 0.45],
    flip_y=0.02,
    class_sep=1.0,
    random_state=42
)

# Etapas 3 e 4. Explorar e separar o teste
print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Classes:", np.unique(y, return_counts=True))

df = pd.DataFrame(X, columns=[f"x{i}" for i in range(16)])
df["classe"] = y
df.head()
df.describe()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Etapas 5 e 6. Mesmos folds e uma referência
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
splits = list(cv.split(X_train, y_train))

baseline = GradientBoostingClassifier(n_estimators=150, random_state=42)
auc_base = cross_val_score(
    baseline, X_train, y_train, cv=splits, scoring="roc_auc"
).mean()
print(f"AUC baseline: {auc_base:.4f}")

# Etapas 7 e 8. Grid Search
param_grid = {
    "max_depth": [1, 2, 3],
    "learning_rate": [0.03, 0.10],
    "min_samples_leaf": [1, 5]
}

modelo_base = GradientBoostingClassifier(n_estimators=150, random_state=42)
grid = GridSearchCV(
    estimator=modelo_base,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=splits,
    n_jobs=1,
    refit=False,
    return_train_score=True,
    error_score="raise"
)

inicio = time.perf_counter()
grid.fit(X_train, y_train)
tempo_grid = time.perf_counter() - inicio

print("best_params_:", grid.best_params_)
print("best_score_:", grid.best_score_)
print("tempo:", tempo_grid)

# Etapas 9 e 10. Random Search
param_distributions = {
    "max_depth": randint(1, 4),
    "learning_rate": loguniform(0.02, 0.15),
    "min_samples_leaf": randint(1, 6)
}

random_search = RandomizedSearchCV(
    estimator=modelo_base,
    param_distributions=param_distributions,
    n_iter=12,
    scoring="roc_auc",
    cv=splits,
    random_state=42,
    n_jobs=1,
    refit=False,
    return_train_score=True,
    error_score="raise"
)

inicio = time.perf_counter()
random_search.fit(X_train, y_train)
tempo_random = time.perf_counter() - inicio

print("best_params_:", random_search.best_params_)
print("best_score_:", random_search.best_score_)
print("tempo:", tempo_random)

# Etapas 11 e 12. Sampler e pruner
sampler = optuna.samplers.TPESampler(
    seed=42,
    n_startup_trials=5
)

pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=50,
    interval_steps=25,
    n_min_trials=3
)

# Etapa 13. Função objetivo
CHECKPOINTS = [25, 50, 75, 100, 150]

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    }

    modelos = [
        GradientBoostingClassifier(
            n_estimators=CHECKPOINTS[0],
            warm_start=True,
            random_state=42,
            **params
        )
        for _ in splits
    ]

    for checkpoint in CHECKPOINTS:
        aucs = []
        for modelo, (idx_tr, idx_va) in zip(modelos, splits):
            modelo.set_params(n_estimators=checkpoint)
            modelo.fit(X_train[idx_tr], y_train[idx_tr])
            proba = modelo.predict_proba(X_train[idx_va])[:, 1]
            aucs.append(roc_auc_score(y_train[idx_va], proba))

        media_auc = float(np.mean(aucs))
        trial.set_user_attr("ultimo_checkpoint", checkpoint)
        trial.report(media_auc, step=checkpoint)

        if checkpoint < CHECKPOINTS[-1] and trial.should_prune():
            raise optuna.TrialPruned()

    return media_auc

# Etapas 14 e 15. Criar e executar o estudo
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner
)

inicio = time.perf_counter()
study.optimize(objective, n_trials=12, n_jobs=1, show_progress_bar=True)
tempo_optuna = time.perf_counter() - inicio

completos = study.get_trials(states=[TrialState.COMPLETE])
podados = study.get_trials(states=[TrialState.PRUNED])

print("best_params:", study.best_params)
print("best_value:", study.best_value)
print("completos:", len(completos))
print("interrompidos:", len(podados))
print("tempo:", tempo_optuna)

# Etapa 16. Orçamento, trabalho e tempo
N_FOLDS = len(splits)

def arvores_optuna(study):
    return sum(
        t.user_attrs.get("ultimo_checkpoint", 0) * N_FOLDS
        for t in study.trials
    )

arvores_grid = 12 * N_FOLDS * 150
arvores_random = 12 * N_FOLDS * 150
arvores_opt = arvores_optuna(study)

# Etapas 17 a 22. Modelos finais, teste e tabela final
melhores = {
    "Grid": (grid.best_params_, grid.best_score_, tempo_grid),
    "Random": (random_search.best_params_, random_search.best_score_, tempo_random),
    "Optuna": (study.best_params, study.best_value, tempo_optuna)
}

linhas = []
for nome, (params, auc_cv, tempo_busca) in melhores.items():
    final = GradientBoostingClassifier(
        n_estimators=150, random_state=42, **params
    )
    final.fit(X_train, y_train)
    proba = final.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    linhas.append({
        "Estratégia": nome,
        "AUC CV": auc_cv,
        "AUC teste": roc_auc_score(y_test, proba),
        "Acur.": accuracy_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "Log Loss": log_loss(y_test, proba),
        "Busca (s)": tempo_busca
    })