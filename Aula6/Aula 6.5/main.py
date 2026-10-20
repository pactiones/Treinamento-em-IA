# Aula 6.5 · Exercício guiado: as métricas em dez etapas
# preveja: escreva o resultado esperado antes de executar
import numpy as np

# 01 · Carregar y_real e y_pred
# y_real = resposta certa, y_pred = o que o modelo previu
# classe 1 = spam, classe 0 = mensagem normal
y_real = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 0, 1, 1])

# 02 · Acertos e accuracy · preveja: ____
# accuracy = acertos / total
acertos = (y_real == y_pred)
print("acertos:", acertos.sum())
print("accuracy:", acertos.mean())

# 03 · Contar TP, TN, FP e FN
# TP: era 1 e previu 1 | TN: era 0 e previu 0
# FP: era 0 e previu 1 | FN: era 1 e previu 0
def contar(y_real, y_pred):
    tp = np.sum((y_real == 1) & (y_pred == 1))
    tn = np.sum((y_real == 0) & (y_pred == 0))
    fp = np.sum((y_real == 0) & (y_pred == 1))
    fn = np.sum((y_real == 1) & (y_pred == 0))
    return tp, tn, fp, fn

tp, tn, fp, fn = contar(y_real, y_pred)
print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)

# 04 · Construir a matriz de confusão
# linhas = classe real, colunas = classe prevista
# [[TN, FP],
#  [FN, TP]]
def matriz_confusao(y_real, y_pred):
    tp, tn, fp, fn = contar(y_real, y_pred)
    return np.array([[tn, fp],
                     [fn, tp]])

print(matriz_confusao(y_real, y_pred))

# 05 · Implementar precision · preveja: ____
# precision = TP / (TP + FP)
def divide(num, den):
    return num / den if den != 0 else 0.0

def precision(tp, fp):
    return divide(tp, tp + fp)

p = precision(tp, fp)
print("precision:", p)

# 06 · Implementar recall
# recall = TP / (TP + FN)
def recall(tp, fn):
    return divide(tp, tp + fn)

r = recall(tp, fn)
print("recall:", r)

# 07 · Implementar F1-score · preveja: ____
# F1 = 2·P·R / (P + R)
def f1_score(p, r):
    return divide(2 * p * r, p + r)

print("F1:", f1_score(p, r))

# 08 · Estender para três classes
# 0 = Adelie, 1 = Chinstrap, 2 = Gentoo
# M[i, j] = reais da classe i previstos como j
y_real3 = np.array([0]*10 + [1]*5 + [2]*10)
y_pred3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                    0, 1, 1, 1, 2,
                    1, 2, 2, 2, 2, 2, 2, 2, 2, 2])

classes = np.unique(np.concatenate([y_real3, y_pred3]))
k = len(classes)

M = np.zeros((k, k), dtype=int)
for real, prev in zip(y_real3, y_pred3):
    M[real, prev] += 1
print(M)

# 09 · Montar o relatório de classificação
# one-vs-rest: TP = diagonal, FP = resto da coluna,
# FN = resto da linha, support = soma da linha
def razao(num, den):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den != 0)
    return out

tp3 = np.diag(M)
fp3 = M.sum(axis=0) - tp3      # coluna sem a diagonal
fn3 = M.sum(axis=1) - tp3      # linha sem a diagonal
prec3 = razao(tp3, tp3 + fp3)
rec3  = razao(tp3, tp3 + fn3)
f13   = razao(2 * prec3 * rec3, prec3 + rec3)
support = M.sum(axis=1)
acc3 = tp3.sum() / M.sum()

nomes = ["Adelie", "Chinstrap", "Gentoo"]
print(f"{'':>14}{'precision':>11}{'recall':>9}{'f1-score':>10}{'support':>9}")
for i in range(k):
    print(f"{nomes[i]:>14}{prec3[i]:>11.2f}{rec3[i]:>9.2f}{f13[i]:>10.2f}{support[i]:>9}")
print(f"{'accuracy':>14}{'':>20}{acc3:>10.2f}{support.sum():>9}")
print(f"{'macro avg':>14}{prec3.mean():>11.2f}{rec3.mean():>9.2f}{f13.mean():>10.2f}{support.sum():>9}")
print(f"{'weighted avg':>14}"
      f"{np.average(prec3, weights=support):>11.2f}"
      f"{np.average(rec3, weights=support):>9.2f}"
      f"{np.average(f13, weights=support):>10.2f}{support.sum():>9}")

# 10 · ROC e AUC do exemplo binário · preveja: ____
# TPR = TP / (TP + FN)    FPR = FP / (FP + TN)
# scores = probabilidades que o modelo deu para cada mensagem;
# o y_pred da etapa 01 é exatamente scores >= 0.5
scores = np.array([0.95, 0.10, 0.70, 0.55, 0.85, 0.38, 0.45, 0.25, 0.62, 0.78])

def curva_roc(y_real, scores):
    ordem = np.argsort(-scores, kind="mergesort")
    y, s = y_real[ordem], scores[ordem]
    fim = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp = np.cumsum(y)[fim]
    fp = (fim + 1) - tp
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.r_[0, tp] / P
    fpr = np.r_[0, fp] / N
    return fpr, tpr

# AUC pela regra dos trapézios: A = (x2 - x1) * (y1 + y2) / 2
def auc(fpr, tpr):
    larg = fpr[1:] - fpr[:-1]
    alt = (tpr[1:] + tpr[:-1]) / 2
    return np.sum(larg * alt)

fpr, tpr = curva_roc(y_real, scores)
print("FPR:", fpr)
print("TPR:", tpr)
print("AUC:", auc(fpr, tpr))

def meu_classification_report(y_real, y_pred):
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    # Considera todas as classes que aparecem nos valores reais ou previstos.
    classes = np.unique(np.concatenate([y_real, y_pred]))
    k = len(classes)

    # Linhas: classe real. Colunas: classe prevista.
    M = np.zeros((k, k), dtype=int)
    for real, previsto in zip(y_real, y_pred):
        i = np.where(classes == real)[0][0]
        j = np.where(classes == previsto)[0][0]
        M[i, j] += 1
        
    tp = np.diag(M)
    fp = M.sum(axis=0) - tp
    fn = M.sum(axis=1) - tp
    support = M.sum(axis=1)

    # precision = TP / (TP + FP)
    denominador_precision = tp + fp
    precision = np.zeros(k, dtype=float)
    np.divide(
        tp, denominador_precision,
        out=precision,
        where=denominador_precision != 0
    )

    # recall = TP / (TP + FN)
    denominador_recall = tp + fn
    recall = np.zeros(k, dtype=float)
    np.divide(
        tp, denominador_recall,
        out=recall,
        where=denominador_recall != 0
    )

    # F1 = 2 * precision * recall / (precision + recall)
    denominador_f1 = precision + recall
    f1 = np.zeros(k, dtype=float)
    np.divide(
        2 * precision * recall,
        denominador_f1,
        out=f1,
        where=denominador_f1 != 0
    )

    accuracy = tp.sum() / M.sum()

    # Macro: média simples
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    # Weighted: media ponderada
    total = support.sum()
    weighted_precision = np.sum(precision * support) / total
    weighted_recall = np.sum(recall * support) / total
    weighted_f1 = np.sum(f1 * support) / total

    resultados = {
        "classes": classes,
        "matriz_confusao": M,
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "support": support,
        "accuracy": accuracy,
        "macro avg": (
            macro_precision, macro_recall, macro_f1, total
        ),
        "weighted avg": (
            weighted_precision, weighted_recall, weighted_f1, total
        ),
    }

    largura_nome = max(14, max(len(str(c)) for c in classes) + 2)
    print(
        f"{'':>{largura_nome}}"
        f"{'precision':>11}{'recall':>9}{'f1-score':>10}{'support':>9}"
    )

    for i, classe in enumerate(classes):
        print(
            f"{str(classe):>{largura_nome}}"
            f"{precision[i]:>11.2f}"
            f"{recall[i]:>9.2f}"
            f"{f1[i]:>10.2f}"
            f"{support[i]:>9}"
        )

    print(
        f"{'accuracy':>{largura_nome}}"
        f"{'':>20}"
        f"{accuracy:>10.2f}"
        f"{total:>9}"
    )

    print(
        f"{'macro avg':>{largura_nome}}"
        f"{macro_precision:>11.2f}"
        f"{macro_recall:>9.2f}"
        f"{macro_f1:>10.2f}"
        f"{total:>9}"
    )

    print(
        f"{'weighted avg':>{largura_nome}}"
        f"{weighted_precision:>11.2f}"
        f"{weighted_recall:>9.2f}"
        f"{weighted_f1:>10.2f}"
        f"{total:>9}"
    )

    return resultados

relatorio = meu_classification_report(y_real3, y_pred3)

print("\nMatriz de confusão:")
print(relatorio["matriz_confusao"])