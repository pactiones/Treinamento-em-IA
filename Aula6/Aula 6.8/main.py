import re
import numpy as np
 
def tokenizar(texto):
    return re.findall(r"[a-záàâãéêíóôõúç]+", texto.lower())
 
def construir_vocabulario(textos_treino):
    palavras = set()
    for txt in textos_treino:
        palavras.update(tokenizar(txt))
    return {p: j for j, p in enumerate(sorted(palavras))}
 
def vetorizar(textos, vocab):
    X = np.zeros((len(textos), len(vocab)), dtype=int)
    for i, txt in enumerate(textos):
        for t in tokenizar(txt):
            if t in vocab:
                X[i, vocab[t]] += 1
    return X
 
textos = [
    "promoção grátis grátis",
    "ganhe promoção",
    "reunião hoje",
    "agenda reunião"
]
 
y = np.array([1, 1, 0, 0])
 
vocab = construir_vocabulario(textos)
X = vetorizar(textos, vocab)
 
print(tokenizar("Ganhe já, GRÁTIS!"))
print(vocab)
print(X.shape)
print(X)
print(np.unique(y))
 
alpha = 1.0
classes = np.unique(y)
 
# 6 · Prioris P(C)
priori = np.array([np.mean(y == c) for c in classes])
 
# 7 · Contagens por classe
contagens = np.array([X[y == c].sum(axis=0)
                      for c in classes])
 
# 8 · Laplace
totais = contagens.sum(axis=1, keepdims=True)
prob = (contagens + alpha) / (totais + alpha * X.shape[1])
assert np.allclose(prob.sum(axis=1), 1)
 
# 9 · Log-probabilidades
log_priori = np.log(priori)
log_prob = np.log(prob)
 
def log_pontuacoes(X, log_priori, log_prob):
    return X @ log_prob.T + log_priori
 
def prever(X, classes, log_priori, log_prob):
    S = log_pontuacoes(X, log_priori, log_prob)
    return classes[np.argmax(S, axis=1)]
 
def prever_proba(X, log_priori, log_prob):
    S = log_pontuacoes(X, log_priori, log_prob)
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)
 
def acuracia(y_true, y_pred):
    return np.mean(y_true == y_pred)
 
x = vetorizar(["grátis promoção"], vocab)
 
print(log_pontuacoes(x, log_priori, log_prob))
print(prever(x, classes, log_priori, log_prob))
print(prever_proba(x, log_priori, log_prob))
 
 
class NaiveBayesDoZero:
 
    def __init__(self, alpha=1.0):
        self.alpha = alpha
 
    def fit(self, X, y):
        self.classes_ = np.unique(y)
 
        priori = np.array([
            np.mean(y == c)
            for c in self.classes_
        ])
 
        self.contagens_ = np.array([
            X[y == c].sum(axis=0)
            for c in self.classes_
        ])
 
        totais = self.contagens_.sum(axis=1, keepdims=True)
 
        prob = (self.contagens_ + self.alpha) / (
            totais + self.alpha * X.shape[1]
        )
 
        self.log_priori_ = np.log(priori)
        self.log_prob_ = np.log(prob)
 
        return self
 
    def predict(self, X):
        S = X @ self.log_prob_.T + self.log_priori_
        return self.classes_[np.argmax(S, axis=1)]
 
    def predict_proba(self, X):
        S = X @ self.log_prob_.T + self.log_priori_
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(axis=1, keepdims=True)
 
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
 
 
# Divisão treino/teste
teste = [i for i in range(len(textos)) if i % 4 == 3]
treino = [i for i in range(len(textos)) if i % 4 != 3]
 
textos_treino = [textos[i] for i in treino]
textos_teste = [textos[i] for i in teste]
 
y_treino = y[treino]
y_teste = y[teste]
 
# Vocabulário somente do treino
vocab = construir_vocabulario(textos_treino)
 
X_treino = vetorizar(textos_treino, vocab)
X_teste = vetorizar(textos_teste, vocab)
 
# Treinar
modelo = NaiveBayesDoZero(alpha=1.0)
modelo.fit(X_treino, y_treino)
 
# Prever
y_pred = modelo.predict(X_teste)
 
print("Real:", y_teste)
print("Previsto:", y_pred)
print("Acurácia:", modelo.score(X_teste, y_teste))
 
# Nova mensagem
mensagem = ["Parabéns! Você ganhou uma promoção!"]
X_novo = vetorizar(mensagem, vocab)
 
print("Classe:", modelo.predict(X_novo))
print("Probabilidades:", modelo.predict_proba(X_novo))