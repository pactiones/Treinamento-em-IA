import re
import numpy as np

mensagens = [
    # Spam (rótulo 1)
    "Ganhe um celular grátis hoje",
    "Promoção especial clique para aproveitar",
    "Você ganhou um prêmio resgate agora",
    "Crédito aprovado sem consulta solicite já",
    "Últimas horas para ganhar dinheiro fácil",
    "Clique aqui e receba seu bônus",
    "Oferta exclusiva compre com desconto",
    "Parabéns você foi selecionado para ganhar",
    "Empréstimo rápido com aprovação garantida",
    "Acesse agora sua recompensa grátis",
    "Desconto imperdível só até hoje",
    "Ganhe pontos e troque por prêmios",
    "Seu pacote de dados grátis está disponível",
    "Invista pouco e lucre muito rapidamente",
    "Você tem um prêmio esperando clique aqui",
    "Compre agora e receba outro produto grátis",
    "Promoção termina hoje aproveite o desconto",
    "Resgate seu cupom antes que expire",
    "Ganhe uma viagem sem pagar nada",
    "Oferta de crédito fácil para você",
    "Clique e concorra a um carro novo",
    "Parabéns sua conta recebeu um bônus",
    "Renda extra garantida trabalhando de casa",
    "Aproveite a oferta e compre agora",

    # Normal (rótulo 0)
    "A reunião começa às nove da manhã",
    "Enviei o relatório para sua revisão",
    "Vamos almoçar juntos depois da aula",
    "A consulta foi marcada para terça-feira",
    "Pode trazer o livro para a biblioteca",
    "O projeto será apresentado na próxima semana",
    "Avise quando chegar ao escritório",
    "A professora adiou a entrega do trabalho",
    "Comprei pão e leite no mercado",
    "O ônibus chegou atrasado hoje",
    "Preciso estudar para a prova de matemática",
    "Minha família vai visitar no domingo",
    "A equipe discutiu os resultados do projeto",
    "Deixei as chaves sobre a mesa",
    "A aula de programação termina ao meio-dia",
    "Vamos conversar sobre o planejamento amanhã",
    "O documento está salvo na pasta compartilhada",
    "Meu irmão trabalha perto do centro",
    "A previsão indica chuva durante a tarde",
    "Reservei uma mesa para o jantar",
    "O computador precisa de uma atualização",
    "A reunião de hoje foi cancelada",
    "Estou lendo um livro sobre história",
    "Mandei uma mensagem para confirmar o horário"
]

y = np.array([1] * 24 + [0] * 24)

# Embaralha separadamente cada classe e reserva 6 mensagens de cada
rng = np.random.default_rng(42)
indices_teste = []

for classe in [0, 1]:
    indices_classe = np.where(y == classe)[0]
    indices_classe = rng.permutation(indices_classe)
    indices_teste.extend(indices_classe[:6])

indices_teste = np.array(indices_teste)
indices_treino = np.array([
    i for i in range(len(mensagens))
    if i not in indices_teste
])

textos_treino = [mensagens[i] for i in indices_treino]
textos_teste = [mensagens[i] for i in indices_teste]
y_treino = y[indices_treino]
y_teste = y[indices_teste]

print("Mensagens no total:", len(mensagens))
print("Treino:", len(textos_treino), "-- Teste:", len(textos_teste))
print("Classes no treino:", np.unique(y_treino, return_counts=True))
print("Classes no teste:", np.unique(y_teste, return_counts=True))

def tokenizar(texto):
    return re.findall(r"[a-záàâãéêíóôõúç]+", texto.lower())

def construir_vocabulario(textos_treino):
    palavras = set()
    for texto in textos_treino:
        palavras.update(tokenizar(texto))
    return {palavra: j for j, palavra in enumerate(sorted(palavras))}

def vetorizar(textos, vocab):
    X = np.zeros((len(textos), len(vocab)), dtype=int)

    for i, texto in enumerate(textos):
        for palavra in tokenizar(texto):
            if palavra in vocab:
                X[i, vocab[palavra]] += 1

    return X

vocab = construir_vocabulario(textos_treino)
X_treino = vetorizar(textos_treino, vocab)
X_teste = vetorizar(textos_teste, vocab)

print("\nFormato de X_treino:", X_treino.shape)
print("Formato de X_teste:", X_teste.shape)
print("Palavras no vocabulário:", len(vocab))

class NaiveBayesDoZero:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        # P(classe): proporção de mensagens de cada classe no treino
        priori = np.array([
            np.mean(y == classe)
            for classe in self.classes_
        ])

        # Para cada classe, soma quantas vezes cada palavra aparece
        contagens = np.array([
            X[y == classe].sum(axis=0)
            for classe in self.classes_
        ])

        # Total de palavras observadas em cada classe
        totais = contagens.sum(axis=1, keepdims=True)

        # Laplace: evita probabilidades iguais a zero
        prob = (contagens + self.alpha) / (
            totais + self.alpha * X.shape[1]
        )

        self.log_priori_ = np.log(priori)
        self.log_prob_ = np.log(prob)

        return self

    def _log_pontuacoes(self, X):
        # Cada linha recebe uma pontuação para cada classe
        return X @ self.log_prob_.T + self.log_priori_

    def predict(self, X):
        pontuacoes = self._log_pontuacoes(X)
        indices_vencedores = np.argmax(pontuacoes, axis=1)
        return self.classes_[indices_vencedores]

    def predict_proba(self, X):
        pontuacoes = self._log_pontuacoes(X)

        # Estabiliza o cálculo antes de aplicar exponencial
        pontuacoes = pontuacoes - pontuacoes.max(axis=1, keepdims=True)

        valores = np.exp(pontuacoes)
        return valores / valores.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

modelo = NaiveBayesDoZero(alpha=1.0)
modelo.fit(X_treino, y_treino)

y_pred = modelo.predict(X_teste)
probabilidades = modelo.predict_proba(X_teste)

print("\nRótulos reais:  ", y_teste)
print("Previsões:      ", y_pred)
print("Probabilidades:")
print(probabilidades)

def acuracia(y_real, y_previsto):
    return np.mean(y_real == y_previsto)

def matriz_confusao(y_real, y_previsto, classes):
    matriz = np.zeros((len(classes), len(classes)), dtype=int)

    for i, classe_real in enumerate(classes):
        for j, classe_prevista in enumerate(classes):
            matriz[i, j] = np.sum(
                (y_real == classe_real) &
                (y_previsto == classe_prevista)
            )

    return matriz

print("\nAcurácia:", acuracia(y_teste, y_pred))
print("Ordem das classes:", modelo.classes_)
print("Matriz de confusão:")
print(matriz_confusao(y_teste, y_pred, modelo.classes_))

mensagens_novas = [
    "Parabéns você ganhou um prêmio grátis",
    "A reunião começa amanhã de manhã"
]

X_novas = vetorizar(mensagens_novas, vocab)
classes_novas = modelo.predict(X_novas)
probs_novas = modelo.predict_proba(X_novas)

for texto, classe, probs in zip(mensagens_novas, classes_novas, probs_novas):
    nome_classe = "spam" if classe == 1 else "normal"

    print()
    print(texto)
    print("Classificação:", nome_classe)
    print("P(normal):", round(probs[0], 3))
    print("P(spam):  ", round(probs[1], 3))

"""
Mensagem “Parabéns você ganhou um prêmio grátis” — classificada como spam. O modelo teve essa 
classificação porque palavras como “ganhou”, “prêmio” e “grátis” aparecem associadas a mensagens 
de spam nos dados de treino. A combinação dessas palavras elevou a probabilidade estimada de spam para 0,999. 

Mensagem “A reunião começa amanhã de manhã” — classificada como normal. Palavras como “reunião”, “amanhã” e 
“manhã” podem ter favorecido a classe normal por aparecerem mais em mensagens normais no treino. O modelo 
estimou P(normal) = 0,981 e P(spam) = 0,019. Essa decisão depende das palavras que ele viu no conjunto de treino, 
não de uma compreensão real do significado da mensagem.
"""