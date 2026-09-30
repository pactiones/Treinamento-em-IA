import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

y = df["Survived"]
X = df.drop(columns=["Survived", "PassengerId"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cv = StratifiedKFold(5, shuffle=True, random_state=42)

#Transformar, codificar, escalonar, treinar
class Features(BaseEstimator, TransformerMixin): #Transformação
    def __init__(self, usar=()):
        self.usar = usar

    def fit(self, X, y=None):
        self.ticket_counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        X = X.copy()
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        X["FarePerPerson"] = X["Fare"] / X["FamilySize"]
        X["Title"] = (
            X["Name"]
            .str.extract(r",\s*([^.]+)\.", expand=False)
            .str.strip()
            .replace(["Mlle", "Ms"], "Miss")
            .replace("Mme", "Mrs")
        )
        X["Title"] = X["Title"].where(
            X["Title"].isin(["Mr", "Miss", "Mrs", "Master"]),
            "Rare"
        )
        X["AgeGroup"] = pd.cut(
            X["Age"],
            bins=[0, 12, 18, 60, np.inf],
            labels=["Child", "Teenager", "Adult", "Senior"],
            include_lowest=True
        )
        '''
        Hipotese de que talvez a idade dos passageiros faça diferença e tenha prioridade. Talvez crianças recebem prioridade na hora de
        evacuação, enquanto talvez dois adultos com uma diferença de pouca idade não tenha diferença.
        '''
        X["WomanOrChild"] = (
            (X["Sex"] == "female") | (X["Age"] <= 12)
        ).astype(int)
        '''
        Hipotese de que talvez ser criança ou mulher faça diferença e tenha prioridade. Provavelmente crianças e mulheres tenham 
        uma ordem de prioridade durante a evacuação.
        '''
        X["TicketGroupSize"] = (
            X["Ticket"].map(self.ticket_counts_).fillna(1)
        )
        return X.drop(columns=["Name", "Cabin", "Ticket"])
        '''
        Hipótese de que passageiros que compartilhavam o mesmo ticket provavelmente viajavam em grupo, mesmo quando não eram familiares.
        '''


def make_model(num_cols, cat_cols):
    num = Pipeline([
        ("imp", SimpleImputer(strategy="median")), #scaling
        ("sc", RobustScaler())
    ])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")), #encoding
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
    ])
    pre = ColumnTransformer([
        ("num", num, num_cols),
        ("cat", cat, cat_cols)
    ])
    return Pipeline([
        ("feat", Features()),
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000)) #treinar
    ])


# ---------- Passos 9 a 12 · medir, adicionar, comparar, interpretar ----------
NUM = ["Age", "SibSp", "Parch", "Fare"] # 9. baseline
CAT = ["Pclass", "Sex", "Embarked"]

s = cross_val_score(make_model(NUM, CAT), X_train, y_train, cv=cv)
ref = s.mean()
print(f"baseline {ref:.3f} +/- {s.std():.3f}")

LIMIAR = 0.002 # declarado ANTES de ver os numeros

candidatas = [
    ("FamilySize", "num"),
    ("IsAlone", "num"),
    ("FarePerPerson", "num"),
    ("Title", "cat"),
    ("AgeGroup", "cat"),
    ("WomanOrChild", "num"),
    ("TicketGroupSize", "num")
]

num, cat = list(NUM), list(CAT)

for nome, tipo in candidatas: # 10-11. adicionar e comparar
    n2, c2 = (
        (num + [nome], cat)
        if tipo == "num"
        else (num, cat + [nome])
    )

    s = cross_val_score(make_model(n2, c2), X_train, y_train, cv=cv)
    ganho = s.mean() - ref

    print(
        f"{nome:>16} {s.mean():.3f} +/- "
        f"{s.std():.3f} ganho {ganho:+.3f}"
    )

    if ganho > LIMIAR:
        num, cat, ref = n2, c2, s.mean() # 12. interpretar e decidir

print("\nfeatures mantidas:", num, cat)

# Teste: usado UMA vez, no final, com o conjunto escolhido
final = make_model(num, cat).fit(X_train, y_train)

print(
    f"acuracia no teste: "
    f"{final.score(X_test, y_test):.3f}"
)

'''
A feature que apresentou maior impacto geral foi Title, aumentando a acurácia média de 0,802 para 0,819, 
com um ganho de 1,7 ponto percentual. Confirmando a hipótese de que o título contém informações importantes
sobre sexo, idade e posição social, fatores relacionados à prioridade na evacuação.
'''