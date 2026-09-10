import pandas as pd

data = {
    'Nome': ['Mateus', 'João', 'Rafael', 'Gabriel', 'Lucas', 'Pedro'],
    'Idade': [20, 19, 21, 18, 22, 23],
    'Nota': [8.5, 7.0, 9.0, 6.5, 8.0, 7.5],
    'Curso': ['GES', 'GEC', 'GES', 'GEC', 'GES', 'GEC']
}

df = pd.DataFrame(data)

print("5 primeiras linhas do dataframe: \n", df.head(5))
print("\n")
print("5 ultimas linhas do dataframe: \n", df.tail(5))
print("\n")
print("Tamanho do dataframe: \n", df.shape)
print("\n")
print("Colunas do dataframe: \n", df.columns)
print("\n")
print("Informações do dataframe: \n", df.info())
print("\n")
print("Estatísticas do dataframe: \n", df.describe())
print("\n")
print("Coluna nota: \n", df['Nota'])
print("\n")
print("Coluna nome e nota: \n", df[['Nome', 'Nota']])
print("\n")
print("Nota maior/igual a 7: \n", df[df['Nota'] >= 7])
print("\n")
print("Menor para maior nota: \n", df.sort_values("Nota"))
print("\n")
print("Maior para menor nota: \n", df.sort_values("Nota", ascending=False))
print("\n")
print("Valores ausentes: \n", df.isnull().sum())