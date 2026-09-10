import numpy as np

notas = np.array([50, 67, 88, 91, 100])

media = np.mean(notas)
maiorNota = np.max(notas)
menorNota = np.min(notas)
soma = np.sum(notas)

print("Média:", media)
print("Maior nota:", maiorNota)
print("Menor nota:", menorNota)
print("Soma:", soma)