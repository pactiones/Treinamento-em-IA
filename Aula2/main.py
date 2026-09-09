nomes = ["Evelyn", "Pedro", "Gustavo", "Mateus", "Felipe", "Marcelo"]
notas = [8, 6, 9, 5, 7, 10]


aprovado = 0
reprovado = 0

for nome in nomes:
    print("Aluno: ", nome)
    
i = 0

while i < 6:
    if notas[i] >= 7:
        aprovado +=1
    else:
        reprovado +=1
        
    i+=1

Treinamento_IA = {
    "Evelyn": {"nota": 8, "situacao": "Aprovada"},
    "Pedro": {"nota": 6, "situacao": "Reprovado"},
    "Gustavo": {"nota": 9, "situacao": "Aprovado"},
    "Mateus": {"nota": 5, "situacao": "Reprovado"},
    "Felipe": {"nota": 7, "situacao": "Aprovado"},
    "Marcelo": {"nota": 10, "situacao": "Aprovado"}
}

print("Total de alunos: 6; Aprovados; ", aprovado, "Reprovados: ", reprovado)

print(Treinamento_IA)
