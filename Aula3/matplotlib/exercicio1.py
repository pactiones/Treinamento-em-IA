import matplotlib.pyplot as plt

Escolinha_de_IA = {
    "Alunos": ["Evelyn", "Marcelo", "Pedro", "Gustavo",
                "Mateus", "Felipe"],
    "Periodo": ["P3", "P4", "P5", "P6", "P7", "P8"]
}

Aulas = {
    "Aulas": ["24", "26", "23"],
    "Faltas": ["8", "5", "3"]
}

#Grafico de barras
plt.bar(Escolinha_de_IA["Alunos"], Escolinha_de_IA["Periodo"])
plt.xlabel("Alunos")
plt.ylabel("Período")
plt.title("Periodo dos alunos da Escolinha de IA")
plt.show()

#Histograma
plt.hist(Escolinha_de_IA["Periodo"])
plt.xlabel("Alunos")
plt.ylabel("Período")
plt.title("Periodo dos alunos da Escolinha de IA")
plt.show()

#Scatter
plt.scatter(Aulas["Aulas"], Aulas["Faltas"])
plt.xlabel("Aulas")
plt.ylabel("Faltas")
plt.title("Faltas por Aulas")

plt.show()