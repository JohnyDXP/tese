from shutil import copyfile
import os

#Leitura da pasta base
livros = os.listdir('Livros')
nomesPDF = []
os.chdir("Livros")
sourceFolder = os.getcwd()
os.chdir("..")

#Obter os nomes dos livros num array
for livro in livros:
   nomesPDF.append(livro[0:4])

#criação do ficheiro de output
dirname = "OUTPUT"
os.makedirs(dirname, exist_ok=True)

#Mudança para a pasta output
os.chdir(dirname)

#criação das pastas
for nome in nomesPDF:
   os.makedirs(nome, exist_ok=True)
   os.chdir(nome)
   currentFolder = os.getcwd()
   copyfile(sourceFolder + "/" + nome + ".pdf", currentFolder + "/" + nome + ".pdf")
   os.chdir("..")


