import shutil
from shutil import copyfile
import os
from preproc import pagesplitting
from preproc import gender
from preproc import fix_page_rotation
from skimage import io
from pdf2image import convert_from_path
import cv2

#Leitura da pasta base
os.chdir('TRABALHO PROPRIO');
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
shutil.rmtree(dirname)
os.makedirs(dirname, exist_ok=True)

#Mudança para a pasta output
os.chdir(dirname)
os.makedirs('Livros',exist_ok=True)
os.makedirs("FEMALE",exist_ok=True)
os.chdir("FEMALE")
femDir= os.getcwd()
os.chdir('..')
os.makedirs("MALE",exist_ok=True)
os.chdir("MALE")
maleDir= os.getcwd()
os.chdir('..')
os.chdir('Livros')
#criação das pastas
for nome in nomesPDF:
   os.makedirs(nome, exist_ok=True)
   os.chdir(nome)
   currentFolder = os.getcwd()
   copyfile(sourceFolder + "/" + nome + ".pdf", currentFolder + "/" + nome + ".pdf")
   os.chdir("..")

#splitting pages
currentFolder = os.getcwd()
pagesplitting(currentFolder)



#fazer diretorio fem e male


#getting gender
bookFolders = os.listdir('.')
currentFolder=os.getcwd()
for folder in bookFolders:

   bookGender = gender(folder)
   thisFolder=os.getcwd()
   if bookGender == 'M':
      print('O livro ' + folder + ' é menino')
      shutil.move(folder,maleDir)
   if bookGender == 'F':
      print('O livro ' + folder + ' é menina')
      shutil.move(folder,femDir)

os.chdir(femDir)
os.chdir("..")
os.rmdir("Livros")


os.chdir(maleDir)
bookFolders = os.listdir('.')
print(bookFolders)
for book in bookFolders:
   os.chdir(book)
   pages = os.listdir('.')
   for page in pages:
      img = cv2.imread(page)
      img = fix_page_rotation(img,'BOY')
      cv2.imwrite(page + "fixed" + ".png",img)
   os.chdir("..")

os.chdir("..")


os.chdir(femDir)
bookFolders = os.listdir('.')
for book in bookFolders:
   os.chdir(book)
   pages = os.listdir('.')
   for page in pages:
      img = cv2.imread(page)
      img = fix_page_rotation(img,'GIRL')
      cv2.imwrite(page + "fixed" + ".png",img)
   os.chdir("..")


