#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto1 Entrega1
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
from skimage.filters import threshold_otsu
from scipy.io import loadmat
import os
import glob
import numpy as np
import skimage.io as io
import requests
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

image_url="https://estaticos.muyinteresante.es/uploads/images/article/57a2ef2a5cafe82d7b8b4567/elefante_0.jpg"
r=requests.get(image_url)
with open("Elefantes", "wb") as f: # se trabaja con f como la abreviación para abrir un archivo para escritura "Elefantes"
	f.write(r.content) #se escribe con .write en el archivo previamente mencionado el contenido de la descarga de la imagen realizado previamente con .content
carga_imagen=io.imread("Elefantes") # se carga la imagen del archivo creado con io.imread
##input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure("VisualizaciónAnotaciones")
plt.subplot(3,2,1)
plt.title("Imagen a color")
plt.imshow(carga_imagen)
plt.axis("off")
plt.tight_layout()
plt.subplot(3,2,3)
plt.title("Anotación Clasificación")
plt.imshow(carga_imagen)
plt.axis("off")
plt.tight_layout()
plt.subplot(3,2,4)
plt.title("Anotación Detección")
plt.axis("off")
plt.imshow(carga_imagen)
plt.tight_layout()
plt.subplot(3,2,5)
plt.title("Anotación Segmentación \nSemántica")
plt.imshow(carga_imagen)
plt.axis("off")
plt.subplot(3,2,6)
plt.title("Anotación Segmentación \nde Instancias")
plt.axis("off")
plt.imshow(carga_imagen)
plt.tight_layout()
##input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
monedaURL="https://web.stanford.edu/class/ee368/Handouts/Lectures/Examples/11-Edge-Detection/Hough_Transform_Circles/coins.png"
monedas = requests.get(monedaURL)
with open("Monedas", "wb") as f: # se trabaja con f como la abreviación para abrir un archivo para escritura "Monedas"
	f.write(monedas.content) #se escribe con .write en el archivo previamente mencionado el contenido de la descarga de la imagen realizado previamente con .content
monedas = io.imread("Monedas") # se carga la imagen del archivo creado con io.imread
print(monedas.shape)
vectorColor = monedas.flatten()
plt.figure("ImagenHistograma")
plt.subplot(1,2,1)
plt.imshow(monedas,cmap="gray")
plt.title("Imagen monedas")
plt.axis('off')
plt.subplot(1,2,2)
plt.hist(vectorColor,bins=256)
plt.title('Histograma imagen monedas')
plt.tight_layout()
plt.show()
plt.savefig("ImagenHistograma")


