#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto1 Entrega1
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
from skimage.filters import threshold_otsu
import cv2
import nibabel
from scipy.io import loadmat
import os
import glob
import numpy as np
import skimage.io as io
import requests
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
##
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
plt.figure("HistogramaMonedas")
plt.subplot(1,2,1)
plt.imshow(monedas,cmap="gray")
plt.title("Imagen monedas")
plt.axis('off')
plt.subplot(1,2,2)
plt.hist(vectorColor,bins=256)
plt.title('Histograma imagen monedas')
plt.tight_layout()
plt.show()
#plt.savefig("HistogramaMonedas",Bbox_inches="tight")
plt.savefig("HistogramaMonedas")
##umbral de binarización de acuerdo al método de Otsu
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
binOtsu=threshold_otsu(monedas)
#monedas_binOtsu=monedas<binOtsu
monedas_binOtsu=monedas>binOtsu
print(binOtsu)
plt.figure("BinOtsu")
plt.title("Binarización de la imagen con Otsu")
plt.imshow(monedas_binOtsu, cmap="gray")
plt.axis('off')
##binarización con percentil 60
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
calculo_percentil60=np.percentile(monedas,60)
monedas_percentil60=monedas>calculo_percentil60 #NO SÉ SI ES ASÍ
plt.figure("BinOtsu")
plt.title("Binarización de la imagen con percentil 60")
plt.imshow(monedas_percentil60, cmap="gray")
plt.axis('off')
##binarización con umbral = 175
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
monedas_umbral175= monedas > 175 #NO SÉ SI ES ASÍ
plt.figure("BinOtsu")
plt.title("Binarización de la imagen con umbral 175")
plt.imshow(monedas_umbral175, cmap="gray")
plt.axis('off')
##selección de dos umbrales arbitrarios y establecer rango
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#FALTA

##subplot para máscaras con segmentaciones en escala de grises
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
#cv2.threshold : como primer parámetro recibe la imágen en escala de grises, el segundo parámetro es el valor del umbral, el tercer parámetro el valor máximo para dar al pixel en caso dado de que sea mayor al umbral? (no estoy segura); como cuarto parámetro recibe el tipo de umbralización. La función retorna como primer output un retval y como segundo output la imagen con el umbral indicado
# retval: For this, our cv2.threshold() function is used, but pass an extra flag, cv2.THRESH_OTSU. For threshold value, simply pass zero. Then the algorithm finds the optimal threshold value and returns you as the second output, retVal. If Otsu thresholding is not used, retVal is same as the threshold value you used
#cv2.THRESH_TOZERO : si el pixel tiene un valor mayor al del umbral que indica el 2do parámetro de cv2.threshold el pixel mantiene el nivel de gris correspondiente a la imagen original; en caso de que el valor del pixel sea menor al umbral (indicado en el parámatro previamente mencionado) se le asigna un 0, es decir negro
retval_Otsu,segmentacion_Otsu=cv2.threshold(monedas,binOtsu,255,cv2.THRESH_TOZERO)
retval_percen60,segmentacion_percentil60=cv2.threshold(monedas,calculo_percentil60,255,cv2.THRESH_TOZERO)
retval_umbral175,segmentacion_umbral175=cv2.threshold(monedas,175,255,cv2.THRESH_TOZERO)

##subplot para máscaras con segmentaciones en escala de grises
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure("MascarasySegmentaciones")
plt.subplot(2,4,1)
plt.imshow(monedas_binOtsu,cmap="gray")
plt.title("Máscara 1:\nOtsu")
plt.axis('off')
#plt.tight_layout()
plt.subplot(2,4,2)
plt.imshow(monedas_percentil60,cmap="gray")
plt.title("Máscara 2:\nPercentil 60")
plt.axis('off')
#plt.tight_layout()
plt.subplot(2,4,3)
plt.imshow(monedas_umbral175,cmap="gray")
plt.title("Máscara 3: Umbral\narbitrario 175")
plt.axis('off')
#plt.tight_layout()
plt.subplot(2,4,4)
plt.imshow(monedas,cmap="gray") #FALTA
plt.title("Máscara 4: Umbral\nrango ____")#FALTA
plt.axis('off')
#plt.tight_layout()
plt.subplot(2,4,5)
plt.imshow(monedas_binOtsu*monedas,cmap="gray")
plt.title("Segmentación 1:\nOtsu")
plt.axis('off')
#plt.tight_layout()
plt.subplot(2,4,6)
plt.imshow(monedas_percentil60*monedas,cmap="gray")
plt.title("Segmentación 2:\nPercentil 60")
plt.axis('off')
#plt.tight_layout()
plt.subplot(2,4,7)
plt.imshow(monedas*monedas_umbral175,cmap="gray")
plt.title("Segmentación 3:\nUmbral 175")
plt.axis('off')
#plt.tight_layout()
plt.subplot(2,4,8)
plt.imshow(monedas,cmap="gray") #FALTA
plt.title("Segmentación 4: Umbral\nrango ____")#FALTA
plt.axis('off')
#plt.tight_layout()

##PROBLEMA BIOMÉDICO
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
archivosresonancias=glob.glob(os.path.join("Heart_Data","Data","*.nii.gz"))
#print(len(archivosresonancias))
for i in archivosresonancias:
	#range(1,len(archivosresonancias)+1):
	#print(i)
	carga=nibabel.load(i)
	print(carga) #después lo comentamos es para ver qué atributo es el que nos sirve para saber la info del enunciado
	#carga.atributo1