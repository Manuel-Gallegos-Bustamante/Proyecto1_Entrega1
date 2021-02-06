#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto1 Entrega1
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
##
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
plt.subplot(3,2,3)
plt.title("Anotación Clasificación")
plt.imshow(io.imread("Clasificacion.png"))
plt.axis("off")
plt.subplot(3,2,4)
plt.title("Anotación Detección")
plt.axis("off")
plt.imshow(io.imread("Deteccion.jpeg"))
plt.subplot(3,2,5)
plt.title("Anotación Segmentación \nSemántica")
plt.imshow(io.imread("Seg_Semantica.jpeg"))
plt.axis("off")
plt.subplot(3,2,6)
plt.title("Anotación Segmentación \nde Instancias")
plt.axis("off")
plt.imshow(io.imread("Seg_Instancias.jpeg"))
plt.tight_layout()
plt.show()
plt.savefig("VisualizaciónAnotaciones",Bbox_inches="tight")
##input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
monedaURL="https://web.stanford.edu/class/ee368/Handouts/Lectures/Examples/11-Edge-Detection/Hough_Transform_Circles/coins.png"
monedas = requests.get(monedaURL)
with open("Monedas", "wb") as f: # se trabaja con f como la abreviación para abrir un archivo para escritura "Monedas"
	f.write(monedas.content) #se escribe con .write en el archivo previamente mencionado el contenido de la descarga de la imagen realizado previamente con .content
monedas = io.imread("Monedas") # se carga la imagen del archivo creado con io.imread
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
monedas_binOtsu=monedas>binOtsu
#print(binOtsu)
plt.figure("BinOtsu")
plt.title("Binarización de la imagen con Otsu")
plt.imshow(monedas_binOtsu, cmap="gray")
plt.axis('off')
##binarización con percentil 60
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
calculo_percentil60=np.percentile(monedas,60)
monedas_percentil60=monedas>calculo_percentil60
#print(calculo_percentil60)
plt.figure("Percentil 60")
plt.title("Binarización de la imagen con percentil 60")
plt.imshow(monedas_percentil60, cmap="gray")
plt.axis('off')
##binarización con umbral = 175
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
monedas_umbral175 = monedas > 175
plt.figure("Umbral 175")
plt.title("Binarización de la imagen con umbral 175")
plt.imshow(monedas_umbral175, cmap="gray")
plt.axis('off')
##selección de dos umbrales arbitrarios y establecer rango
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
monedas_copia = monedas.copy()
#monedas_copia = monedas_copia.flatten()
#print(monedas_copia)
for i in range(0, len(monedas_copia)):
	for j in range(0, len(monedas_copia[i])):
		if monedas_copia[i][j] > 65 and monedas_copia[i][j] < 250:
			monedas_copia[i][j] = 255
		else:
			monedas_copia[i][j] = 0
plt.figure("Umbral arbitrario")
plt.title("Umbral arbitrario") 
plt.imshow(monedas_copia, cmap='gray')
plt.axis('off')
##subplot para máscaras con segmentaciones en escala de grises
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure("MascarasySegmentaciones")
plt.subplot(2,4,1)
plt.imshow(monedas_binOtsu,cmap="gray")
plt.title("Máscara 1:\nOtsu")
plt.axis('off')
plt.subplot(2,4,2)
plt.imshow(monedas_percentil60,cmap="gray")
plt.title("Máscara 2:\nPercentil 60")
plt.axis('off')
plt.subplot(2,4,3)
plt.imshow(monedas_umbral175,cmap="gray")
plt.title("Máscara 3: Umbral\narbitrario 175")
plt.axis('off')
plt.subplot(2,4,4)
plt.imshow(monedas_copia,cmap="gray")
plt.title("Máscara 4: Umbral\nrango 65-250")
plt.axis('off')
plt.subplot(2,4,5)
plt.imshow(monedas_binOtsu*monedas,cmap="gray")
plt.title("Segmentación 1:\nOtsu")
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(monedas_percentil60*monedas,cmap="gray")
plt.title("Segmentación 2:\nPercentil 60")
plt.axis('off')
plt.subplot(2,4,7)
plt.imshow(monedas*monedas_umbral175,cmap="gray")
plt.title("Segmentación 3:\nUmbral 175")
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(monedas_copia * monedas,cmap="gray")
plt.title("Segmentación 4: Umbral\nrango 65-250")
plt.axis('off')
plt.tight_layout()
plt.show()

##PROBLEMA BIOMÉDICO
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
archivosresonancias=glob.glob(os.path.join("Heart_Data","Data","*.nii.gz"))
info = {}
for i in archivosresonancias:
	carga = nibabel.load(i)
	paciente = (str(carga.header['intent_name']).replace("b'",""))[:-1]
	if paciente not in info:
		x, y = carga.shape
		info[paciente] = {'filas':x, 'columnas':y,'cortes':int(carga.header['slice_end'])}
	#print(carga) #después lo comentamos es para ver qué atributo es el que nos sirve para saber la info del enunciado
	#carga.atributo1
	#Atributo identificar paciente -> intent_name     : b'Patient 3'
	#Atributo identificar #total cortes -> slice_end       : 35
	#Atributo identificar #corte -> descrip         : b'Slice 1'
	#Atributo resolución corte->  bitpix          : 16

vol1=np.zeros([info['Patient 12']['filas'], info['Patient 12']['columnas'],info['Patient 12']['cortes']])
vol2=np.zeros([info['Patient 14']['filas'], info['Patient 14']['columnas'],info['Patient 14']['cortes']])
vol3=np.zeros([info['Patient 3']['filas'], info['Patient 3']['columnas'],info['Patient 3']['cortes']])

#print(vol1.shape, vol2.shape, vol3.shape)
for i in archivosresonancias:
	carga = nibabel.load(i)
	paciente = (str(carga.header['intent_name']).replace("b'",""))[:-1]
	corte = int((str(carga.header['descrip']).replace("b'Slice ", ""))[:-1])
	if paciente == 'Patient 12':
		vol1[:,:,corte] = carga.get_fdata()
	elif paciente == 'Patient 14':
		vol2[:,:,corte] = carga.get_fdata()
	elif paciente == 'Patient 3':
		vol3[:,:,corte] = carga.get_fdata()
##
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.ion()
plt.show()
for i in range(len(vol1[0,0])):
   plt.imshow(vol1[:,:,i], cmap='gray')
   plt.axis('off')
   plt.title(f'Resonancia paciente 12, corte {i}')
   plt.draw()
   plt.pause(0.001)
   plt.clf()
##
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.ion()
plt.show()
for i in range(len(vol2[0,0])):
   plt.imshow(vol2[:,:,i], cmap='gray')
   plt.axis('off')
   plt.title(f'Resonancia paciente 14, corte {i}')
   plt.draw()
   plt.pause(0.001)
   plt.clf()
##
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.ion()
plt.show()
for i in range(len(vol3[0,0])):
   plt.imshow(vol3[:,:,i], cmap='gray')
   plt.axis('off')
   plt.title(f'Resonancia paciente 3, corte {i}')
   plt.draw()
   plt.pause(0.001)
   plt.clf()