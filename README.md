# proyecto_reconocimiento_facial_BUAP
En este repositorio se realiza el proyecto de reconocimiento facial usando la base de datos de CelebA como parte del curso de Redes Neuronales otoño 2022 de la Facultad de Físico Matemáticas de la BUAP.

##OBJETIVO:
  1.  Entrenar  una  red  convolucional  para  predecir  los  atributos  de  la  basede datos CelebA que contiene imagenes de rostros etiquetadas con 40 atributos.
  2.  Posteriormente  construir  un  modelo  con  las  capas  convolucionales  del modelo entrenado pero quitandole el   clasificador (capas densas del final).
  3.  Aumentar un clasificador apropiado para distinguir solo si es el rostro o no (una sola neurona de salida).
  4.  Congelar los pesos de la parte pre-entrenada (o primeras capas)5.  Entrenar este modelo.


##DATASET
La base de datos consiste de una colección de 202,599 imagenes de rostros de celebridades y además un archivo de texto binario que tiene etiquetado 40 atributos de cada imagen, por ejemplo: 
![000001](https://user-images.githubusercontent.com/80428982/201582832-e54f4048-5f73-4f22-a2bd-f37ecf7fdc7d.jpg)
el archivo de texto tiene distintos atributos, por ejemplo: cabello negro, cabello rubio, lentes, nariz grande, etc.
![image](https://user-images.githubusercontent.com/80428982/201583035-d2e883b4-c96f-4ccb-ac0a-50883b125f2f.png)
donde 1 se refiere a verdadero y -1 a falso.


## 1. Red convolucional para predecir atributos de la base CelebA.
### Cargar y procesar los datos de CelebA. 
Una vez obtenida la base de datos tenemos que cargar los dos tipos de datos: una carpeta con puras imagenes de rostros y un archivo de texto binario con los atributos de cada imagen. 

Comenzamos con el segundo porque al ser un archivo de texto con datos estructurados es más fácil de manejar si lo tratamos como un csv, es decir, un archivo Comma Separated Value, el cual tiene formato de tabla, para convertir el archivo de texto a un archivo CSV usamos la paquetería de Pandas.

Para las imagenes definimos la ruta en la que se encuentra la carpeta e iniciamos el procesamiento de imagenes que en este caso consiste en asignarle el nombre de la imagen, sus atributos y la imagen misma en un solo dato el cual llamamos imagen etiquetada, transformar la imagen de 3 canales RGB a un tensor en escala de grises, redimensionar la imagen y por ultimo normalizar la intensidad de los tonos de grises, el cual va de 0 (negro) a 255 (blanco).

Una vez procesados los datos podemos comenzar con el modelo predictivo, el cual consistirá de una secuencia de redes neuronales convolucionales 2D con su respectiva activación ReLu y un MaxPooling2D para procesar las matrices de imagenes y por ultimo se "aplanara" la red para usar una red neuronal densa y una activación sigmoide que procese los datos para poder predecir si una imagen contiene o no cada atributo de los 40 considerados. Para este tipo de problemas es conveniente usar una funcion de costo binaria como el Binary_crossentropy y el optimizador RMSprop (Root Mean Squared Propagation) con una metrica binaria.



###2. Generacion de Imagenes con Image Data Generator.




###3. RNA para reconocimiento facial




