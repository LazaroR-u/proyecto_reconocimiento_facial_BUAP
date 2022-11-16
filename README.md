# 🚀 PROYECTO RECONOCIMIENTO FACIAL CON CELEBA Y PYTHON. BUAP.
En este repositorio se realiza el proyecto de reconocimiento facial usando la base de datos de CelebA como parte del curso de Redes Neuronales otoño 2022 de la Facultad de Físico Matemáticas de la BUAP.

## 🔎 OBJETIVO:
  1.  Entrenar  una  red  convolucional  para  predecir  los  atributos  de  la  base de datos CelebA que contiene imagenes de rostros etiquetadas con 40 atributos.
  
  2.  Posteriormente  construir  un  modelo  con  las  capas  convolucionales  del modelo entrenado pero quitandole el   clasificador (capas densas del final).
   
  3.  Aumentar un clasificador apropiado para distinguir solo si es el rostro o no (una sola neurona de salida).
  
  4.  Congelar los pesos de la parte pre-entrenada (o primeras capas)
  
  5.  Entrenar este modelo.


## 👨👱‍♀️ DATASET
La base de datos que se usa es la de CelebA, la cual consiste de:
  * Una colección de 202,599 imagenes de rostros de celebridades
  * Un archivo de texto binario que tiene etiquetado 40 atributos de cada imagen.
Se puede encontrar en el sitio de kaggle: 
https://www.kaggle.com/datasets/jessicali9530/celeba-dataset


## 1. MODELO 1: IDENTIFICADOR DE ATRIBUTOS EN IMAGENES 📈
### Procesamiento de los datos de CelebA

Una vez obtenida la base de datos e importadas las paqueterias, tenemos que cargar los dos tipos de datos: una carpeta con puras imagenes de rostros y un archivo de texto binario con los atributos de cada imagen. 

Comenzamos con el segundo porque al ser un archivo de texto con datos estructurados es más fácil de manejar si lo tratamos como un csv, es decir, un archivo Comma Separated Value, el cual tiene formato de tabla, para convertir el archivo de texto a un archivo CSV usamos la paquetería de Pandas.


Para las imagenes definimos la ruta en la que se encuentra la carpeta e iniciamos el procesamiento de imagenes que en este caso consiste en asignarle el nombre de la imagen, sus atributos y la imagen misma en un solo dato el cual llamamos imagen etiquetada, transformar la imagen de 3 canales RGB a un tensor en escala de grises, redimensionar la imagen y por ultimo normalizar la intensidad de los tonos de grises, el cual va de 0 (negro) a 255 (blanco).


### Estructura del modelo identificador 
Una vez procesados los datos podemos comenzar con el modelo predictivo, el cual consiste de una secuencia de redes neuronales convolucionales 2D con su respectiva activación ReLu y un MaxPooling2D para procesar las matrices de imagenes y por ultimo se "aplana" la red para usar una red neuronal densa y una activación sigmoide que procese los datos para poder predecir si una imagen contiene o no cada atributo de los 40 considerados. 

![image](https://user-images.githubusercontent.com/80428982/202064225-03735a07-eb5f-47d8-b19f-ee916c80bb93.png)

Para este tipo de problemas es conveniente usar una funcion de costo binaria como el Binary_crossentropy y el optimizador RMSprop (Root Mean Squared Propagation) con una metrica binaria.

![image](https://user-images.githubusercontent.com/80428982/202064345-848713bc-dc5a-469e-acfc-6cb584b5e352.png)

### Entrenamiento de la red
Para finalizar con este primer modelo, se entrenó con 10 epocas, con lo cual se obtuvo una precision cerca del 90%.

![image](https://user-images.githubusercontent.com/80428982/202064163-75f08693-ef34-4e16-913e-5e60a4ae1a93.png)


## 2. MODELO 2: CLASIFICADOR DE RECONOCIMIENTO FACIAL ✅❌

### Procesamiento de datos
En este caso volvemos a cargar los datos, los cuales serán imagenes. Para el conjunto de entrenamiento y de validación tenemos que deben estar las imagenes deben estar etiquetadas para poder entrenar el modelo, por ello hay dos carpetas en cada conjunto, una con fotos de nuestro rostro y otra con fotos de otras personas, por otro lado, para el conjunto de prueba estarán mezcladas las fotos de nosotros con las de otras personas para evaluar el modelo.
Una vez cargadas las imagenes, tenemos que procesarlas, para esto podemos usar la clase ImageDataGenerator para normalizar y redimensionar las imagenes.

### Estructura del modelo clasificador
Para el modelo clasificador o de reconocimiento facial se usa una técnica llamada **Transfer Learning**, la cual consiste en usar un modelo pre entrenado para crear otro modelo, basta con entrenar el modelo original con los nuevos datos de interés. En nuestro caso, el modelo original ya fue entrenado para identificar atributos en los rostros, por lo que es capaz de distinguir rostros, ahora usaremos este desempeño para que pueda reconocer mi rostro, solo basta con usar las capas convolucionales del modelo entrenado y agregarle un clasificador, el cual tendrá como salida un sí o no. 



Cargamos el modelo 1 pre entrenado. 
![image](https://user-images.githubusercontent.com/80428982/202092822-f6c0b9fc-654b-4c79-b756-56bb34129e0a.png)


Definimos la estructura del modelo 2, el clasificador y congelamos los pesos de las capas convolucionales, dado que ya fueron entrenadas. 

![image](https://user-images.githubusercontent.com/80428982/202092894-4827456a-4c5d-497d-ac67-83911f804de5.png)

Definimos la funcion de costo, el optimizador y la metrica correspondiente. 

![image](https://user-images.githubusercontent.com/80428982/202093007-4538d2ae-1bb2-4e8f-bd17-0b79be631521.png)

En este caso después de usar distintos optimizadores, se optó por utilizar el optimizador Adam dado que con otros el resultado era ineficiente, igualmente me di cuenta que al disminuir el learning rate se obtenían mejores resultados. 


Por ultimo, se entrena el modelo.

![image](https://user-images.githubusercontent.com/80428982/202093350-eda09380-f94f-45ee-a326-bb7e8953e837.png)


Algunos problemas encontrados fueron al momento de procesar los datos del modelo 1 dado que la lista de atributos era un archivo de texto tipo estructurado pero separado por espacios en blanco, se tuvo que reemplazar los espacios en blanco por comas para poder tratarlo como un CSV con la paquetería de Pandas. 
En el proceso del Modelo 2 hubo complicaciones desde el principio dado que al usar ImageDataGenerator no leía mis fotos, después de intentar arreglarlo me di cuenta que era por el tipo de formato, la clase ImageDataGenerator no lee formatos tipo .JFIF, por lo cual hubo que transformar el formato para poder procesarlo.  Por otra parte, en la generación de imágenes del modelo 2, me salió un error de que el directorio de una imagen mía no existía y por lo tanto no se podían generar nuevas imagenes, este error aún no sé cómo solucionarlo, así que decidí ir a lo seguro y usar la funcion flow_from_directory() la cual transforma las imagenes mientras se entrena, de esta forma pude entrenar el modelo. 

Conclusiones: 
Antes de realizar este proyecto, desconocía lo versatil que puede ser trabajar con los modelos de redes neuronales, el poder crear un modelo nuevo a partir de otro ya entrenado es una herramienta muy poderosa que es escalable fuera del area de las imagenes. 
Si bien trabajar con redes neuronales e imagenes puede ser abrumador porque todo tiene que encajar correctamente, también es verdad que conociendo las bases y leyendo la documentación correspondiente de keras, es posible tener una idea general de lo que sucede. 
