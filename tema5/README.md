# Algoritmos de PNL
## Distancia de Levenstein
##### Algoritmo formal
Sea Σ un alfabeto finito. Dadas dos cadenas A = a1, a2, ..., an y B = b1, b2, ..., bn sobre Σ donde |A| = n y
|B| = m. La Distancia de Levenstein L(A, B) se define como el costo del camino de costo mínimo en un grafo de
ediciones.
1. Objetivo. La función D(i, j) representa la distancia mínima para transformar el prefijo de la primera palabra (de
longitud i) en el prefijo de la segunda palabra (de longitud j).
2. Los Casos Base (Condiciones de Frontera). Si j = 0: Para convertir una palabra de longitud i en una cadena vacía, necesitamos i eliminaciones.
Si i = 0: Para convertir una cadena vacía en una palabra de longitud j, necesitamos j inserciones.
Esto es lo que llena la primera fila y la primera columna de nuestra matriz.
3. El Corazón del Algoritmo (Inducción). Cuando estamos comparando caracteres en cualquier otra posición, la fórmula busca el mínimo de tres
caminos posibles, lo que garantiza que siempre tomamos la ruta más eficiente:
  - D(i-1, j) + 1 (Eliminación). Costo de la celda de arriba más 1.
  - D(i, j-1) + 1 (Inserción). Costo de la celda de la izquierda más 1.
  - D(i-1, j-1) + 1_(ai!=bj) (Sustitución). Costo de la celda de arriba más: 1 si se cumple ai != bj, 0 de otro modo.
4. ¿Por qué es una "Recurrencia"? Se llama así porque el valor de un estado actual (i, j) depende directamente de los valores de los estados anteriores (sus vecinos en la matriz). Es la esencia de la Programación Dinámica: dividimos un problema complejo (comparar dos palabras largas) en subproblemas más pequeños (comparar letras individuales) y reutilizamos los resultados ya calculados.
En resumen: Es la regla lógica que dicta cómo se debe llenar cada celda de la matriz de distancias para encontrar el camino más corto entre dos cadenas de texto.

##### Especificación del algoritmo (Wagner-Fischer)
Este algoritmo computa la distancia en una estructura de datos matricial M de dimensiones (n+1) x (m+1).

Pasos formales:
```
1. Inicialización de casos base:
  Para toda i en {0, .., n}: M[i,0] <- i (transformación de una cadena de long. i a una cadena vacía)
  Para toda j en {0, .., m}: M[0,j] <- j (transformación de una cadena vacía a una de long. j)
2. Iteración de estados:
  Para cada fila i desde 1 hasta n:
    Para cada columna j desde 1 hasta m:
      Si A[i] = B[j], costo de sustitución: W <- 0
      Si A[i] != B[j], costo de sustitución: W <- 1
      M[i,j] <- min(M[i-1,j]+1, M[i,j-1]+1, M[i-1,j-1]+W)
3. Finalización:
  El valor de L(A,B) es M[n,m]  
```

##### Métrica (Vladimir Levenshtein, 1965)
Vladimir Levenshtein fue un matemático soviético que definió la distancia teórica entre dos secuencias.
Él estableció la regla de qué operaciones se permiten (inserción, eliminación, sustitución) y la lógica de la
función de recurrencia que acabas de ver. Sin embargo, su publicación original no se centraba en cómo
una computadora debía calcularla de forma eficiente, sino en la teoría de códigos de corrección de
errores.
##### El Algoritmo (Wagner y Fischer, 1974)
Robert Wagner y Michael Fischer fueron quienes tomaron esa definición matemática y desarrollaron el
algoritmo de Programación Dinámica que utiliza la matriz O(n x m) para resolverlo en tiempo
polinomial. Antes de ellos, calcular la distancia de Levenshtein de forma ingenua (por fuerza bruta o
recursión simple) era extremadamente costoso (exponencial).
##### Diferencia clave :
•Levenshtein: Define qué estamos midiendo (la distancia de edición).
•Wagner-Fischer: Define cómo lo calculamos eficientemente (usando la matriz y programación
dinámica).

## Word2Vec
##### Algoritmo formal (Skip-gram con Softmax)
No es una red neuronal profunda sino una red de una sola capa oculta (projection layer) que busca maximizar la probabilidad del contexto.
1. Función de objetivo (log-verosimilitud). Dada una secuencia de palabras de entrenamiento w1, w2, ..., wT, el objetivo es maximizar el promedio de la log-probabilidad.
2. Función Softmax. La probabilidad p(w0 dado w1) se define usando la exponencial del producto punto de los vectores.

##### Métricas de complejidad
1. Complejidad temporal (entrenamiento): O(E x T x c x log V). E es el número de épocas, T el número de palabras en el corpus y V el tamaño del vocabulario. (Nota: el uso de Hierarchical Softmax o Negative Sampling reduce el costo de V a log V)
2. Complejidad espacial: O(V x D). Donde D es la dimensionalidad del embedding (100-300). Hay que almacenar dos matrices de pesos: una para los vectores de entrada y otra para los de salida.

##### Pseudocódigo
```
ALGORITMO Word2Vec_SkipGram(corpus, dimension, ventana,
tasa_aprendizaje)
  Vocabulario <- Obtener_Palabras_Unicas(corpus)
  V <- longitud(Vocabulario)
  // Inicializar matrices de pesos con valores aleatorios pequeños
  W_entrada <- Matriz_Aleatoria(V, dimension)
  W_salida <- Matriz_Aleatoria(dimension, V)

  PARA epoca DESDE 1 HASTA N_epocas:
    PARA CADA palabra_central EN corpus:
      contexto <- Obtener_Ventana(palabra_central, ventana)
      PARA CADA palabra_contexto EN contexto:
        // Forward pass
        h <- W_entrada[indice_central]
        u <- h * W_salida
        y_pred <- Softmax(u)

        // Calcular error (e = y_pred - y_real)
        error <- Calcular_Error(y_pred, palabra_contexto)

        // Backpropagation (Gradiente Descendente)
        Actualizar W_salida usando error y h
        Actualizar W_entrada usando error y W_salida
  RETORNAR W_entrada
FIN ALGORITMO
```

# Algoritmos de Visión computacional
## Sobel
##### Concepto y Características Matemáticas
El operador de Sobel es un filtro de detección de bordes que utiliza una aproximación discreta del
gradiente de la intensidad de una imagen. Se basa en la convolución de la imagen con dos kernels
(máscaras) de 3x3 que calculan las derivadas parciales en las direcciones horizontal (Gx) y vertical (Gy).

Kernels: Gx y Gy matrices de 3x3

Magnitud del Gradiente (G): G = sqrt(Gx^2 + Gy^2)
Dirección del Gradiente (θ): θ = arctan(Gy/Gx)

##### Algoritmo en lenguaje natural
1. Carga: Leer la imagen en escala de grises.
2. Preparación: Añadir un marco (padding) a la imagen para manejar los bordes durante la
convolución.
3. Convolución: Deslizar las máscaras Gx y Gy sobre cada píxel.
4. Cálculo: Calcular el gradiente resultante combinando ambos resultados (magnitud).
5. Normalización: Asegurar que los valores de intensidad estén en el rango [0, 255].
6. Salida: Generar la imagen resultante con los bordes resaltados

##### Algoritmo formal
Entrada: imagen I de dimensiones MxN
Salida: imagen E (bordes).
1. Para cada pixel (i,j) en I:
2. Calcular Gx(i,j)
3. Calcular Gy(i,j)
4. E(i,j) = sqrt(Gx(i,j)^2 + Gy(i,j)^2)
5. Si E(i,j) > umbral, marcar como borde

##### Métricas de complejidad
1. Complejidad Temporal: O(M x N), donde M y N son las
dimensiones de la imagen. Por cada píxel, realizamos un número
constante de operaciones (9 multiplicaciones y sumas por kernel).
2. Complejidad Espacial: O(M x N) para almacenar las derivadas
parciales y la imagen resultante.

##### Pseudocódigo
```
Función Sobel(imagen):
filas, cols = dimensiones(imagen)
resultado = matriz(filas, cols)

Para i de 1 a filas-2:
  Para j de 1 a cols-2:
    gx = convolucion(imagen, i, j, kernel_x)
    gy = convolucion(imagen, i, j, kernel_y)
    resultado[i, j] = raiz_cuadrada(gx^2 + gy^2)
Retornar normalizar(resultado)
```

## SIFT (Scale-Invariant Feature Transform)
##### Concepto y Características Matemáticas
SIFT opera sobre la premisa de extraer puntos clave (keypoints) que se mantienen estables bajo diferentes
transformaciones.
1. Espacio de Escala: Se utiliza el Diferencial de Gaussianas (DoG) para detectar puntos de interés en diferentes
escalas, aproximando el Laplaciano de Gaussiano (LoG).

DoG(x, y, σ) = L(x, y, kσ) − L(x, y, σ)

2. Orientación: Se asigna una orientación dominante a cada punto clave basándose en los gradientes locales,
logrando invariancia a la rotación.
3. Descriptor: Se crea un vector de 128 dimensiones que describe la región alrededor del punto clave, garantizando
invariancia a la escala y robustez a cambios de iluminación.

##### Algoritmo en lenguaje natural
1. Detección de Extremos en Escala: Construir una pirámide de imágenes desenfocadas (Gaussianas) y
restar niveles adyacentes para hallar el DoG. Identificar máximos/mínimos locales.
2. Localización de Puntos Clave: Refinar la posición de los puntos y eliminar aquellos con bajo contraste o
que estén sobre bordes.
3. Asignación de Orientación: Calcular gradientes en una vecindad alrededor del punto; crear un histograma
de orientaciones y asignar la dirección dominante.
4. Generación de Descriptores: Dividir la vecindad en una cuadrícula de 4X4 subregiones, calculando
histogramas de gradiente para cada una, resultando en un vector de 4x4x8=128 elementos.

##### Algoritmo formal
Entrada: imagen I
Salida: Conjunto de descriptores {D1, D2, ..., Dn}.
1. DoG <- Diferencia de Gaussianas en octavas
2. K <- Extremos locales en espacio-escala (DoG)
3. Para cada punto p en K:
  - Calcular gradiente m(x,y) y ángulo theta(x,y)
  - Asignar orientación dominante Op
  - Generar descriptor Dp normalizado

##### Métricas de complejidad
1. Complejidad Temporal: O(N) donde N es el número
de píxeles, pero con una constante muy alta debido a
la construcción de la pirámide Gaussiana y el cálculo de
los 128 descriptores. Es computacionalmente costoso
en comparación con Sobel.
2. Complejidad Espacial: O(N x S), donde S es el número
de escalas, debido al almacenamiento de la pirámide
de imágenes

##### Pseudocódigo
```
Función SIFT(imagen):
  piramide = ConstruirPiramideGaussiana(imagen)
  dog = CalcularDiferenciaGaussianas(piramide)
  puntos_clave = DetectarExtremos(dog)
  descriptores = []
  Para p en puntos_clave:
    orientacion = CalcularOrientacionDominante(p)
    desc = CrearDescriptor128D(p, orientacion)
    descriptores.agregar(desc)

Retornar descriptores
```
