# Regresión Lineal (Simple y Múltiple)
## Algoritmo en palabras
* Inicialización: Se asignan valores aleatorios a los parámetros (pesos ω y sesgo b).
* Predicción: Se calcula el producto punto entre las características y los pesos, sumando el sesgo: y^ = Xω + b.
* Cálculo del Error: Se mide la diferencia entre la predicción y el valor real usando el Error Cuadrático Medio (MSE).
* Optimización: Se calculan las derivadas parciales (gradientes) de la función de costo respecto a cada parámetro.
* Actualización: Se ajustan los pesos en dirección opuesta al gradiente para minimizar el error.

## Métricas de complejidad
Considerando n ejemplos y d características:
* Complejidad en tiempo:
    * Entrenamiento (Gradiente Descendiente): O(knd), donde k es el número de iteraciones.
    * Entrenamiento (ecuación normal): O(d3), debido a la inversión de la matriz (X^T)X.
    * Predicción: O(d).
* Complejidad en espacio: O(d) para almacenar los pesos del modelo.

## Formalismo matemático
Hipótesis: h_(theta) (x) = (theta^T)x

Función de costo es MSE

Regla de actualización: theta_j := theta_j - alpha(derivada parcial de J(theta) con respecto de theta_j)
## Pseudocódigo
```
Algoritmo: Regresión Lineal por Gradiente Descendiente
Entrada: Matriz X, vector y, tasa de aprendizaje alpha, iteraciones T
Salida: Pesos theta

Inicializar theta en ceros o valores aleatorios

Para t desde 1 hasta T:
Calcular predicciones: error = (X * theta) - y
Calcular gradiente: grad = (1/m) * (X_transpuesta * error)
Actualizar pesos: theta = theta - alpha * grad
Retornar theta
```

# Regresión Logística
## Algoritmo en palabras
Es similar a la lineal, pero aplica la función Sigmoide al resultado de la combinación lineal. Esto "aplasta" cualquier valor real a un rango entre 0 y 1. En lugar de MSE, utiliza Entropía Cruzada Binaria (Log Loss) porque el MSE en logística genera una superficie de costo no convexa, lo que dificultaría encontrar el mínimo global.

## Métricas de complejidad
* Complejidad en tiempo:
    * O(knd), es ligeramente más costoso que la lineal por el cálculo del exponente en la sigmoide, pero mantiene la misma clase de complejidad.
* Complejidad en espacio: O(d).

A diferencia de la lineal, la logística no tiene una "Ecuación Normal" (solución cerrada) debido a la naturaleza no lineal de la sigmoide. Por ello, siempre dependemos de métodos iterativos como el Gradiente descendiente o Newton-Raphson.

## Formalismo matemático
Sigmoide: g(z) = 1 / (1+e^(-z))

Hipótesis: h_theta (x) = g((theta^T)x)

Función de costot: MSE pero con logaritmos raros

## Pseudocódigo
```
Algoritmo: Regresión Logística Binaria
Entrada:
    X: Matriz de características (m ejemplos, n rasgos)
    y: Vector de etiquetas reales (0 o 1)
    alpha: Tasa de aprendizaje
    T: Número de iteraciones
Salida:
Vector de parámetros theta (pesos y sesgo)

1. Inicializar theta (pueden ser ceros o valores aleatorios pequeños)
2. Para cada iteración t = 1 hasta T:
    a. Calcular el producto lineal:
        z = X * theta
    b. Aplicar la función de activación (Sigmoide):
        h = 1 / (1 + exp(-z))
    c. Calcular el error (diferencia entre probabilidad y etiqueta real):
        error = h - y
    d. Calcular el gradiente (derivada de la Log-Loss):
        gradiente = (1 / m) * (X_transpuesta * error)
    e. Actualizar los parámetros:
        theta = theta - (alpha * gradiente)
3. Retornar theta
```
# Support Vector Machines (SVM)
## Algoritmo en palabras
Imagina que tienes dos grupos de puntos (clases) en un plano. Una SVM intenta trazar una línea (o un hiperplano en más dimensiones) que no solo separe los grupos, sino que pase lo más lejos posible de los puntos más cercanos de ambos bandos.
* Vectores de Soporte: Son los puntos de datos que están
justo en el borde de cada clase. Son "críticos" porque si los
mueves, la línea divisoria cambia.
* Margen: Es la distancia entre la línea divisoria y los vectores
de soporte. El objetivo de la SVM es maximizar este margen.
* Truco del Kernel (Kernel Trick): Cuando los datos no se
pueden separar con una línea recta, la SVM los proyecta
matemáticamente a una dimensión superior donde sí sean
separables. Es como elevar puntos de una mesa al aire para
poder pasar una hoja de papel entre ellos.

## Métricas de complejidad
* Complejidad en tiempo:
    * Entrenamiento: Típicamente entre O(n2∙d) y O(n3∙d). Esto hace que las SVM sean lentas con conjuntos de datos muy grandes (millones de muestras).
    * Predicción: O(vd), donde v es el número de vectores de soporte. Es muy eficiente una vez entrenado.
* Complejidad en espacio: O(n2) en la implementación básica, ya que necesita almacenar la matriz de similitudes (kernel).

## Formalismo matemático
Buscamos un hiperplano definido por (ω^T)x + b = 0.

Función de decisión:
f(x) = sign((ω^T) x + b)

Problema de optimización:
Queremos minimizar 1/2 ||ω||^2 (que equivale a maximizar el margen 2||ω||) sujeto a la restricción de que cada punto quede del lado correcto:
y^i((ω^T) x^i + b) ≥ 1

#### Características matemáticas adicionales:

1. Convexidad: El problema de optimización es
cuadrático y convexo, lo que garantiza que siempre
encontraremos el mínimo global, no hay mínimos
locales como en las Redes Neuronales.
2. Kernel Trick: Permite calcular el producto punto en
dimensiones altas sin transformar explícitamente los
datos.
3. Margen Blando (C): Si los datos tienen ruido,
introducimos una variable de holgura (ξ) y un
parámetro de penalización C para permitir algunos
errores a cambio de un margen más ancho.


## Pseudocódigo
```
Aunque las SVM suelen resolverse con programación cuadrática,
para entender su implementación en código usamos la pérdida
de "bisagra" (Hinge Loss).

Algoritmo: SVM (Hard Margin / Linear)
Entrada: X, y (etiquetas -1 o 1), alpha, lambda (regularización)
Salida: w, b

1. Inicializar w en ceros w=0, b = 0
2. Para cada iteración/época:
    Para cada ejemplo xi:
        Si yi * (w * xi - b) >= 1:
        # El punto está fuera del margen,solo aplicamos regularización
            dw = 2 * lambda * w
            db = 0
        Si no:
            # El punto es un vector de soporte o está mal clasificado
            dw = 2 * lambda * w - (xi * yi)
            db = yi
        Actualizar pesos y sesgo: w = w - alpha * dw, b = b - alpha * db
```

# k-Nearest Neighbors (k-NN)
## Algoritmo en palabras
Imagina que te mudas a un nuevo vecindario y quieres saber
si el costo de vida será alto o bajo. Lo más natural es
preguntar a tus k vecinos más cercanos. Si la mayoría dice que
es caro, probablemente para ti también lo sea.
* Naturaleza "Lazy" (Perezosa): A diferencia de SVM, k-NN
no genera un modelo o una línea divisoria durante el
entrenamiento. Simplemente guarda los datos y "se pone
a trabajar" solo cuando le pides una predicción.
* Voto de pluralidad: Para clasificar un punto nuevo, el
algoritmo mide la distancia entre ese punto y todos los
demás en el conjunto de entrenamiento, selecciona los k
más cercanos y asigna la clase más frecuente.
* El parámetro k: Es el número de vecinos. Un k pequeño (ej. k=1) es muy sensible al ruido (overfitting), mientras que un k muy grande suaviza demasiado las fronteras (underfitting).

## Métricas de complejidad
Aquí es donde k-NN contrasta fuertemente con SVM.Mientras que SVM es lento para entrenar pero rápido para predecir, k-NN es lo opuesto.
* Complejidad en tiempo:
    * Entrenamiento: O(1). No hay fase de optimización; solo se almacenan los datos en memoria.
    * Predicción: O(nd). Para cada consulta, debemos calcular la distancia contra los n ejemplos de entrenamiento, cada uno con d dimensiones.
        * Nota: Esto hace que k-NN sea inviable para sistemas en tiempo real con millones de datos, a menos que usemos estructuras como KD-Trees o Ball Trees que reducen esto a O(d ∙ log(n))
* Complejidad en espacio: O(nd). Necesitamos mantener todo el dataset en memoria RAM para poder comparar los puntos en cada predicción.

## Formalismo matemático
k-NN es un algoritmo de aprendizaje supervisado no paramétrico, lo que significa que no asume una distribución predefinida para los datos.
1. Función de Distancia
La métrica más común es la Distancia Euclidiana. Dado un conjunto D de entrenamiento y un nuevo punto, se obtienen los k puntos de D más cercanos al nuevo punto y se cuenta la frecuencia de las clases de cada uno de esos k puntos, tomándose para el punto nuevo la clase más frecuente.

## Pseudocódigo
```
Este pseudocódigo sigue la lógica de "fuerza bruta" para que
los chicos comprendan el flujo fundamental antes de optimizar.
Algoritmo: k-Nearest Neighbors (k-NN)
Entrada: Dataset de entrenamiento(X, y), nuevo punto a clasificar xtest,
parámetro k.
Salida: Clase predicha para xtest.

1. Para cada ejemplo (xi, yi) en el dataset de entrenamiento:
    Calcular la distancia di entre xtest y xi (ej. Euclideana).
    Almacenar la distancia di junto con su etiqueta yi en una lista.
2. Ordenar la lista de distancias de menor a mayor.
3. Seleccionar los primeros k elementos de la lista ordenada (los k vecinos
más cercanos).
4. Contar la frecuencia de cada etiqueta entre esos k vecinos.
5. Retornar la etiqueta con la frecuencia más alta (voto mayoritario).
```

# Árbol de Decisión (Decision Trees)
## Algoritmo en palabras
Imagina que estás jugando a las "20 preguntas". Para adivinar
un objeto, haces preguntas estratégicas: "¿Es un animal?",
"¿Tiene cuatro patas?", "¿Ladra?".
* Divisiones Binarias: El algoritmo busca la característica que
mejor separa los datos en grupos puros (donde todos los
puntos pertenecen a la misma clase).
* Nodos y Hojas: Las preguntas son los nodos, y las respuestas
finales (la clasificación) son las hojas.
* Ganancia de Información: El árbol no elige preguntas al azar;
usa medidas matemáticas (como la Entropía o el Índice Gini)
para elegir la pregunta que reduzca más el "desorden" o
incertidumbre en cada paso.

## Métricas de complejidad
* Complejidad en tiempo:
    * Entrenamiento: O(nd∙log(n)). Es más costoso que KNN porque debe evaluar cada característica en cada nivel para encontrar el mejor corte.
    * Predicción: O(profundidad del árbol). Generalmente es O(log(n)), lo que lo hace extremadamente rápido en producción.
* Complejidad en espacio: O(nodos). Solo guardamos las reglas de decisión (si x > 5 entonces ...), no los datos originales.

## Formalismo matemático
A diferencia de KNN, este es un modelo paramétrico
(una vez entrenado, puedes borrar los datos y
quedarte solo con la estructura del árbol).
1. Entropía (H). Mide el grado de desorden en un conjunto de datos S: es -1 multiplicado por la suma de los logaritmos base dos multiplicados por las probabilidades frecuentistas.
2. Ganancia de información (IG). Es la métrica para decidir que característica usar para dividir. El algoritmo elige la característica A que maximiza la ganancia.

## Pseudocódigo (ID3 / CART)
```
Algoritmo: Construcción de Árbol de Decisión
Entrada: Dataset D (características y etiquetas).
1. Si todos los ejemplos en D son de la misma clase:
    Retornar un nodo hoja con esa etiqueta.
2. Si no hay más características para dividir:
    Retornar un nodo hoja con la clase mayoritaria.
3. Si no:
    Calcular la Ganancia de Información para cada característica disponible.
    Elegir la característica A con la mayor ganancia.
    Crear un Nodo de Decisión basado en A. 
    Para cada valor posible de A, crear una rama y repetir el
    proceso recursivamente con el subconjunto de datos resultante.
```

# Random Forest
## Algoritmo en palabras
Imagina que quieres comprar un coche nuevo. En lugar de
preguntarle a un solo experto (Árbol de Decisión), le
preguntas a 100 personas diferentes.
* Bagging (Bootstrap Aggregating): Cada persona recibe un
subconjunto distinto de datos para estudiar.
* Selección Aleatoria de Características: A cada persona se le
prohíbe ver algunas características (por ejemplo, a unos no
les dejas ver el precio, a otros no les dejas ver el motor). Esto
obliga a que cada árbol aprenda cosas distintas.
* Sabiduría de las Masas: Al final, todos votan. La clase que
reciba más votos es la ganadora.

## Métricas de complejidad
* Complejidad en tiempo:
* Entrenamiento: O(Bnd∙log(n)) donde B es el número de árboles. Como los árboles
son independientes, ¡esto se puede
paralelizar fácilmente!
* Predicción: O(B ∙ profundidad del árbol).
Siegue siendo muy rápido, aunque B veces
más lento que un solo árbol.
* Complejidad en espacio: O(B ∙ nodos) Necesitamos guardar todos los árboles en memoria.

## Formalismo matemático
Random Forest utiliza el principio de reducción de varianza.
1. Bootstrapping. Dado un dataset D de tamaño N, creamos B muestras
aleatorias Db mediante muestreo con reemplazo. Esto
significa que algunos datos pueden repetirse y otros
pueden no aparecer.
2. El Clasificador de Bosque
El modelo final es una colección de árboles. La predicción final para un dato x se define por voto mayoritario.

## Pseudocódigo
```
Algoritmo: Random Forest
Entrada: Dataset D, número de árboles B, subconjunto de
características m.
1. Para b = 1 hasta B:
    Crear una muestra Bootstrap Db de los datos originales.
    Entrenar un árbol Tb usando Db .
    En cada división del árbol, seleccionar solo m características
    al azar de las d disponibles.
    Elegir el mejor corte entre esas m características.
2. Para predecir un dato nuevo:
    Pasar el dato por los B árboles.
    Realizar la votación y devolver la clase más frecuente.
```

# Gradient Boosting
## Algoritmo en palabras
Si Random Forest es una democracia donde todos votan al
mismo tiempo, el Gradient Boosting es un proceso de
aprendizaje secuencial y meritocrático.
* Aprendizaje del Error: El primer modelo hace una predicción
burda. El segundo modelo no intenta predecir el resultado
real, sino que intenta predecir el error (residuo) que cometió
el primero.
* Corrección en Cascada: Cada nuevo árbol que añadimos se
enfoca exclusivamente en corregir lo que los anteriores
todavía no entienden bien.
* Optimización del Gradiente: El nombre viene de que usamos
el descenso de gradiente para minimizar una función de
pérdida, moviendo nuestras predicciones en la dirección que
reduce el error más rápidamente.

## Métricas de complejidad
* Complejidad en tiempo:
    * Entrenamiento: O(Mnd * log(n)) donde
M es el número de iteraciones (árboles). A
diferencia de Random Forest, no es
paralelizable fácilmente porque cada árbol
depende del anterior.
    * Predicción: O(M ∙ profundidad del árbol).
Es muy eficiente, aunque depende del
número de árboles acumulados.
* Complejidad en espacio: O(M ∙ nodos) Necesitamos almacenar la secuencia
completa de árboles para poder reconstruir la
predicción final.

## Formalismo matemático
1. Modelo Aditivo: La predicción final es la suma
ponderada de modelos simples.
2. Función de Pérdida (L): Buscamos minimizar una función como el Error Cuadrático Medio (MSE).
3. Pseudo-Residuos: En cada paso m, calculamos el
gradiente negativo de la pérdida respecto a la
predicción actual.

## Pseudocódigo
```
Pseudocódigo de Gradient Boosting
Algoritmo: Gradient Boosting (Regresión/Clasificación)
Entrada: Dataset D, Función de pérdida L, Número de iteraciones M,
Tasa de aprendizaje η.
1. Inicializar el modelo con una constante:
    Fo(x) = argmin(suma de L(yi, gamma))
2. For each m=1 hasta M:
    Calcular los residuos rim para cada ejemplo del dataset.
    Entrenar un árbol de decisión hm(x) usando los residuos
    rim como el nuevo "objetivo" (target).
    Actualizar el modelo sumando una fracción del nuevo árbol:
    Fm = Fm−1 x + η ∙ hm(x)
    Salida: El modelo final FM(x).
```

# XGBoost (eXtreme Gradient Boosting)
## Algoritmo en palabras
Si el Gradient Boosting es un aprendiz que corrige errores,
XGBoost es ese mismo aprendiz con un sistema de alto
rendimiento y un estricto sentido de la autodisciplina.
* Regularización extrema: A diferencia del Boosting
tradicional, XGBoost castiga a los árboles que se vuelven
demasiado complejos, lo que ayuda muchísimo a evitar el
sobreajuste (overfitting).
* Paralelización de estructura: Aunque el boosting es
secuencial por naturaleza, XGBoost es extremadamente
rápido porque paraleliza la búsqueda de los puntos de
corte (splits) dentro de cada nivel del árbol.
* Manejo de datos dispersos: Tiene una "dirección por
defecto" incorporada; si encuentra un valor nulo, ya sabe
hacia qué lado de la rama enviarlo basándose en la
reducción de la pérdida.

## Métricas de complejidad
* Complejidad en tiempo:
    * Entrenamiento: O M ∙ d ∙ nlog(n) donde M
es el número de árboles. Gracias al uso de
block structures y cómputo fuera de memoria
(out-of-core), es drásticamente más veloz
que el GBM estándar.
    * Predicción: O M ∙ profundidad del árbol .
Sigue siendo la suma de las predicciones de
los M árboles.

* Complejidad en espacio: O M ∙ nodos + O(n ∙ d)
Necesita espacio para la estructura de los árboles y
para los índices de las características ordenadas
(histogramas).

# K means
## Algoritmo en palabras
Imagina que tienes una nube de puntos en un plano y quieres
dividirlos en K grupos (clusters).
* Centroides: Cada grupo tiene un "representante" llamado
centroide, que es el centro geométrico de sus puntos.
* Asignación: Cada punto decide unirse al grupo cuyo
centroide esté más cerca (usualmente por distancia
euclidiana).
* Actualización: Una vez que todos los puntos eligieron
bando, los centroides se mueven al promedio real de sus
nuevos miembros.
* Iteración: Repetimos esto hasta que los centroides dejen
de moverse. Es como un baile donde los líderes buscan el
centro de su gente y la gente sigue a su líder más cercano.

## Métricas de complejidad
* Complejidad en tiempo:
    * Entrenamiento: O I ∙ K ∙ n ∙ d , donde:
        * I es el número de iteraciones hasta la convergencia.
        * K es el número de clusters.
        * n es el número de muestras.
        * d es la cantidad de característica (dimensiones).
* Complejidad en espacio: O (n + K) ∙ d Ya que
necesitamos guardar los datos originales y las
coordenadas de los K centroides.

## Formalismo matemático
K-means busca minimizar la Inercia (o suma de
cuadrados dentro del cluster - WCSS).
Distancia Euclidiana: Es la métrica estándar para
medir qué tan "cerca" está un punto x de un centroide
μ.
Convergencia: El algoritmo garantiza converger a
un mínimo, aunque puede ser un mínimo local.
Por eso es común correrlo varias veces con
diferentes inicios aleatorios.
Sensibilidad: Es muy sensible a los valores
atípicos (outliers) y a la escala de los datos (¡hay
que normalizar!).

## Pseudocódigo
```
Pseudocódigo
Algoritmo: K-means Clustering
Entrada: Dataset X y número de clusters K.
1. Inicializar K centroides aleatoriamente (pueden ser puntos del
mismo dataset).
2. Repetir hasta que no haya cambios (o se llegue a un máximo de
iteraciones):
    Paso de asignación: Para cada punto xi, encontrar el
    centroide μj más cercano y asignarlo al clúster Cj
    Paso de actualización: Para cada clúster Cj, calcular el nuevo centroide μj como el promedio de todos los puntos asignados a él:
2. Salida: Los K centroides y las etiquetas de pertenencia de cada
punto.
```

# DBSCAN
## Algoritmo en palabras
DBSCAN ve el mundo como densidades. Imagina que estás en una fiesta:
• Puntos Núcleo (Core Points): Son personas que están en el centro de un grupo denso (tienen al menos MinPts amigos
cerca).
• Puntos Frontera (Border Points): Son personas que tienen pocos amigos cerca, pero al menos uno de ellos es un "Punto
Núcleo".
• Ruido (Noise/Outliers): Son personas que están solas y no tienen a ningún "Punto Núcleo" cerca.
El algoritmo expande los grupos conectando puntos núcleo que están a una distancia máxima ε (epsilon) entre sí.

## Métricas de complejidad
Métricas de complejidad
* Complejidad en tiempo:
• Promedio: O n logn si se usan estructuras
de datos espaciales (como KD-Trees) para
buscar vecinos.
• Peor caso: O n
2
si se calcula la matriz de

distancias completa.

• Complejidad en espacio: O n para almacenar las
etiquetas y los datos.

## Formalismo matemático
1. Parámetro ∈ (Epsilon): Define el radio de la
vecindad. Si es muy pequeño, todo será ruido; si
es muy grande, todos los puntos se unirán en un
solo grupo.

2. Parámetro MinPts : Define la "masa crítica" para
considerar una región como densa. Una regla de
oro común es MinPts ≥ d + 1 (donde d es la
dimensión).

3. No Paramétrico en K: A diferencia de K-means,
DBSCAN descubre el número de clústers
automáticamente.

## Pseudocódigo
```
Algoritmo: DBSCAN
Entrada: Dataset X, ∈, MinPts.

1. Etiquetar todos los puntos como NO VISITADOS.
2. Para cada punto P en X:
    Si P ya fue visitado, continuar al siguiente.
    Marcar P como VISITADO.
    Buscar vecinos de P a distancia ∈ .
    Si cantidad de vecinos < MinPts:
        Marcar P como RUIDO.
    Si no:
        Crear un nuevo Cluster C y añadir P.
        Expandir Cluster: Para cada vecino P' de P, si no ha sido visitado,
        marcar como VISITADO y si tiene suficientes vecinos, añadir
        sus propios vecinos a la lista de búsqueda. Añadir P' al Cluster C.
```             

# Hierarchical Clustering (Agrupamiento Jerárquico)
## Algoritmo en palabras
Existen dos tipos, pero el más común es el Aglomerativo (de abajo hacia arriba):
* Individualismo inicial: Cada punto empieza siendo su propio cluster.
* Fusión por cercanía: En cada paso, buscamos los dos clusters más cercanos y los combinamos en uno solo.
* Estructura de Árbol: Repetimos el proceso hasta que todos los puntos forman un único gran cluster.
* Decisión: Al final, nosotros decidimos a qué "altura" cortar el árbol para obtener el número de grupos que deseemos.

Más formalmente:
Dado un conjunto de n puntos:
1. Calcular la Matriz de distancia inicial entre todos los
puntos.
2. Encontrar el par de clusters (Ci, Cj) con la distancia mínima según un criterio de enlace (linkage).
3. Fusionar Ci y Cj para formar un nuevo cluster Cnuevo.
4. Actualizar la Matriz de Distancia eliminando las
filas/columnas de i, j y añadiendo la del nuevo cluster.
5. Repetir hasta que quede un solo cluster.

## Métricas de complejidad
Métricas de complejidad
• Complejidad en tiempo:
• Promedio: O(n3) en su implementación básica, aunque existen versiones optimizadas de O(n2 * logn).

• Complejidad en espacio:O(n2), ya que
necesitamos almacenar la matriz de distancias
completa entre todos los pares de puntos.

## Formalismo matemático
Lo más importante aquí es cómo medimos la distancia
entre grupos (Criterios de Enlace):
1. Single Linkage (Enlace Simple): Distancia mínima
entre un punto de A y un punto de B. Tiende a
crear clusters alargados.
2. Complete Linkage (Enlace Completo): Distancia
máxima entre un punto de A y un punto de B.
Crea clusters compactos.
3. Average Linkage (Enlace Promedio): Promedio de
todas las distancias entre pares de puntos.
4. Ward's Method: Minimiza el aumento de la
varianza total dentro de los clusters al fusionarlos.

## Pseudocódigo
```
Algoritmo: Hierarchical Agglomerative Clustering (HAC)
Entrada: Dataset X, métrica de distancia, criterio de enlace.

1. Inicializar cada xi como un cluster Ci
2. Calcular la matriz de distancias D entre todos los clusters.
3. Mientras número de clusters > 1:
    Buscar (Ci, Cj) tales que dist(Ci, Cj) sea mínima.
    Combinar Ci y Cj en Cunion .
    Actualizar D usando el criterio de enlace (ej. Ward o Average).
    Registrar la fusión y la distancia para el dendrograma.
4. Salida: Historia de fusiones (Z-matrix).
```          

# Isolation Forest (Bosque de Aislamiento)
## Algoritmo en palabras
Imagina que tienes una hoja de papel llena de puntos. Para "aislar" un punto:
1. Eliges una línea al azar (característica) y un punto de corte al azar.
2. Si un punto está muy lejos de los demás, bastarán pocos cortes para dejarlo solo en un espacio.
3. Si un punto está en el centro de una multitud, necesitarás muchísimos cortes para separarlo de sus vecinos.
Principio fundamental: Las anomalías son "pocas y diferentes", por lo que son más fáciles de aislar y quedan en las ramas más
cortas de los árboles.

## Métricas de complejidad
• Complejidad en tiempo:
• Entrenamiento: O(t ∙ ψ ∙ logψ), donde t es el número de árboles
y ψ es el tamaño de la
submuestra. Es extremadamente
eficiente porque no necesita
procesar todo el dataset por
árbol.
• Predicción: O t ∙ logψ , lo que
lo hace ideal para sistemas de
fraude en tiempo real.

• Complejidad en espacio:O t ∙ ψ , muy
baja ocupación de memoria
comparado con métodos basados en
distancias como KNN.

• Complejidad en espacio:O(n2), ya que
necesitamos almacenar la matriz de distancias
completa entre todos los pares de puntos.

## Formalismo matemático
orita

## Pseudocódigo
```
Algoritmo: Isolation Forest (iForest)
Entrada: Dataset X, número de árboles t, tamaño de submuestra ψ.
1. Para i = 1 hasta t:
    Tomar una submuestra X' de tamaño ψ.
    Construir un iTree:
        Seleccionar una característica q al azar.
        Seleccionar un valor de corte p al azar entre el min y max de q.
        Dividir los datos en Xizq y Xder .
        Repetir recursivamente hasta que el nodo tenga un solo punto o se alcance el límite de altura.
2. Para cada punto nuevo x:
    Calcular la longitud del camino h(x) en cada uno de los t árboles.
    Calcular el Anomaly Score s(x, ψ).
``` 
