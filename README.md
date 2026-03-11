# DFS (Depth First Search)

La búsqueda en profundidad es un algoritmo de exploración sistemática no informada para grafos, caracterizado por la exploración exhaustiva de una rama antes de proceder al _backtracking_.
## Principios:
1. Prioridad de Profundidad: El algoritmo expande el nodo descubierto más recientemente que aún posea aristas sin explorar incidentes a él.
2. Mecanismo de LIFO: DFS utiliza una pila (explícita o mediante la pila de llamadas en recursividad). Esto dicta que el último nodo en ser descubierto es el primero en ser procesado.
3. _Backtracking_: Cuando un nodo v ha sido completamente explorado (todas sus aristas incidentes llevan a nodos ya visitados), la búsqueda retrocede para explorar aristas que parten del nodo desde el cual se descubrió v.

## Propiedades:
1. Completitud: Es completo en grafos finitos. En grafos infinitos o con ciclos, puede no terminar si no se implementa un control de estados visitados.
2. Optimalidad: No es óptimo, no garantiza encontrar el camino más corto.
3. Complejidad Temporal: O(V + E), con V el número de vértices y E el de aristas. Cada vértice y arista se visitan una cantidad constante de veces.
4. Complejidad Espacial: O(V) en el peor de los casos (grafo lineal), determinado por la profundidad máxima de la pila de recursión.

## Algoritmo:
1. Comenzar en el nodo raíz.
2. Moverse a un vecino, luego al vecino de ese vecino, y así sucesivamente.
3. Si se llega a un nodo sin vecinos no visitados, regresar al último nodo que tenía opciones pendientes (_backtracking_).
4. El último nodo en entrar es el primero en ser explorado.

## Pseudocódigo:
```
1. DFS(Grafo, NodoActual, Visitados)
2.    Insertar NodoActual en Visitados
3.    PROCESAR NodoActual (imprimir o guardar)
4.    PARA cada Vecino de NodoActual:
5.       SI Vecino no está en Visitados:
6.          DFS(Grafo, Vecino, Visitados)
```
