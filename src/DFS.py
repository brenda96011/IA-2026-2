'''
=============================================================================
UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO (UNAM)
Facultad de Ciencias
Materia: Inteligencia Artificial
Docente: Dra. Jessica Sarahi Méndez Rincón
Ayudante de Laboratorio: Diego Eduardo Peña Villegas
Alumno: Brenda Rodríguez Jiménez
Año escolar: 2026-2
Copyright: (c) 2025 UNAM - MIT License
Version: 1.0
This software is for educational purposes. 
The accuracy of the models depends strictly on the quality 
and preprocessing of the input data.
-----------------------------------------------------------------------------
UNAM IA Library: A professional toolkit for AI developed at UNAM.
=============================================================================
'''

def dfs(grafo, nodo, visitados=None):
    if visitados is None:
        visitados = []

    visitados.append(nodo)
    print(nodo, end=" ")

    for v in grafo[nodo]:
        if v not in visitados:
            dfs(grafo, v, visitados)


grafo = {0:[1,2,3], 1:[0,4,5], 2:[0,6], 3:[0,7], 4:[1], 5:[1], 6:[2], 7:[3]}
print("DFS empezando desde el nodo 0:")
dfs(grafo, 0)
