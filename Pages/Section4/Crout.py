import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math
from streamlit_extras.echo_expander import echo_expander

def is_zero_matrix(A):
    """
    The function checks if a given matrix is a zero matrix, i.e., all its elements are zero.

    :param A: A is a matrix (a list of lists) that is being checked to see if it is a zero matrix, meaning all of its
    elements are zero
    :return: a boolean value. It returns True if the input matrix A is a zero matrix (i.e., all elements are zero), and
    False otherwise.
    """
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] != 0:
                return False
    return True


st.cache(max_entries=1000)
def crout_factorization(A):
    """
    The crout_factorization function performs Crout factorization on a given matrix A and returns the lower triangular
    matrix L and upper triangular matrix U.

    :param A: A is a square matrix of size n x n that needs to be factorized into a lower triangular matrix L and an upper
    triangular matrix U using Crout's method
    :return: The function `crout_factorization` returns two matrices, `L` and `U`, which are the lower and upper triangular
    matrices obtained from the Crout factorization of the input matrix `A`.
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Paso 1: Inicializar L y U con ceros
    for i in range(n):
        L[i][i] = 1

    # Paso 2: Calcular elementos de L y U
    for i in range(n):
        for j in range(i, n):
            sum1 = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - sum1

        for j in range(i+1, n):
            sum2 = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - sum2) / U[i][i]

    return L, U


st.header('4. Factorización LU y sus Aplicaciones')



r'''
# 4.4 Solución de sistemas bandados (Método de Crout)

El Método de Crout es un algoritmo utilizado para resolver sistemas de ecuaciones lineales en los cuales la matriz de
coeficientes es una matriz bandada. Una matriz bandada es aquella en la cual la mayoría de los elementos son ceros,
excepto aquellos que se encuentran cerca de la diagonal principal y algunas diagonales adyacentes.

El objetivo del Método de Crout es descomponer la matriz de coeficientes $A$ en dos matrices: una matriz triangular
inferior $L$ y una matriz triangular superior $U$, de tal manera que $A = LU$. La matriz $L$ tiene unos en la diagonal
principal y ceros por encima de la diagonal, mientras que la matriz $U$ tiene la diagonal principal y elementos
distintos de cero por debajo de la diagonal.

Una vez que se ha obtenido la descomposición $A = LU$, se puede resolver el sistema de ecuaciones $Ax = b$ mediante los
siguientes pasos:

1. Se resuelve el sistema triangular inferior $Ly = b$ mediante sustitución hacia adelante, donde $y$ es un vector
desconocido.
2. Se resuelve el sistema triangular superior $Ux = y$ mediante sustitución hacia atrás, donde $x$ es el vector solución
del sistema original.

El Método de Crout es especialmente eficiente para sistemas de ecuaciones bandadas, ya que aprovecha la estructura
especial de la matriz para reducir la cantidad de operaciones requeridas.

En resumen, el Método de Crout es una técnica rigurosa y eficiente para resolver sistemas de ecuaciones lineales con
matrices bandadas. Al descomponer la matriz de coeficientes en las matrices $L$ y $U$, se simplifica la resolución del
sistema mediante sustitución hacia adelante y hacia atrás, obteniendo una solución precisa y confiable.

$$A = LU$$

Donde:
$A$ es la matriz de coeficientes,
$L$ es la matriz triangular inferior con unos en la diagonal principal y ceros por encima de la diagonal,
$U$ es la matriz triangular superior con la diagonal principal y elementos distintos de cero por debajo de la diagonal.

Para resolver el sistema de ecuaciones $Ax = b$, se realiza la siguiente secuencia de pasos:
1. Resolver el sistema triangular inferior $Ly = b$ mediante sustitución hacia adelante.
2. Resolver el sistema triangular superior $Ux = y$ mediante sustitución hacia atrás.

El Método de Crout es una herramienta poderosa para resolver sistemas de ecuaciones lineales bandadas, brindando una
solución eficiente y precisa.


## Algoritmo
El algoritmo del Método de Crout para la factorización LU de una matriz bandada se puede describir de la siguiente manera:

**Entrada**: Matriz de coeficientes $A$ de tamaño $n \times n$ y vector de términos independientes $b$ de tamaño $n$.

1. Inicializar la matriz triangular inferior $L$ y la matriz triangular superior $U$ con ceros del mismo tamaño que $A$.
2. Para cada fila $i$ desde 1 hasta $n$, hacer:
     - Para cada columna $j$ desde 1 hasta $i$, calcular el elemento $L_{ij}$ utilizando la fórmula:
       $$L_{ij} = A_{ij} - \sum_{k=1}^{j-1} L_{ik}U_{kj}$$
     - Para cada columna $j$ desde $i$ hasta $n$, calcular el elemento $U_{ij}$ utilizando la fórmula:
       $$U_{ij} = \frac{1}{L_{ii}} \left(A_{ij} - \sum_{k=1}^{i-1} L_{ik}U_{kj}\right)$$
3. Calcular el vector solución $x$ mediante los siguientes pasos:
     - Resolver el sistema triangular inferior $Ly = b$ mediante sustitución hacia adelante, obteniendo el vector $y$.
     - Resolver el sistema triangular superior $Ux = y$ mediante sustitución hacia atrás, obteniendo el vector solución $x$.
4. Devolver el vector solución $x$.

El algoritmo de Crout es un método eficiente para resolver sistemas de ecuaciones lineales bandadas, ya que evita la
necesidad de realizar operaciones costosas con ceros en la matriz.

Es importante tener en cuenta que el Método de Crout solo es aplicable a matrices bandadas y puede requerir
modificaciones si la matriz tiene elementos nulos fuera de la banda.

## Ejemplo

Supongamos que tenemos la siguiente matriz bandada:

$$
A = \begin{bmatrix}
2 & 3 & 0 & 0 \\
1 & 4 & 5 & 0 \\
0 & 6 & 7 & 8 \\
0 & 0 & 9 & 10
\end{bmatrix}
$$

Esta matriz tiene una banda tridiagonal, lo que significa que los elementos no nulos están en la diagonal principal y en las dos diagonales adyacentes a ella.

Aplicaremos el Método de Crout para obtener la factorización LU de esta matriz.

Comenzamos inicializando la matriz triangular inferior $L$ y la matriz triangular superior $U$ con ceros:

$$
L = \begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix},
\quad
U = \begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

Luego, aplicamos el algoritmo del Método de Crout para calcular los elementos de las matrices $L$ y $U$:

1. Para la fila 1:
   - Calculamos $L_{11}$ y $U_{11}$:
     $$L_{11} = 2, \quad U_{11} = 1$$
   - No hay elementos $L$ ni $U$ en la banda anterior a la diagonal principal.

2. Para la fila 2:
   - Calculamos $L_{21}$, $L_{22}$ y $U_{22}$:
     $$L_{21} = \frac{1}{L_{11}} \cdot A_{21} = \frac{1}{2} \cdot 1 = \frac{1}{2}, \quad L_{22} = A_{22} - L_{21} \cdot U_{12} = 4 - \frac{1}{2} \cdot 3 = \frac{5}{2}, \quad U_{22} = 1$$
   - No hay elementos $L$ ni $U$ en la banda anterior a la diagonal principal.

3. Para la fila 3:
   - Calculamos $L_{31}$, $L_{32}$, $L_{33}$, $U_{32}$ y $U_{33}$:
     $$L_{31} = \frac{1}{L_{11}} \cdot A_{31} = \frac{1}{2} \cdot 0 = 0, \quad L_{32} = \frac{1}{L_{22}} \cdot A_{32} = \frac{2}{5} \cdot 6 = \frac{12}{5}, \quad L_{33} = A_{33} - L_{31} \cdot U_{13} - L_{32} \cdot U_{23} = 7 - 0 - \frac{12}{5} \cdot 0 = 7, \quad U_{32} = \frac{1}{L_{22}} \cdot A_{23} = \frac{2}{5} \cdot 5 = 2, \quad U_{33} = 1$$
   - No hay elementos $L$ ni $U$ en la banda anterior a la diagonal principal.

4. Para la fila 4:
   - Calculamos $L_{41}$, $L_{42}$, $L_{43}$, $L_{44}$, $U_{42}$ y $U_{43}$:
     $$L_{41} = \frac{1}{L_{11}} \cdot A_{41} = \frac{1}{2} \cdot 0 = 0, \quad L_{42} = \frac{1}{L_{22}} \cdot A_{42} = \frac{2}{5} \cdot 0 = 0, \quad L_{43} = \frac{1}{L_{33}} \cdot A_{43} = \frac{1}{7} \cdot 8 = \frac{8}{7}, \quad L_{44} = A_{44} - L_{41} \cdot U_{14} - L_{42} \cdot U_{24} - L_{43} \cdot U_{34} = 10 - 0 - 0 - \frac{8}{7} \cdot 9 = \frac{10}{7}, \quad U_{42} = \frac{1}{L_{22}} \cdot A_{24} = \frac{2}{5} \cdot 0 = 0, \quad U_{43} = \frac{1}{L_{33}} \cdot A_{34} = \frac{1}{7} \cdot 9 = \frac{9}{7}$$

Luego de completar los pasos anteriores, obtenemos las matrices $L$ y $U$:

$$
L = \begin{bmatrix}
2 & 0 & 0 & 0 \\
\frac{1}{2} & \frac{5}{2} & 0 & 0 \\
0 & \frac{12}{5} & 7 & 0 \\
0 & 0 & \frac{8}{7} & \frac{10}{7}
\end{bmatrix},
\quad
U = \begin{bmatrix}
1 & 3 & 0 & 0 \\
0 & 1 & 5 & 0 \\
0 & 0 & 1 & 8 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

La factorización LU de la matriz banda $A$ se puede utilizar para resolver sistemas de ecuaciones lineales de forma
eficiente, ya que reduce la cantidad de operaciones necesarias.

'''

st.subheader('Método :triangular_ruler:')


r = st.number_input('Ingrese el tamño de la matriz:', min_value=1, max_value=10,value=4)


cols = [str(i+1) for i in range(r)]
mat1 = pd.DataFrame(np.zeros((r,r)),columns=cols)
st.write('Ingrese la matriz:')
mat2 = st.data_editor(mat1, key='mat1')


m = [[(sp.Matrix(mat2))[i,j] for j in range(r)] for i in range(r)]


st.latex('A = ' + sp.latex(sp.Matrix(m)))

if st.button('Calcular'):
    if not(is_zero_matrix(m)):

        try:
            if sp.Matrix(m).det() == 0:
                st.write('La matriz no tiene solucion :(  **|A| = 0**')
            else:
                solucion = crout_factorization(np.array(m).astype(float))
                st.write('Matriz triangular inferior L:')
                st.latex(r'''\mathbb{L} \approx ''' + sp.latex(sp.Matrix(solucion[0])) )
                st.write('Matriz triangular superior U:')
                st.latex(r'''\mathbb{U} \approx ''' + sp.latex(sp.Matrix(solucion[1])) )

        except:
            if sp.Matrix(m).det() == 0:
                st.write('La matriz no tiene solucion :(')
            else:
                st.write('Algo salio mal :(')

with echo_expander(code_location="below", label="Implementación en Python"):
    import numpy as np
    import sympy as sp
    def crout_factorization(A):
        n = len(A)
        L = np.zeros((n, n))
        U = np.zeros((n, n))

        # Paso 1: Inicializar L y U con ceros
        for i in range(n):
            L[i][i] = 1

        # Paso 2: Calcular elementos de L y U
        for i in range(n):
            for j in range(i, n):
                sum1 = sum(L[i][k] * U[k][j] for k in range(i))
                U[i][j] = A[i][j] - sum1

            for j in range(i+1, n):
                sum2 = sum(L[j][k] * U[k][i] for k in range(i))
                L[j][i] = (A[j][i] - sum2) / U[i][i]

        return L, U
