import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math


st.title('3. Solución de Sistemas de Ecuaciones Lineales')

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


def gauss_pivoteo_total(A, b):
    n = len(A)
    # Crear matriz aumentada
    aumentada = np.concatenate((A, np.array([b]).T), axis=1)
    steps = []

    # Eliminación gaussiana con pivoteo parcial
    for i in range(n-1):

        # Encuentra el índice del pivote máximo en valor absoluto
        pivot_index = np.argmax(np.abs(aumentada[i:, i])) + i
        #print(aumentada[max_index])

        if pivot_index != i:
            aumentada[[i, pivot_index]] = aumentada[[pivot_index, i]]
            steps.append(f'R_{i} <---> R_{pivot_index}')
            steps.append(sp.latex(sp.Matrix(aumentada)))

        # Eliminación gaussiana
        for j in range(i+1, n):
            factor = aumentada[j, i] / aumentada[i, i]
            steps.append(str(-1*factor)+'*R_'+str(i)+' + R_'+str(j)+' <---> R_'+str(j))
            steps.append(sp.latex(sp.Matrix(aumentada)))
        aumentada[j] = aumentada[j]- (factor * aumentada[i])



    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = aumentada[i][-1]
        for j in range(i+1, n):
            x[i] -= aumentada[i][j] * x[j]
        x[i] /= aumentada[i][i]

    return x,steps, np.round(aumentada.astype(float), decimals=4)

r'''
# 3.2.2 Método de Gauss-Jordan y pivoteo total

El Método de Gauss-Jordan es una técnica utilizada para resolver sistemas de ecuaciones lineales mediante operaciones de
fila en una matriz aumentada. El pivoteo total es una variante de este método que busca evitar divisiones por cero y
reducir el error numérico.

El proceso comienza con la construcción de una matriz aumentada que representa el sistema de ecuaciones lineales. Luego,
se realizan operaciones de fila para convertir la matriz en una forma escalonada reducida por filas, también conocida
como forma canónica.

El pivoteo total implica seleccionar el elemento máximo en valor absoluto en cada columna como el pivote, lo que ayuda a
minimizar los errores de redondeo y garantiza una mejor estabilidad numérica.

A través de una serie de pasos, se realiza una eliminación hacia adelante y hacia atrás para convertir la matriz en una
forma escalonada reducida por filas. Esto implica la reducción de los coeficientes por debajo y por encima de cada pivote
a cero mediante la combinación lineal de filas.

Al finalizar el proceso, se obtiene una matriz en forma canónica, donde las filas representan las ecuaciones y las columnas
contienen los coeficientes y las soluciones del sistema de ecuaciones. La solución del sistema se puede leer directamente a
partir de la matriz resultante.

El Método de Gauss-Jordan y el pivoteo total son herramientas poderosas para resolver sistemas de ecuaciones lineales de
manera precisa y eficiente, brindando soluciones numéricas confiables.

La eliminación de Gauss-Jordan y el pivoteo total se pueden expresar como:

$$
\begin{array}{cccc|c}
a_{11} & a_{12} & \cdots & a_{1n} & b_1 \\
a_{21} & a_{22} & \cdots & a_{2n} & b_2 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} & b_m \\
\end{array}
$$

El proceso de eliminación se realiza mediante operaciones de fila, como intercambio de filas, multiplicación de filas
por un escalar y suma/resta de filas multiplicadas por un escalar.

- Intercambio de filas: $R_i \leftrightarrow R_j$
- Multiplicación de una fila por un escalar: $k \cdot R_i$
- Suma/resta de filas multiplicadas por un escalar: $R_i \pm k \cdot R_j$

El pivoteo total implica seleccionar el pivote como el elemento máximo en valor absoluto en cada columna. :

$$
\text{{Pivote}} = \max_{i=k}^{m} |a_{ik}|
$$

A lo largo del proceso de eliminación, se aplican estas operaciones hasta obtener la forma escalonada reducida por filas,
que se puede expresar como:

$$
\begin{array}{cccc|c}
1 & 0 & \cdots & 0 & c_1 \\
0 & 1 & \cdots & 0 & c_2 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & 1 & c_m \\
\end{array}
$$

donde $c_1, c_2, \ldots, c_m$ son las soluciones del sistema de ecuaciones lineales.

El Método de Gauss-Jordan y el pivoteo total son herramientas fundamentales en el ámbito del análisis numérico y
encuentran aplicaciones en diversos campos de la ciencia.


## Ejemplo

Consideremos el siguiente sistema de ecuaciones lineales:

$$
\begin{align*}
2x + 3y - z &= 1 \\
x - 2y + 4z &= -2 \\
3x + y - 3z &= 3 \\
\end{align*}
$$

Podemos escribir este sistema en forma matricial como:

$$
\begin{bmatrix}
2 & 3 & -1 & \,|\, 1 \\
1 & -2 & 4 & \,|\, -2 \\
3 & 1 & -3 & \,|\, 3 \\
\end{bmatrix}
$$

Ahora, aplicaremos el Método de Gauss-Jordan con pivoteo total para convertir esta matriz en su forma escalonada
reducida por filas.

Paso 1: Encontrar el pivote máximo en valor absoluto en la columna 1.

El elemento máximo en valor absoluto es 3 en la fila 3. Realizamos el intercambio de filas para que la fila 3 se
convierta en la primera fila.

$$
\begin{bmatrix}
3 & 1 & -3 & \,|\, 3 \\
1 & -2 & 4 & \,|\, -2 \\
2 & 3 & -1 & \,|\, 1 \\
\end{bmatrix}
$$

Paso 2: Convertir el pivote en 1.

Dividimos la primera fila por 3 para convertir el pivote en 1.

$$
\begin{bmatrix}
1 & \frac{1}{3} & -1 & \,|\, 1 \\
1 & -2 & 4 & \,|\, -2 \\
2 & 3 & -1 & \,|\, 1 \\
\end{bmatrix}
$$

Paso 3: Hacer ceros debajo y encima del pivote.

Restamos la primera fila de las filas 2 y 3 multiplicadas por el coeficiente adecuado para hacer ceros debajo y
encima del pivote.

$$
\begin{bmatrix}
1 & \frac{1}{3} & -1 & \,|\, 1 \\
0 & -\frac{7}{3} & 5 & \,|\, -3 \\
0 & \frac{7}{3} & -3 & \,|\, -1 \\
\end{bmatrix}
$$

Paso 4: Continuar el proceso para convertir la matriz en forma escalonada reducida por filas.

Dividimos la segunda fila por $-\frac{7}{3}$ para convertir el nuevo pivote en 1.

$$
\begin{bmatrix}
1 & \frac{1}{3} & -1 & \,|\, 1 \\
0 & 1 & -\frac{15}{7} & \,|\, \frac{9}{7} \\
0 & \frac{7}{3} & -3 & \,|\, -1 \\
\end{bmatrix}
$$

Restamos la segunda fila multiplicada por $\frac{7}{3}$ de la tercera fila para hacer cero el coeficiente debajo del
pivote.

$$
\begin{bmatrix}
1 & \frac{1}{3} & -1 & \,|\, 1 \\
0 & 1 & -

\frac{15}{7} & \,|\, \frac{9}{7} \\
0 & 0 & 0 & \,|\, -\frac{8}{7} \\
\end{bmatrix}
$$

En este punto, la tercera fila consiste en ceros y la solución no es única. Podemos ver que el sistema es incompatible y
no tiene solución.

El Método de Gauss-Jordan y el pivoteo total nos permitieron obtener la forma escalonada reducida por filas de la matriz
y determinar que el sistema no tiene solución.

'''



st.subheader('Método :triangular_ruler:')


r = st.number_input('Ingrese el tamño de la matriz:', min_value=1, max_value=10,value=4)


cols = [str(i+1) for i in range(r)]
cols.append('x')
mat1 = pd.DataFrame(np.zeros((r,r+1)),columns=cols)
st.write('Ingrese la matriz:')
mat2 = st.data_editor(mat1, key='mat1')






m = [[(sp.Matrix(mat2))[i,j] for j in range(r)] for i in range(r)]
b = [(sp.Matrix(mat2))[i,-1] for i in range(r)]

st.latex('A = ' + sp.latex(sp.Matrix(m))+ ' , b = '+sp.latex(sp.Matrix(b)))


if not(is_zero_matrix(m)):
    try:
        if sp.Matrix(m).det() == 0:
            st.write('La matriz no tiene solucion :(')
        else:
            solucion = gauss_pivoteo_total(np.array(m), np.array(b))
            st.write('Matriz escalonada:')
            st.latex(sp.latex(sp.Matrix(solucion[2])))
            st.write('Solucion aproximada:')
            st.latex(r'''\hat{x} \approx ''' + sp.latex(sp.Matrix(solucion[0])))
            sol = sp.Matrix(m).inv() * sp.Matrix(b)
            st.write('Error con respecto a la solucion:')
            st.latex(' \hat{x} = ' + sp.latex(sp.Matrix(solucion[0])))
            st.latex('error = ' + sp.latex(abs(sol-sp.Matrix(solucion[0]))))
            st.write('Pasos realizados:')
            for t in solucion[1]:
                st.latex(t)
    except:
        if sp.Matrix(m).det() == 0:
            st.write('La matriz no tiene solucion :(')
        else:
            st.write('Algo salio mal :(')




