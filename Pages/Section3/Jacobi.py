import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math



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
def jacobi(A, b, x0, tol, max_iter):
    n = len(A)
    x = x0.copy()
    x_prev = x0.copy()
    iteration = 0
    steps = [[i for i in x0]]
    steps[0].append(0)
    while iteration < max_iter:
        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x_prev[j]

            x[i] = (b[i] - s) / A[i, i]
        xapp = x.copy()
        xapp.append(np.linalg.norm(np.array(x).astype(float) - np.array(x_prev).astype(float)))
        xapp.append(str(np.linalg.norm(np.array(x).astype(float) - np.array(x_prev).astype(float)) < tol))

        steps.append(xapp)


        if np.linalg.norm(np.array(x).astype(float) - np.array(x_prev).astype(float)) < tol:
            break

        x_prev = x.copy()
        iteration += 1

    return x,steps

st.title('3. Solución de Sistemas de Ecuaciones Lineales')


r'''
# 3.4.2 Método de Jacobi

El Método de Jacobi es una técnica iterativa utilizada para resolver sistemas de ecuaciones lineales. Dado un sistema
de ecuaciones lineales de la forma:

$$
\begin{align*}
a_{11}x_1 + a_{12}x_2 + \ldots + a_{1n}x_n &= b_1 \\
a_{21}x_1 + a_{22}x_2 + \ldots + a_{2n}x_n &= b_2 \\
\ldots \\
a_{n1}x_1 + a_{n2}x_2 + \ldots + a_{nn}x_n &= b_n \\
\end{align*}
$$

donde $a_{ij}$ son los coeficientes de las variables $x_i$ y $b_i$ son los términos constantes, el objetivo es encontrar
el vector de soluciones $x = [x_1, x_2, \ldots, x_n]$.

El método de Jacobi se basa en descomponer la matriz de coeficientes $A$ en una suma de una matriz diagonal $D$, una
matriz triangular inferior $L$, y una matriz triangular superior $U$, es decir:

$$
A = D + L + U
$$

donde $D$ contiene los elementos de la diagonal de $A$, $L$ contiene los elementos debajo de la diagonal y $U$ contiene
los elementos encima de la diagonal.

La idea central del método de Jacobi es iterar hasta encontrar una solución aproximada al sistema de ecuaciones.
En cada iteración, se calcula una nueva aproximación para cada variable $x_i$ utilizando la fórmula:

$$
x_i^{(k+1)} = \frac{1}{a_{ii}} \left(b_i - \sum_{j=1, j \neq i}^{n} a_{ij}x_j^{(k)}\right)
$$

donde $x_i^{(k+1)}$ es la nueva aproximación de $x_i$ en la iteración $k+1$, $x_j^{(k)}$ es el valor actual de $x_j$ en
la iteración $k$, $a_{ii}$ es el coeficiente de la variable $x_i$ en la ecuación correspondiente, y $b_i$ es el término
constante.

Este proceso se repite hasta que se cumple un criterio de convergencia, como por ejemplo, que la norma del vector de
diferencia entre dos iteraciones consecutivas sea menor que una tolerancia predefinida.

El método de Jacobi tiene algunas ventajas y limitaciones. Por un lado, es un método iterativo que puede aplicarse a
sistemas de ecuaciones grandes y dispersos. Además, se garantiza la convergencia cuando la matriz de coeficientes es
estrictamente diagonal dominante. Sin embargo, la convergencia puede ser lenta en algunos casos y depende de la elección
adecuada de los valores iniciales. Además, no todos los sistemas de ecuaciones tienen una matriz diagonalizable,
lo que puede afectar la convergencia del método.

En resumen, el método de Jacobi es una técnica iterativa utilizada para resolver sistemas de ecuaciones lineales.
A través de la descomposición de la matriz de coeficientes y la actualización iterativa de las variables, se busca
encontrar una solución aproximada.


## Ejemplo

A continuación se presenta un ejemplo numérico para ilustrar la aplicación del Método de Jacobi en la resolución de un
sistema de ecuaciones lineales.

Supongamos que tenemos el siguiente sistema de ecuaciones:

$$
\begin{align*}
3x + y - z &= 5 \\
2x - 4y + 2z &= -3 \\
x + y + 3z &= 8 \\
\end{align*}
$$

El objetivo es encontrar las soluciones para las variables $x$, $y$ y $z$.

Para aplicar el Método de Jacobi, primero debemos descomponer la matriz de coeficientes y el vector de términos constantes:

$$
A = \begin{bmatrix}
3 & 1 & -1 \\
2 & -4 & 2 \\
1 & 1 & 3 \\
\end{bmatrix},
\quad
b = \begin{bmatrix}
5 \\
-3 \\
8 \\
\end{bmatrix}
$$

La matriz $A$ se descompone en una matriz diagonal $D$, una matriz triangular inferior $L$ y una matriz triangular
superior $U$, de la siguiente manera:

$$
D = \begin{bmatrix}
3 & 0 & 0 \\
0 & -4 & 0 \\
0 & 0 & 3 \\
\end{bmatrix},
\quad
L = \begin{bmatrix}
0 & 0 & 0 \\
-2 & 0 & 0 \\
-1 & -1 & 0 \\
\end{bmatrix},
\quad
U = \begin{bmatrix}
0 & -1 & 1 \\
0 & 0 & 2 \\
0 & 0 & 0 \\
\end{bmatrix}
$$

A continuación, se elige una aproximación inicial para las variables $x$, $y$ y $z$. Por ejemplo, podemos tomar
$x^{(0)} = y^{(0)} = z^{(0)} = 0$.

Luego, aplicamos la fórmula de actualización iterativa del Método de Jacobi para obtener las nuevas aproximaciones
$x^{(k+1)}$, $y^{(k+1)}$ y $z^{(k+1)}$ en cada iteración $k$:

$$
x^{(k+1)} = \frac{1}{3} \left(5 - y^{(k)} + z^{(k)}\right)
$$
$$
y^{(k+1)} = \frac{1}{-4} \left(-3 - 2x^{(k)} + 2z^{(k)}\right)
$$
$$
z^{(k+1)} = \frac{1}{3} \left(8 - x^{(k)} - y^{(k)}\right)
$$

Iteramos este proceso hasta que se cumpla un criterio de convergencia, por ejemplo, cuando la norma del vector de
diferencia entre dos iteraciones consecutivas sea menor que una tolerancia predefinida.

El resultado final será una aproximación de la solución del sistema de ecuaciones, es decir, los valores aproximados de
$x$, $y$ y $z$.

Es importante destacar que el Método de Jacobi puede requerir un número variable de iteraciones según el sistema de
ecuaciones y la elección de valores iniciales. Además, para asegurar la convergencia, es necesario que la matriz de
coeficientes sea diagonal dominante o estrictamente diagonal dominante.

En este ejemplo específico, aplicando el Método de Jacobi con una tolerancia de $10^{-6}$, después de varias
iteraciones, se obtiene la solución aproximada $x \approx 1.777$, $y \approx -0.469$ y $z \approx 2.331$.

El Método de Jacobi es una herramienta útil en el análisis numérico para resolver sistemas de ecuaciones lineales,
y su aplicación se extiende a diversas áreas de la ciencia y la ingeniería.

'''



st.subheader('Método :triangular_ruler:')


r = st.number_input('Ingrese el tamño de la matriz:', min_value=1, max_value=10,value=4)


cols = [str(i+1) for i in range(r)]
cols.append('x')
mat1 = pd.DataFrame(np.zeros((r,r+1)),columns=cols)
st.write('Ingrese la matriz:')
mat2 = st.data_editor(mat1, key='mat1')

x0 = st.text_input('Ingrese el valor inicial de $x_0$:',value=str(list(np.zeros(r))))
error=st.text_input('Ingrese el error de tolerancia:',value='1e-6')

max_iter = st.number_input('Ingrese el número máximo de iteraciones:', min_value=1, max_value=10000,value=10)



m = [[(sp.Matrix(mat2))[i,j] for j in range(r)] for i in range(r)]
b = [(sp.Matrix(mat2))[i,-1] for i in range(r)]

st.latex('A = ' + sp.latex(sp.Matrix(m))+ ' , b = '+sp.latex(sp.Matrix(b)))


if not(is_zero_matrix(m)):

    try:
        if sp.Matrix(m).det() == 0:
            st.write('La matriz no tiene solucion :(  **|A| = 0**')
        else:
            solucion = jacobi(np.array(m), np.array(b),sp.parse_expr(x0),float(error),max_iter)
    #        st.write('Matriz escalonada:')
    #        st.latex(sp.latex(sp.Matrix(solucion[-1])))
            st.write('Solucion aproximada:')
            st.latex(r'''\hat{x} \approx ''' + sp.latex(sp.Matrix(solucion[0])))
            sol = sp.Matrix(m).inv() * sp.Matrix(b)
            st.write('Error con respecto a la solucion:')
            st.latex(' \hat{x} = ' + sp.latex(sp.Matrix(sol)))
            st.latex('error = ' + sp.latex(abs(sol-sp.Matrix(solucion[0]))))
    #        st.write('Pasos realizados:')
            cols = ['x_'+str(i) for i in range(r)]
            cols.append('|| x_k ||  ')
            cols.append('|| x_k || < '+error)
            spd = pd.DataFrame(solucion[1], columns=cols)
            st.dataframe(spd)
    except:
        if sp.Matrix(m).det() == 0:
            st.write('La matriz no tiene solucion :(')
        else:
            st.write('Algo salio mal :(')
