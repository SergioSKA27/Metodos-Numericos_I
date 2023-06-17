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

st.title('3. Solución de Sistemas de Ecuaciones Lineales')
def gauss_seidel(A, b, omega, x0, tol, max_iter):
    n = len(A)
    x = x0.copy()
    iteration = 0
    error = tol + 1
    steps = [[i for i in x0]]
    steps[0].append(0)

    while error > tol and iteration < max_iter:
        x_prev = x.copy()

        for i in range(n):
            sum1 = 0
            for j in range(i):
                sum1 += A[i, j] * x[j]

            sum2 = 0
            for j in range(i + 1, n):
                sum2 += A[i, j] * x_prev[j]

            x[i] = (1 - omega) * x_prev[i] + (omega / A[i, i]) * (b[i] - sum1 - sum2)
        xapp = list(x.copy())
        xapp.append(np.linalg.norm(np.array(x).astype(float) - np.array(x_prev).astype(float)))
        xapp.append(str(np.linalg.norm(np.array(x).astype(float) - np.array(x_prev).astype(float)) < tol))
        steps.append(xapp)
        error = np.linalg.norm(np.array(x).astype(float) - np.array(x_prev).astype(float))
        x_prev = x.copy()
        iteration += 1

    return x,steps

r'''
# 3.4.4 Método de relajación

El Método de Relajación, también conocido como Método de Sobrerrelajación Sucesiva (SOR, por sus siglas en inglés), es
un algoritmo iterativo utilizado para resolver sistemas de ecuaciones lineales. A diferencia del Método de Gauss-Seidel,
el Método de Relajación introduce un factor de relajación para mejorar la convergencia.

Consideremos un sistema de ecuaciones lineales de la forma $Ax = b$, donde $A$ es una matriz de coeficientes $n \times n$,
$x$ es el vector desconocido de tamaño $n$, y $b$ es el vector de términos independientes de tamaño $n$.

El Método de Relajación se basa en descomponer la matriz $A$ en tres componentes: una parte inferior triangular ($L$),
una parte diagonal ($D$), y una parte superior triangular ($U$). Entonces, la matriz $A$ puede expresarse
como $A = L + D + U$.

La idea principal del Método de Relajación es actualizar la solución en cada iteración mediante una combinación
de la solución obtenida en la iteración anterior y una solución parcial calculada utilizando el Método de Gauss-Seidel.
La fórmula de actualización es la siguiente:

$
x_i^{(k+1)} = (1 - \omega) x_i^{(k)} + \frac{\omega}{a_{ii}} \left(b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} a_{ij} x_j^{(k)}\right)
$

donde $x_i^{(k+1)}$ es el valor actualizado de la variable $x_i$ en la iteración $k+1$, $a_{ij}$ son los coeficientes
de la matriz $A$, $b_i$ es el término independiente correspondiente, $n$ es el tamaño del sistema de ecuaciones, $\omega$ es el factor de relajación y $k$ es el número de iteraciones.

El factor de relajación $\omega$ influye en la convergencia del método. Si $\omega = 1$, el Método de Relajación se
reduce al Método de Gauss-Seidel. Valores de $\omega$ mayores a 1 aceleran la convergencia, pero pueden llevar a
inestabilidades si se exceden ciertos límites. Por otro lado, valores de $\omega$ menores a 1 pueden suavizar
las oscilaciones en el proceso iterativo, pero pueden aumentar la cantidad de iteraciones necesarias para converger.

Es importante destacar que la convergencia del Método de Relajación depende de las propiedades de la matriz $A$.
En particular, el método converge si y solo si la matriz $A$ es estrictamente diagonalmente dominante o si es
simétrica y definida positiva.

En resumen, el Método de Relajación es un algoritmo iterativo utilizado para resolver sistemas de ecuaciones lineales.
Mediante el uso de un factor de relajación, permite mejorar la convergencia del Método de Gauss-Seidel. Sin embargo,
es importante tener en cuenta las propiedades de la matriz y elegir adecuadamente el valor del factor de relajación
para garantizar la convergencia y estabilidad del método.


## Ejemplo

Consideremos el siguiente sistema de ecuaciones lineales:

$$
\begin{align*}
3x_1 - x_2 + x_3 &= 4 \\
x_1 + 5x_2 - 2x_3 &= 3 \\
2x_1 - 3x_2 + 7x_3 &= 7
\end{align*}
$$

Podemos expresar este sistema en forma matricial como $Ax = b$, donde

$$
A = \begin{bmatrix} 3 & -1 & 1 \\ 1 & 5 & -2 \\ 2 & -3 & 7 \end{bmatrix}, \quad x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}, \quad b = \begin{bmatrix} 4 \\ 3 \\ 7 \end{bmatrix}
$$

Para resolver este sistema utilizando el Método de Relajación, necesitamos descomponer la matriz $A$ en sus componentes
$D$, $L$ y $U$. En este caso, tenemos:

$$
D = \begin{bmatrix} 3 & 0 & 0 \\ 0 & 5 & 0 \\ 0 & 0 & 7 \end{bmatrix}, \quad L = \begin{bmatrix} 0 & 0 & 0 \\ 1 & 0 & 0 \\ 2 & -3 & 0 \end{bmatrix}, \quad U = \begin{bmatrix} 0 & -1 & 1 \\ 0 & 0 & -2 \\ 0 & 0 & 0 \end{bmatrix}
$$

El próximo paso es elegir un valor apropiado para el factor de relajación $\omega$. Supongamos que seleccionamos
$\omega = 1.2$.

A continuación, podemos aplicar la fórmula iterativa del Método de Relajación para obtener las soluciones aproximadas
en cada iteración. Supongamos que empezamos con una estimación
inicial $x^{(0)} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$.

Las iteraciones se realizan de la siguiente manera:

- Iteración 1:
  $$
  \begin{align*}
  x_1^{(1)} &= (1 - 1.2) \cdot 0 + \frac{1.2}{3} \left(4 - (-1 \cdot 0) - (1 \cdot 0)\right) = 1.2 \\
  x_2^{(1)} &= (1 - 1.2) \cdot 0 + \frac{1.2}{5} \left(3 - (1 \cdot 0) - (-2 \cdot 0)\right) = 0.72 \\
  x_3^{(1)} &= (1 - 1.2) \cdot 0 + \frac{1.2}{7} \left(7 - (2 \cdot 0) - (-3 \cdot 0)\right) = 1.028571
  \end{align*}
  $$

- Iteración 2:
  $$
  \begin{align*}
  x_1^{(2)} &= (1 - 1.2) \cdot 1.2 + \frac{1.2}{3} \left(4 - (-1 \cdot 0.72) - (1 \cdot 1.028571)\right) = 1.174286 \\
  x_2^{(2)} &= (1 - 1.2) \cdot 0.72 + \frac{1.2}{5} \left(3 - (1 \cdot 1.2) - (-2 \cdot 1.028571)\right) = 0.828 \\
  x_3^{(2)} &= (1 - 1.2) \cdot 1.028571 + \frac{1.2}{7} \left(7 - (2 \cdot 1.174286) - (-3 \cdot 0.828)\right) = 0.987082
  \end{align*}
  $$

- Iteración 3:
  $$
  \begin{align*}
  x_1^{(3)} &= (1 - 1.2) \cdot 1.174286 + \frac{1.2}{3} \left(4 - (-1 \cdot 0.828) - (1 \cdot 0.987082)\right) = 1.040134 \\
  x_2^{(3)} &= (1 - 1.2) \cdot 0.828 + \frac{1.2}{5} \left(3 - (1 \cdot 1.174286) - (-2 \cdot 0.987082)\right) = 0.836297 \\
  x_3^{(3)} &= (1 - 1.2) \cdot 0.987082 + \frac{1.2}{7} \left(7 - (2 \cdot 1.040134) - (-3 \cdot 0.836297)\right) = 1.003855
  \end{align*}
  $$

Estas iteraciones continúan hasta que se alcance una tolerancia predefinida o se cumpla un número máximo de iteraciones.

Al finalizar, obtenemos la solución aproximada del sistema de ecuaciones
como $x \approx \begin{bmatrix} 1.040134 \\ 0.836297 \\ 1.003855 \end{bmatrix}$.

Es importante destacar que el Método de Relajación puede converger a la solución exacta o
proporcionar una buena aproximación, dependiendo del sistema de ecuaciones y del factor de relajación elegido.

'''
st.subheader('Método :triangular_ruler:')


r = st.number_input('Ingrese el tamño de la matriz:', min_value=1, max_value=10,value=4)


cols = [str(i+1) for i in range(r)]
cols.append('x')
mat1 = pd.DataFrame(np.zeros((r,r+1)),columns=cols)
st.write('Ingrese la matriz:')
mat2 = st.data_editor(mat1, key='mat1')

x0 = st.text_input('Ingrese el valor inicial de $x_0$:',value=str(list(np.zeros(r))))
ome = st.text_input('Ingrese el valor de $\omega$:',value='1.2')
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
            solucion = gauss_seidel(np.array(m), np.array(b), float(ome),sp.parse_expr(x0),float(error),max_iter)

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
