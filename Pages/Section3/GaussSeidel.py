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
def gauss_seidel(A, b, x0, max_iter, tol):
    n = len(A)
    x = np.copy(x0)
    steps = [[i for i in x0]]
    steps[0].append(0)
    for _ in range(max_iter):
        x_prev = np.copy(x)

        for i in range(n):
            ss = 0
            for j in range(n):
                if j != i:
                    ss += A[i][j] * x[j]

            x[i] = (b[i] - ss) / A[i][i]

        xapp = list(x.copy())
        xapp.append(np.linalg.norm(np.array(x).astype(float) - np.array(x_prev).astype(float)))
        xapp.append(str(np.linalg.norm(np.array(x).astype(float) - np.array(x_prev).astype(float)) < tol))

        steps.append(xapp)
        x_prev = x.copy()

        if np.linalg.norm(np.array(x).astype(float) - np.array(x_prev).astype(float)) < tol:
            break

    return x,steps




st.title('3. Solución de Sistemas de Ecuaciones Lineales')

r'''
# 3.4.3 Método de Gauss-Seidel

El Método de Gauss-Seidel es un algoritmo iterativo utilizado para resolver sistemas de ecuaciones lineales.
A diferencia del Método de Jacobi, el Método de Gauss-Seidel actualiza los valores de la solución a medida que avanza
en las iteraciones, lo que lo convierte en un método más eficiente en términos de convergencia.

El proceso del Método de Gauss-Seidel se basa en la descomposición de la matriz de coeficientes en una parte inferior
triangular ($L$) y una parte superior triangular ($U$). La matriz $A$ se puede expresar como $A = L + U$, donde $L$
contiene los elementos debajo de la diagonal principal y $U$ contiene los elementos encima de la diagonal principal.

El algoritmo del Método de Gauss-Seidel se puede resumir en los siguientes pasos:

1. Inicializar una aproximación inicial para la solución: $\mathbf{x}^{(0)}$.
2. Para cada ecuación $i$ en el sistema, calcular el lado derecho utilizando los valores más recientes de la solución:
$$
b^{(k)}_i = b_i - \sum_{j=1}^{i-1} a_{ij}x^{(k)}_j - \sum_{j=i+1}^{n} a_{ij}x^{(k-1)}_j.
$$
3. Utilizar los nuevos valores del lado derecho para calcular los nuevos valores de la solución:
$$
x^{(k)}_i = \frac{1}{a_{ii}}\left(b^{(k)}_i - \sum_{j=1}^{i-1} a_{ij}x^{(k)}_j - \sum_{j=i+1}^{n} a_{ij}x^{(k-1)}_j\right).
$$
4. Repetir los pasos 2 y 3 hasta que se cumpla un criterio de convergencia predefinido o se alcance un número máximo de iteraciones.

El criterio de convergencia comúnmente utilizado es la norma del vector diferencia entre dos iteraciones consecutivas,
es decir, si $\|\mathbf{x}^{(k)} - \mathbf{x}^{(k-1)}\| < \text{tolerancia}$. Además, se establece un número máximo de
iteraciones para evitar que el algoritmo entre en un bucle infinito.

El Método de Gauss-Seidel es especialmente efectivo cuando la matriz de coeficientes es diagonalmente dominante o cuando
la matriz se puede descomponer en una matriz triangular inferior y una matriz triangular superior. Sin embargo, en casos
donde la matriz no cumple estas condiciones, el método puede tener dificultades para converger o puede requerir un número
mayor de iteraciones.

En resumen, el Método de Gauss-Seidel es un método iterativo poderoso y eficiente para resolver sistemas de ecuaciones
lineales. Su capacidad para actualizar los valores de la solución en cada paso lo convierte en una opción atractiva en
situaciones donde se busca una convergencia más rápida que la proporcionada por el Método de Jacobi.

## Ejemplo

Consideremos el siguiente sistema de ecuaciones lineales:

$$
\begin{align*}
3x_1 + x_2 - x_3 &= 9 \\
2x_1 - 4x_2 + 2x_3 &= -1 \\
x_1 + x_2 + 3x_3 &= 6 \\
\end{align*}
$$

Podemos representar este sistema en forma matricial como $A\mathbf{x} = \mathbf{b}$, donde:

$$
A = \begin{bmatrix}
3 & 1 & -1 \\
2 & -4 & 2 \\
1 & 1 & 3 \\
\end{bmatrix}, \quad
\mathbf{x} = \begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\end{bmatrix}, \quad
\mathbf{b} = \begin{bmatrix}
9 \\
-1 \\
6 \\
\end{bmatrix}
$$

Para resolver este sistema utilizando el Método de Gauss-Seidel, primero descomponemos la matriz $A$ en una parte inferior triangular ($L$) y una parte superior triangular ($U$):

$$
L = \begin{bmatrix}
0 & 0 & 0 \\
-2 & 0 & 0 \\
-1 & -1 & 0 \\
\end{bmatrix}, \quad
U = \begin{bmatrix}
0 & -1 & 1 \\
0 & 0 & 2 \\
0 & 0 & 0 \\
\end{bmatrix}
$$

Luego, podemos reescribir el sistema de ecuaciones en forma iterativa utilizando la descomposición de Gauss-Seidel:

$$
\begin{align*}
x_1^{(k+1)} &= \frac{1}{3} \left(9 - x_2^{(k)} + x_3^{(k)}\right) \\
x_2^{(k+1)} &= \frac{1}{-4} \left(-1 - 2x_1^{(k+1)} + 2x_3^{(k)}\right) \\
x_3^{(k+1)} &= \frac{1}{3} \left(6 - x_1^{(k+1)} - x_2^{(k+1)}\right)
\end{align*}
$$

Donde $k$ es el número de iteración y $\mathbf{x}^{(k)}$ es el vector de solución en la iteración $k$.

Supongamos una aproximación inicial $\mathbf{x}^{(0)} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$ y apliquemos el Método de Gauss-Seidel:

**Iteración 1:**
$$
\begin{align*}
x_1^{(1)} &= \frac{1}{3} \left(9 - 0 + 0\right) = 3 \\
x_2^{(1)} &= \frac{1}{-4} \left(-1 - 2(3) + 2(0)\right) = -1.75 \\
x_3^{(1)} &= \frac{1}{3} \left(6 - 3 - (-1.75)\right) = 1.25
\end{align*}
$$

**Iteración 2:**
$$
\begin{align*}
x_1

^{(2)} &= \frac{1}{3} \left(9 - (-1.75) + 1.25\right) = 2.25 \\
x_2^{(2)} &= \frac{1}{-4} \left(-1 - 2(2.25) + 2(1.25)\right) = -1.53125 \\
x_3^{(2)} &= \frac{1}{3} \left(6 - 2.25 - (-1.53125)\right) = 1.03125
\end{align*}
$$

Continuamos iterando hasta alcanzar una convergencia deseada o un número máximo de iteraciones.

En este ejemplo, hemos aplicado el Método de Gauss-Seidel para resolver un sistema de ecuaciones lineales.
A medida que avanzamos en las iteraciones, los valores de la solución se actualizan y se acercan a la solución
exacta del sistema. Este método se utiliza ampliamente en aplicaciones numéricas donde la solución exacta es
difícil de obtener o computacionalmente costosa.

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

if st.button('Calcular'):
    if not(is_zero_matrix(m)):

        try:
            if sp.Matrix(m).det() == 0:
                st.write('La matriz no tiene solucion :(  **|A| = 0**')
            else:
                solucion = gauss_seidel(np.array(m), np.array(b),sp.parse_expr(x0),max_iter,float(error))
                st.write('Solucion aproximada:')
                st.latex(r'''\hat{x} \approx ''' + sp.latex(sp.Matrix(solucion[0])))
                sol = sp.Matrix(m).inv() * sp.Matrix(b)
                st.write('Error con respecto a la solucion:')
                st.latex(' \hat{x} = ' + sp.latex(sp.Matrix(sol)))
                st.latex('error = ' + sp.latex(abs(sol-sp.Matrix(solucion[0]))))
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



with echo_expander(code_location="below", label="Implementación en Python"):
    import numpy as np
    import sympy as sp
    def gauss_seidel(A, b, x0, max_iter, tol):
        n = len(A)
        x = np.copy(x0)
        for _ in range(max_iter):
            x_prev = np.copy(x)

            for i in range(n):
                ss = 0
                for j in range(n):
                    if j != i:
                        ss += A[i][j] * x[j]

                x[i] = (b[i] - ss) / A[i][i]

            x_prev = x.copy()

            if np.linalg.norm(np.array(x).astype(float) - np.array(x_prev).astype(float)) < tol:
                break

        return x,steps
