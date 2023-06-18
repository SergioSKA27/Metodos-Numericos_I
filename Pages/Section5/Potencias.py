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


def dot_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def vector_norm(v):
    result = 0
    for i in range(len(v)):
        result += v[i] ** 2
    return np.sqrt(result)

st.cache(max_entries=1000)
def power_method(A, x0, num_iterations):
    x = x0
    for i in range(num_iterations):
        y = np.zeros_like(x)
        for j in range(A.shape[0]):
            for k in range(A.shape[1]):
                y[j] += A[j, k] * x[k]
        eigenvalue = dot_product(y, x) / dot_product(x, x)
        norm_y = vector_norm(y)
        x = y / norm_y
    return eigenvalue, x


st.title('5. Cálculo de Valores y Vectores Propios')


r'''

## 5.1 Método de potencias

El Método de Potencias es un algoritmo utilizado para encontrar el valor propio dominante y su correspondiente vector
propio asociado de una matriz. Es particularmente útil en problemas de álgebra lineal y análisis numérico.

### Descripción del Método

El objetivo del Método de Potencias es encontrar el valor propio dominante $\lambda_1$ y el vector propio correspondiente
$\mathbf{v}_1$ de una matriz cuadrada $\mathbf{A}$. El valor propio dominante es aquel con la mayor magnitud, y
el vector propio asociado es el vector no nulo correspondiente a ese valor propio.

El algoritmo del Método de Potencias se puede describir en los siguientes pasos:

1. Dado un vector inicial no nulo $\mathbf{x}^{(0)}$, normalízalo: $\mathbf{x}^{(0)} = \frac{\mathbf{x}^{(0)}}{\|\mathbf{x}^{(0)}\|}$.
2. Para $k = 1, 2, 3, \ldots$:
   - Calcula el vector $\mathbf{y}^{(k)} = \mathbf{A} \mathbf{x}^{(k-1)}$.
   - Calcula el valor propio dominante estimado $\lambda^{(k)} = \frac{{\mathbf{y}^{(k)}}^\top \mathbf{x}^{(k-1)}}{{\mathbf{x}^{(k-1)}}^\top \mathbf{x}^{(k-1)}}$.
   - Normaliza el vector $\mathbf{x}^{(k)} = \frac{\mathbf{y}^{(k)}}{\|\mathbf{y}^{(k)}\|}$.

El algoritmo continúa iterando hasta que se alcance una condición de convergencia, como la estabilización del valor
propio estimado o la convergencia del vector propio estimado.
El Método de Potencias es un algoritmo eficaz para encontrar el valor propio dominante y el vector propio
correspondiente de una matriz. Su implementación es sencilla y converge rápidamente en la mayoría de los casos.
Sin embargo, es importante tener en cuenta los supuestos de aplicación del método para garantizar resultados correctos.


## Supuestos de aplicación

El Método de Potencias se aplica en los siguientes casos:

1. La matriz $\mathbf{A}$ es cuadrada y diagonalizable, es decir, puede ser factorizada como
$\mathbf{A} = \mathbf{P} \mathbf{D} \mathbf{P}^{-1}$, donde $\mathbf{P}$ es una matriz invertible y $\mathbf{D}$ es
una matriz diagonal.
2. El valor propio dominante $\lambda_1$ tiene una magnitud mayor que el resto de los valores propios.
Esto implica que la matriz $\mathbf{A}$ no tiene valores propios repetidos con la misma magnitud de $\lambda_1$.
3. El vector propio asociado $\mathbf{v}_1$ es linealmente independiente de los demás vectores propios de $\mathbf{A}$.
4. El vector inicial $\mathbf{x}^{(0)}$ no es ortogonal a $\mathbf{v}_1$, es decir,
$\mathbf{x}^{(0)} \neq c \mathbf{v}_1$, donde $c$ es una constante.

### Ejemplo

Consideremos la siguiente matriz $\mathbf{A}$:

$$
\mathbf{A} = \begin{bmatrix}
2 & -1 & 0 \\
-1 & 2 & -1 \\
0 & -1 & 2
\end{bmatrix}
$$

Deseamos encontrar el valor propio dominante y el vector propio correspondiente utilizando

 el Método de Potencias. Tomamos un vector inicial $\mathbf{x}^{(0)} = [1, 1, 1]^\top$ y aplicamos el algoritmo iterativo.

1. Normalizamos el vector inicial: $\mathbf{x}^{(0)} = \frac{\mathbf{x}^{(0)}}{\|\mathbf{x}^{(0)}\|} = \frac{1}{\sqrt{3}} [1, 1, 1]^\top$.

2. Iteración 1:
   - Calculamos el vector $\mathbf{y}^{(1)} = \mathbf{A} \mathbf{x}^{(0)} = \begin{bmatrix}
   1 \\
   0 \\
   -1
   \end{bmatrix}$.
   - Calculamos el valor propio dominante estimado $\lambda^{(1)} = \frac{{\mathbf{y}^{(1)}}^\top \mathbf{x}^{(0)}}{{\mathbf{x}^{(0)}}^\top \mathbf{x}^{(0)}} = \frac{0}{1} = 0$.
   - Normalizamos el vector $\mathbf{x}^{(1)} = \frac{\mathbf{y}^{(1)}}{\|\mathbf{y}^{(1)}\|} = \begin{bmatrix}
   \frac{1}{\sqrt{2}} \\
   0 \\
   -\frac{1}{\sqrt{2}}
   \end{bmatrix}$.

3. Iteración 2:
   - Calculamos el vector $\mathbf{y}^{(2)} = \mathbf{A} \mathbf{x}^{(1)} = \begin{bmatrix}
   \frac{1}{\sqrt{2}} \\
   0 \\
   -\frac{1}{\sqrt{2}}
   \end{bmatrix}$.
   - Calculamos el valor propio dominante estimado $\lambda^{(2)} = \frac{{\mathbf{y}^{(2)}}^\top \mathbf{x}^{(1)}}{{\mathbf{x}^{(1)}}^\top \mathbf{x}^{(1)}} = \frac{\frac{1}{\sqrt{2}}}{\frac{1}{\sqrt{2}}} = 1$.
   - Normalizamos el vector $\mathbf{x}^{(2)} = \frac{\mathbf{y}^{(2)}}{\|\mathbf{y}^{(2)}\|} = \begin{bmatrix}
   \frac{1}{\sqrt{2}} \\
   0 \\
   -\frac{1}{\sqrt{2}}
   \end{bmatrix}$.

Continuamos el proceso iterativo hasta alcanzar la convergencia del valor propio estimado y del vector propio estimado.


'''

st.subheader('Método :triangular_ruler:')


r = st.number_input('Ingrese el tamño de la matriz:', min_value=1, max_value=10,value=4)


cols = [str(i+1) for i in range(r)]

mat1 = pd.DataFrame(np.zeros((r,r)),columns=cols)
st.write('Ingrese la matriz:')
mat2 = st.data_editor(mat1, key='mat1')

x0 = st.text_input('Ingrese el valor inicial de $x_0$:',value=str(list(np.zeros(r))))


max_iter = st.number_input('Ingrese el número máximo de iteraciones:', min_value=1, max_value=10000,value=10)



m = [[(sp.Matrix(mat2))[i,j] for j in range(r)] for i in range(r)]


st.latex('A = ' + sp.latex(sp.Matrix(m)))

if st.button('Calcular'):
    if not(is_zero_matrix(m)):
        try:
            solucion = power_method(np.array(m),np.array(sp.parse_expr(x0)).astype(float),max_iter)
            st.write('Valor propio estimado:')
            st.latex(r'''x \approx ''' + sp.latex((solucion[0])))
            st.write('Vector  propio estimado:')
            st.latex(r'''\hat{x} \approx ''' + sp.latex(sp.Matrix(solucion[1])))
        except:
            st.error('Algo salio mal', icon="⚠️")


with echo_expander(code_location="below", label="Implementación en Python"):
    import numpy as np
    import sympy as sp
    def dot_product(a, b):
        result = 0
        for i in range(len(a)):
            result += a[i] * b[i]
        return result

    def vector_norm(v):
        result = 0
        for i in range(len(v)):
            result += v[i] ** 2
        return np.sqrt(result)

    def power_method(A, x0, num_iterations):
        x = x0
        for i in range(num_iterations):
            y = np.zeros_like(x)
            for j in range(A.shape[0]):
                for k in range(A.shape[1]):
                    y[j] += A[j, k] * x[k]
            eigenvalue = dot_product(y, x) / dot_product(x, x)
            norm_y = vector_norm(y)
            x = y / norm_y
        return eigenvalue, x


