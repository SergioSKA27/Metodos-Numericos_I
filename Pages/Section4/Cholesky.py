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

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1):
            if i == j:
                sum_sq = 0
                for k in range(j):
                    sum_sq += L[i, k] ** 2
                L[i, j] = np.sqrt(A[i, i] - sum_sq)
            else:
                sum_prod = 0
                for k in range(j):
                    sum_prod += L[i, k] * L[j, k]
                L[i, j] = (A[i, j] - sum_prod) / L[j, j]

    return L


st.header('4. Factorización LU y sus Aplicaciones')



r'''
# Resumen: Método de Cholesky

El método de Cholesky es un algoritmo utilizado para factorizar una matriz simétrica y definida positiva en el producto
de una matriz triangular inferior y su traspuesta conjugada, es decir, $A = LL^*$. Esta descomposición es conocida como
descomposición de Cholesky.

## Fundamentos Teóricos

Dada una matriz simétrica y definida positiva $A$, el método de Cholesky encuentra una matriz triangular inferior $L$
cuyos elementos se calculan de la siguiente manera:

$$
L_{ij} = \begin{cases}
      \sqrt{A_{ii} - \sum_{k=1}^{i-1}L_{ik}^2}, & \text{si } i = j \\
      \frac{1}{L_{jj}}\left(A_{ij} - \sum_{k=1}^{j-1}L_{ik}L_{jk}\right), & \text{si } i > j
   \end{cases}
$$

Donde $L_{ij}$ es el elemento en la fila $i$ y la columna $j$ de la matriz $L$, y $A_{ij}$ es el elemento
correspondiente en la matriz $A$.

## Aplicaciones del Método de Cholesky

El método de Cholesky encuentra aplicaciones en diversas áreas de la ciencia y la ingeniería. Algunas de sus
aplicaciones más comunes son:

1. **Resolución de sistemas de ecuaciones lineales**: Una vez que se ha obtenido la descomposición de Cholesky de una
matriz $A$, se puede utilizar para resolver eficientemente sistemas de ecuaciones lineales de la forma $Ax = b$.
La solución se obtiene realizando una sustitución hacia adelante y una sustitución hacia atrás utilizando la matriz
$L$ y su traspuesta conjugada $L^*$. Este método es especialmente útil cuando la matriz $A$ es simétrica y definida
positiva.

2. **Generación de números aleatorios correlacionados**:
El método de Cholesky se utiliza en la generación de números aleatorios correlacionados.
Dada una matriz de covarianza simétrica y definida positiva, la descomposición de Cholesky se aplica para obtener una
matriz triangular inferior $L$. Luego, utilizando una secuencia de números aleatorios independientes y una
transformación lineal, se generan números aleatorios correlacionados con la estructura de la matriz de covarianza original.

3. **Optimización y mínimos cuadrados**: En problemas de optimización y mínimos cuadrados, el método de Cholesky se
utiliza para resolver sistemas de ecuaciones normales. La descomposición de Cholesky permite expresar la matriz Hessiana
como $LL^*$, lo que facilita la solución del sistema lineal asociado de manera más eficiente y estable.

En resumen, el método de Cholesky es un algoritmo poderoso y eficiente para factorizar matrices simétricas y
definidas positivas. Su descomposición se utiliza en la resolución de sistemas de ecuaciones lineales, la generación de
números aleatorios correlacionados y la optimización numérica.


## Algoritmo

El método de Cholesky es un algoritmo utilizado para factorizar una matriz simétrica y definida positiva en el producto
de una matriz triangular inferior y su traspuesta conjugada.

**Entrada:** Una matriz simétrica y definida positiva $A$ de tamaño $n \times n$.

**Salida:** La matriz triangular inferior $L$ tal que $A = LL^*$.

1. Inicializar la matriz triangular inferior $L$ de tamaño $n \times n$ con ceros.

2. Para $i$ desde 1 hasta $n$, hacer:

   - Para $j$ desde 1 hasta $i$, hacer:

     - Si $i = j$, calcular $L_{ij} = \sqrt{A_{ii} - \sum_{k=1}^{i-1}L_{ik}^2}$.

     - Si $i > j$, calcular $L_{ij} = \frac{1}{L_{jj}}\left(A_{ij} - \sum_{k=1}^{j-1}L_{ik}L_{jk}\right)$.

3. Retornar la matriz triangular inferior $L$.

El algoritmo de Cholesky se basa en la propiedad de que una matriz simétrica y definida positiva puede ser factorizada
en el producto de una matriz triangular inferior y su traspuesta conjugada. Durante el proceso de factorización,
se calculan los elementos de la matriz triangular inferior $L$ utilizando las ecuaciones especificadas en el algoritmo.

Es importante destacar que el método de Cholesky solo es aplicable a matrices simétricas y definidas positivas.
Además, si se encuentra un elemento diagonal $L_{ii}$ igual a cero durante la factorización, significa que la matriz
$A$ no es definida positiva y, por lo tanto, no se puede aplicar el método de Cholesky.


## Ejemplo

Supongamos que tenemos la siguiente matriz simétrica y definida positiva:

$$ A = \begin{bmatrix} 4 & 2 & -1 \\ 2 & 5 & -4 \\ -1 & -4 & 6 \end{bmatrix} $$

Aplicaremos el método de Cholesky para obtener la factorización $A = LL^*$.

1. Inicialización de la matriz $L$ de tamaño $3 \times 3$ con ceros:

$$ L = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} $$

2. Para $i = 1$:

   - Para $j = 1$:

     - Calculamos $L_{11} = \sqrt{A_{11}} = \sqrt{4} = 2$.

   - Para $j = 2$:

     - Calculamos $L_{12} = \frac{1}{L_{11}}(A_{12}) = \frac{1}{2}(2) = 1$.

   - Para $j = 3$:

     - Calculamos $L_{13} = \frac{1}{L_{11}}(A_{13}) = \frac{1}{2}(-1) = -\frac{1}{2}$.

3. Para $i = 2$:

   - Para $j = 1$:

     - Calculamos $L_{21} = 0$ (ya que $L$ es triangular inferior).

   - Para $j = 2$:

     - Calculamos $L_{22} = \sqrt{A_{22} - L_{21}^2} = \sqrt{5 - 1^2} = \sqrt{4} = 2$.

   - Para $j = 3$:

     - Calculamos $L_{23} = \frac{1}{L_{22}}(A_{23} - L_{21}L_{31}) = \frac{1}{2}(-4 - 0) = -2$.

4. Para $i = 3$:

   - Para $j = 1$:

     - Calculamos $L_{31} = 0$ (ya que $L$ es triangular inferior).

   - Para $j = 2$:

     - Calculamos $L_{32} = 0$ (ya que $L$ es triangular inferior).

   - Para $j = 3$:

     - Calculamos $L_{33} = \sqrt{A_{33} - L_{31}^2 - L_{32}^2} = \sqrt{6 - 0 - 0} = \sqrt{6}$.

Por lo tanto, la matriz triangular inferior $L$ obtenida es:

$$ L = \begin{bmatrix} 2 & 0 & 0 \\ 1 & 2 & 0 \\ -\frac{1}{2} & -2 & \sqrt{6} \end{bmatrix} $$

Verificamos que $A = LL^*$:

$$ LL^* = \begin{bmatrix} 2 & 0 & 0 \\ 1 & 2 & 0 \\ -\frac{1}{2} & -2 & \sqrt{6} \end{bmatrix} \begin{bmatrix} 2 & 1 & -\frac{1}{2} \\ 0 & 2 & -2 \\ 0 & 0 & \sqrt{6} \end{bmatrix} = \begin{bmatrix} 4 & 2 & -1 \\ 2 & 5 & -4 \\ -1 & -4 & 6 \end{bmatrix} = A $$

Por lo tanto, se ha obtenido la factorización $A = LL^*$ utilizando el método de Cholesky.



'''



st.subheader('Método :triangular_ruler:')


r = st.number_input('Ingrese el tamño de la matriz:', min_value=1, max_value=10,value=4)


cols = [str(i+1) for i in range(r)]
mat1 = pd.DataFrame(np.zeros((r,r)),columns=cols)
st.write('Ingrese la matriz:')
mat2 = st.data_editor(mat1, key='mat1')


m = [[(sp.Matrix(mat2))[i,j] for j in range(r)] for i in range(r)]


st.latex('A = ' + sp.latex(sp.Matrix(m)))


if not(is_zero_matrix(m)):

    try:
        if sp.Matrix(m).det() == 0:
            st.write('La matriz no tiene solucion :(  **|A| = 0**')
        else:
            solucion = cholesky_decomposition(np.array(m).astype(float))
            st.write('Matriz triangular inferior L:')
            st.latex(r'''\mathbb{L} \approx ''' + sp.latex(sp.Matrix(solucion)))
            st.write('Matriz triangular superior U:')
            st.latex(r'''\mathbb{U} \approx ''' + sp.latex(sp.Matrix(solucion).transpose()))
            st.write('Comprobamos que $A = LU$')
            st.latex(sp.latex(sp.Matrix(np.round(solucion,decimals=2)))+' \cdot '+sp.latex(sp.Matrix(np.round(solucion,decimals=2)).transpose()) +' = '+sp.latex(sp.Matrix(solucion)*sp.Matrix(solucion).transpose()))

    except:
        if sp.Matrix(m).det() == 0:
            st.write('La matriz no tiene solucion :(')
        else:
            st.write('Algo salio mal :(')
