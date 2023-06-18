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
def doolittle(A):
  """
  The function implements the Doolittle algorithm for LU decomposition of a given matrix.

  :param A: A is a square matrix of size n x n, which represents the system of linear equations to be solved using the
  Doolittle decomposition method
  :return: The function `doolittle` returns two matrices `L` and `U`, which are the lower and upper triangular matrices
  obtained from the Doolittle decomposition of the input matrix `A`.
  """
  n = len(A)
  L = np.eye(n)
  U = np.copy(A)

  for j in range(n):
    # Eliminación gaussiana en la columna j
    for i in range(j+1, n):
          m = U[i, j] / U[j, j]
          L[i, j] = m
          U[i, j:] -= m * U[j, j:]

  return L, U


st.header('4. Factorización LU y sus Aplicaciones')



r'''
# 4.3 Método Doolittle

El método Doolittle es un algoritmo utilizado para la factorización LU de una matriz cuadrada $A$. Esta factorización
descompone la matriz $A$ en el producto de dos matrices, una matriz triangular inferior $L$ y una matriz triangular
superior $U$. La factorización LU es una herramienta importante en el ámbito de la resolución numérica de sistemas
de ecuaciones lineales, ya que permite resolver sistemas de ecuaciones de forma más eficiente.

## Descripción del algoritmo

El algoritmo del método Doolittle se realiza en forma de eliminación gaussiana con pivoteo parcial. Comienza con una
matriz $A$ y crea una matriz $L$ inicialmente como una matriz identidad y una matriz $U$ inicialmente como una copia
de la matriz $A$. A continuación, se aplican pasos de eliminación gaussiana para triangular la matriz $U$ mientras se
actualiza la matriz $L$. El objetivo es transformar la matriz $A$ en una forma triangular superior $U$ y obtener la
matriz triangular inferior $L$ en el proceso.

El algoritmo se puede describir en los siguientes pasos:

1. Inicializar $L$ como una matriz identidad y $U$ como una copia de $A$.
2. Para cada columna $j$ de la matriz $U$, realizar los siguientes subpasos:
   - Encontrar el pivote, $p_{ij}$, que es el elemento de valor absoluto máximo en la columna $j$ a partir de la fila $j$.
   - Intercambiar las filas para asegurar que el pivote esté en la fila $j$.
   - Actualizar la matriz $U$ realizando la eliminación gaussiana en las filas siguientes, utilizando el elemento pivote
   $p_{ij}$ para calcular los multiplicadores.
   - Almacenar los multiplicadores en la matriz $L$ en la posición correspondiente, es decir, $L_{ij} = \frac{a_{ij}}{p_{ij}}$.
3. La matriz $L$ resultante es la matriz triangular inferior y la matriz $U$ resultante es la matriz triangular superior.

## Supuestos de aplicabilidad

El método Doolittle es aplicable a matrices cuadradas $A$ que son no singulares, es decir,
aquellas que tienen un determinante no nulo ($det(A) \neq 0$). Además, se asume que la matriz $A$
se puede triangular mediante eliminación gaussiana sin encontrar ceros en la diagonal principal durante el proceso.

## Ventajas y aplicaciones

El método Doolittle presenta varias ventajas y aplicaciones:

- Permite resolver sistemas de ecuaciones lineales de manera eficiente utilizando la factorización LU.
- Una vez que se ha realizado la factorización LU, es posible resolver múltiples sistemas de ecuaciones
con la misma matriz $A$, lo que ahorra tiempo computacional.
- La factorización LU también es útil en la resolución de matrices inversas y cálculo de determinantes.

En resumen, el método Doolittle es un algoritmo importante para la factorización LU de matrices y tiene diversas
aplicaciones en la resolución de sistemas de ecuaciones lineales y otros problemas numéricos. Su implementación
proporciona una descomposición útil de la matriz original en dos matrices triangulares, lo que facilita el cálculo
eficiente de soluciones y otras operaciones matriciales.

## Algoritmo

El algoritmo del método de Doolittle se puede describir de la siguiente manera:

**Entrada**: Una matriz cuadrada $A$ de tamaño $n \times n$.

**Salida**: Las matrices $L$ y $U$ que representan la factorización LU de la matriz $A$.

1. Inicializar la matriz $L$ como una matriz identidad de tamaño $n \times n$.
2. Inicializar la matriz $U$ como una copia de la matriz $A$.
3. Para cada columna $j = 1$ hasta $n$ hacer:
   - Para cada fila $i = j+1$ hasta $n$ hacer:
     - Calcular el multiplicador $m = \frac{U_{ij}}{U_{jj}}$.
     - Actualizar la matriz $U$ restando la fila $j$ multiplicada por el multiplicador $m$ de la fila $i$:
       $$U_{ij} = U_{ij} - m \cdot U_{jj}$$
     - Almacenar el multiplicador $m$ en la matriz $L$:
       $$L_{ij} = m$$
4. Devolver las matrices $L$ y $U$.

El algoritmo de Doolittle utiliza eliminación gaussiana con pivoteo parcial para triangular la matriz $U$ y calcular los multiplicadores que se almacenan en la matriz $L$. Al final del algoritmo, se obtienen las matrices $L$ y $U$ que representan la factorización LU de la matriz $A$.

Es importante destacar que este algoritmo asume que la matriz $A$ es no singular (es decir, tiene un determinante no nulo) y que se puede triangular sin encontrar ceros en la diagonal principal durante el proceso de eliminación gaussiana.


## Ejemplo

Aquí tienes un ejemplo del método de Doolittle aplicado a una matriz $A$ de tamaño $3 \times 3$:

**Ejemplo**:

Consideremos la siguiente matriz $A$:
$$ A = \begin{bmatrix} 2 & -1 & 3 \\ 4 & 2 & -1 \\ 1 & 3 & 1 \end{bmatrix} $$

Aplicaremos el método de Doolittle para obtener la factorización LU de $A$.

**Paso 1**: Inicialización de $L$ y $U$:
$$ L = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad U = \begin{bmatrix} 2 & -1 & 3 \\ 4 & 2 & -1 \\ 1 & 3 & 1 \end{bmatrix} $$

**Paso 2**: Aplicación de eliminación gaussiana para triangular $U$:

Para la columna 1:

- Encontramos el pivote $p_{11} = 2$ (elemento máximo en la columna 1).
- Intercambiamos la fila 1 con la fila 2:
  $$ U = \begin{bmatrix} 4 & 2 & -1 \\ 2 & -1 & 3 \\ 1 & 3 & 1 \end{bmatrix} $$
- Calculamos el multiplicador $m_2 = \frac{2}{4} = \frac{1}{2}$ y actualizamos la matriz $U$ y la matriz $L$:
  $$ U = \begin{bmatrix} 4 & 2 & -1 \\ 0 & -2 & \frac{7}{2} \\ 1 & 3 & 1 \end{bmatrix}, \quad L = \begin{bmatrix} 1 & 0 & 0 \\ \frac{1}{2} & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} $$

Para la columna 2:

- Encontramos el pivote $p_{22} = -2$ (elemento máximo en la columna 2).
- No es necesario intercambiar filas, ya que el pivote ya está en la fila 2.
- Calculamos el multiplicador $m_3 = \frac{1}{-2} = -\frac{1}{2}$ y actualizamos la matriz $U$ y la matriz $L$:
  $$ U = \begin{bmatrix} 4 & 2 & -1 \\ 0 & -2 & \frac{7}{2} \\ 0 & 2 & \frac{3}{2} \end{bmatrix}, \quad L = \begin{bmatrix} 1 & 0 & 0 \\ \frac{1}{2} & 1 & 0 \\ 0 & -\frac{1}{2} & 1 \end{bmatrix} $$

**Paso 3**: Resultado final:

Obtenemos la factorización LU de la matriz $A$:
$$ L = \begin{bmatrix} 1 & 0 & 0 \\ \frac{1}{2} & 1 & 0 \\ 0 & -\frac{1}{2} & 1 \end{bmatrix}, \quad U = \begin{bmatrix} 4 & 2 & -1 \\ 0 & -2 & \frac{7}{2} \\ 0 & 2 & \frac{3}{2} \end{bmatrix} $$

En este ejemplo, hemos obtenido la matriz $L$ y $U$ que representan la factorización LU de la matriz $A$ utilizando
el método de Doolittle.

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
              solucion = doolittle(np.array(m).astype(float))
              st.write('Matriz triangular inferior L:')
              st.latex(r'''\mathbb{L} \approx ''' + sp.latex(sp.Matrix(np.round(solucion[0],decimals=6))) )
              st.write('Matriz triangular superior U:')
              st.latex(r'''\mathbb{U} \approx ''' + sp.latex(sp.Matrix(np.round(solucion[1],decimals=6))) )

      except:
          if sp.Matrix(m).det() == 0:
              st.write('La matriz no tiene solucion :(')
          else:
              st.write('Algo salio mal :(')




with echo_expander(code_location="below", label="Implementación en Python"):
    import numpy as np
    import sympy as sp
    def doolittle(A):
      n = len(A)
      L = np.eye(n)
      U = np.copy(A)

      for j in range(n):
          # Eliminación gaussiana en la columna j
          for i in range(j+1, n):
              m = U[i, j] / U[j, j]
              L[i, j] = m
              U[i, j:] -= m * U[j, j:]

      return L, U
