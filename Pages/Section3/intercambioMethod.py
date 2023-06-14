import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math





#check if a matrix is zero matrix
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


def intercambio(A, b):
    """
    The function performs Gaussian elimination with partial pivoting to solve a system of linear equations.

    :param A: A is a numpy array representing the coefficient matrix of a system of linear equations. Each row of the matrix
    represents an equation, and each column represents a variable
    :param b: The parameter "b" is a numpy array representing the constants in a system of linear equations. It has to have
    the same number of rows as the matrix "A" representing the coefficients of the variables
    :return: a NumPy array containing the solutions to the system of linear equations represented by the input matrix A and
    vector b.
    """
    steps = []
    # Crear matriz aumentada
    matriz_aumentada = np.concatenate((A, b.reshape(-1, 1)), axis=1)
    n = len(matriz_aumentada)

    # Iterar por las filas
    for i in range(n):
        # Encontrar el pivote
        pivote = matriz_aumentada[i][i]

        # Si el pivote es cero, realizar intercambio con otra fila
        if pivote == 0:
            # Encontrar una fila con un elemento no nulo en la columna actual
            for j in range(i+1, n):
                if matriz_aumentada[j][i] != 0:
                    # Intercambiar filas
                    steps.append(f'R_{i} <---> R_{j}')
                    steps.append(sp.latex(sp.Matrix(matriz_aumentada)))
                    for k in range(n):
                        matriz_aumentada[j][k], matriz_aumentada[i][k] = matriz_aumentada[i][k], matriz_aumentada[j][k]
                    #matriz_aumentada[[i, j]] = matriz_aumentada[[j, i]]
                    break

        # Aplicar eliminación gaussiana
        for j in range(i+1, n):
            factor = matriz_aumentada[j][i] / pivote
            matriz_aumentada[j] -= factor * matriz_aumentada[i]
            if factor != 0:
                steps.append(str(-1*factor)+'*R_'+str(i)+' + R_'+str(j)+' <---> R_'+str(j))
        steps.append(sp.latex(sp.Matrix(matriz_aumentada)))


    # Realizar sustitución hacia atrás para obtener las soluciones
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = matriz_aumentada[i][-1]
        for j in range(i+1, n):
            x[i] -= matriz_aumentada[i][j] * x[j]
        x[i] /= matriz_aumentada[i][i]

    return x,steps,matriz_aumentada




st.title('3. Solución de Sistemas de Ecuaciones Lineales')


r'''
## 3.1.2 Método de intercambio

El método de intercambio es una técnica rigurosa utilizada en el álgebra lineal para resolver sistemas de ecuaciones
lineales mediante la eliminación de variables. Este método se basa en la manipulación matemática precisa de la matriz
aumentada del sistema para obtener una solución consistente.

1. Procedimiento del método de intercambio:
   - Se comienza con una matriz aumentada $[A | \mathbf{b}]$ de tamaño $m \times (n+1)$, donde $A$ es una matriz
   de coeficientes de tamaño $m \times n$ y $\mathbf{b}$ es el vector de términos constantes de tamaño $m$.
   - Se selecciona una fila $i$ (o columna) que contenga un elemento no nulo en la columna $k$ (llamado el pivote)
   para realizar un intercambio con otra fila (o columna) con el objetivo de asegurar que el pivote sea distinto de cero.
   - Si el pivote es cero, se realiza un intercambio con una fila (o columna) diferente que tenga un elemento no nulo
   en la columna $k$.
   - Luego, se realiza la eliminación gaussiana o la eliminación gauss-jordan para reducir la matriz aumentada a una
   forma escalonada o escalonada reducida.
   - Finalmente, se obtienen las soluciones del sistema mediante sustitución hacia atrás o mediante la lectura directa
   de la matriz escalonada reducida.

2. Importancia del método de intercambio:
   - El método de intercambio es esencial para resolver sistemas de ecuaciones lineales cuando la matriz presenta
   estructuras o patrones especiales, como filas (o columnas) de ceros o coeficientes muy pequeños, que dificultan
   la resolución directa.
   - Permite reorganizar la matriz aumentada de manera sistemática, garantizando la existencia de pivotes no nulos
   y simplificando el proceso de eliminación.

3. Limitaciones del método de intercambio:
   - El método de intercambio puede ser computacionalmente costoso en sistemas grandes, ya que puede requerir varios
   intercambios de filas (o columnas) para asegurar pivotes no nulos.
   - Es fundamental seleccionar adecuadamente las filas (o columnas) de intercambio para evitar la introducción de
   errores y asegurar una resolución correcta y precisa del sistema.

El método de intercambio proporciona un enfoque riguroso y matemáticamente sólido para resolver sistemas de
ecuaciones lineales que presentan desafíos estructurales. Al seguir este procedimiento cuidadosamente,
se puede obtener una solución consistente y precisa para el sistema.



## Ejemplo
Consideremos el siguiente sistema de ecuaciones lineales:

$$
\begin{align*}
2x - 3y &= 1 \\
4x + y &= -2 \\
\end{align*}
$$

Podemos representar este sistema en forma matricial como $A\mathbf{x} = \mathbf{b}$, donde:

$$
A = \begin{bmatrix}
2 & -3 \\
4 & 1 \\
\end{bmatrix}, \quad
\mathbf{x} = \begin{bmatrix}
x \\
y \\
\end{bmatrix}, \quad
\mathbf{b} = \begin{bmatrix}
1 \\
-2 \\
\end{bmatrix}
$$

Para resolver este sistema utilizando el método de intercambio, primero creamos la matriz aumentada $[A | \mathbf{b}]$:

$$
\begin{bmatrix}
2 & -3 & | & 1 \\
4 & 1 & | & -2 \\
\end{bmatrix}
$$

Observamos que el pivote en la columna 1 es 2. Como queremos un pivote no nulo, intercambiamos la primera fila con la segunda fila:

$$
\begin{bmatrix}
4 & 1 & | & -2 \\
2 & -3 & | & 1 \\
\end{bmatrix}
$$

Luego, aplicamos la eliminación gaussiana para reducir la matriz aumentada a una forma escalonada:

$$
\begin{bmatrix}
4 & 1 & | & -2 \\
0 & -5 & | & 5 \\
\end{bmatrix}
$$

El siguiente paso es asegurar que el pivote en la columna 2 sea no nulo. Como ya lo tenemos en la fila 2, no es necesario realizar intercambios adicionales.

Continuamos con la eliminación gaussiana:

$$
\begin{bmatrix}
4 & 1 & | & -2 \\
0 & 1 & | & -1 \\
\end{bmatrix}
$$

Por último, aplicamos la sustitución hacia atrás para obtener las soluciones del sistema. A partir de la última fila, podemos ver que $y = -1$. Sustituyendo este valor en la primera fila, obtenemos $4x + 1 = -2$, lo que implica $x = -\frac{3}{4}$.

Por lo tanto, la solución del sistema de ecuaciones es $x = -\frac{3}{4}$ y $y = -1$.

El método de intercambio nos permitió reorganizar la matriz aumentada y realizar eliminaciones para obtener la solución del sistema. Es importante destacar que los intercambios de filas son cruciales para asegurar pivotes no nulos y evitar divisiones por cero durante el proceso de eliminación.

'''







st.subheader('Calculadora de Matriz Inversa :triangular_ruler:')


r = st.number_input('Ingrese el tamño de la matriz:', min_value=1, max_value=10,value=4)


cols = [str(i+1) for i in range(r)]
cols.append('x')
mat1 = pd.DataFrame(np.zeros((r,r+1)),columns=cols)
st.write('Ingrese la matriz:')
mat2 = st.data_editor(mat1, key='mat1')






m = [[(sp.Matrix(mat2))[i,j] for j in range(r)] for i in range(r)]
b = [(sp.Matrix(mat2))[i,-1] for i in range(r)]

st.latex('A = ' + sp.latex(sp.Matrix(m))+ ' , b = '+sp.latex(sp.Matrix(b)))
r'# Método de intercambio'

if not(is_zero_matrix(m)):
    try:
        if sp.Matrix(m).det() == 0:
            st.write('La matriz no tiene solucion :(')
        else:
            solucion = intercambio(np.array(m), np.array(b))
            st.write('Matriz escalonada:')
            st.latex(sp.latex(sp.Matrix(solucion[2])))
            st.write('Solucion:')
            st.latex('\hat{x} = ' + sp.latex(sp.Matrix(solucion[0])))
            st.write('Pasos realizados:')
            for t in solucion[1]:
                st.latex(t)
    except:
        if sp.Matrix(m).det() == 0:
            st.write('La matriz no tiene solucion :(')
        else:
            st.write('Algo salio mal :(')

