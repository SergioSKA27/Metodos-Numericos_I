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
def gauss_partitioned(A, b):
    """
    The function `gauss_partitioned` performs Gaussian elimination on a partitioned matrix and returns the solution vector
    and a list of steps taken during the elimination process.

    :param A: The matrix A is a square matrix of coefficients in a system of linear equations
    :param b: The parameter b is a numpy array representing the right-hand side of a system of linear equations. It contains
    the constant terms of the equations
    :return: The function `gauss_partitioned` returns a tuple containing the solution to the system of linear equations
    represented by the input matrices `A` and `b`, a list of steps taken to solve the system using Gaussian elimination with
    partial pivoting, and the identity matrix of the same size as `A`.
    """
    n = A.shape[0]

    steps = []

    b1  = b[:n//2]
    b2 = b[n//2:]

    # Descomposición de la matriz A
    A11 = A[:n//2, :n//2]
    A12 = A[:n//2, n//2:]
    A21 = A[n//2:, :n//2]
    A22 = A[n//2:, n//2:]


    steps.append('A_{11} = ' + sp.latex(sp.Matrix(A11)))
    steps.append('A_{12} = ' + sp.latex(sp.Matrix(A12)))
    steps.append('A_{21} = ' + sp.latex(sp.Matrix(A21)))
    steps.append('A_{22} = ' + sp.latex(sp.Matrix(A22)))
    steps.append('b_{1} = ' + sp.latex(sp.Matrix(b1)))
    steps.append('b_{2} = ' + sp.latex(sp.Matrix(b2)))


    A11_inv = np.linalg.inv(A11.astype(float))
    #print('Ainv : ', A11_inv)
    steps.append('A_{11}^{-1} = ' + sp.latex(sp.Matrix(A11_inv)))
    A12_p =np.matmul(A11_inv, A12)
    #print('A12* : ',A12_p )
    steps.append('A_{12}^{*} =  A_{11}^{-1} \cdot A_{12} = '+ sp.latex(sp.Matrix(A12_p)))
    b1_p = A11_inv @ b1
    #print('b1* : ',b1_p )
    steps.append('b_{1}^{*} = A_{11}^{-1} \cdot b_{1} = '+ sp.latex(sp.Matrix(b1_p)))

    A22_p = A22 - (np.matmul(A21, A12_p))
    #print('A22* : ',A22_p )
    steps.append('A_{22}^{*} = A_{22} - A_{21} \cdot A_{12}^{*} = '+ sp.latex(sp.Matrix(A22_p)))
    b2_p = b2- (np.matmul(A21 , b1_p))
    #print('b2* : ',b2_p )
    steps.append('b_{2}^{*} = A_{21} \cdot b_{1}^{*} = '+sp.latex(sp.Matrix(b2_p)) )

    A22_inv = np.linalg.inv(A22_p.astype(float))
    #print('A22_inv : ',A22_inv )
    steps.append('A_{22}^{-1*} = '+sp.latex(sp.Matrix(A22_inv)))
    b2_pp = A22_inv @ b2_p
    #print('b2** : ',b2_pp )
    steps.append('b_{2}^{**} = A_{22}^{-1*} \cdot b_{2}^{*} = ' + sp.latex(sp.Matrix(b2_pp)))
    b1_pp = b1_p- (np.matmul(A12_p , b2_pp))
    #print('b1** : ',b1_pp )
    steps.append('b_{1}^{**} = b_{1}^{*} - A_{12}^{*} \cdot b_{2}^{**} = '+sp.latex(sp.Matrix(b1_pp)))







    return np.concatenate((b1_pp,b2_pp)),steps,np.eye(n)




st.title('3. Solución de Sistemas de Ecuaciones Lineales')


r'''
# 3.3.3 Gauss-Jordan particionado.

Este método nos ayuda a encontrar la solución de un sistema de ecuaciones con un
número muy grande de variables, utilizando el método de Gass-Jordan pero con matrices
en vez de valores. De igual manera que el método anterior particionamos la matriz de la
siguiente manera:

$$
\begin{bmatrix}
    \begin{array}{c|c}
        A_{11}  & A_{12} \\
        \hline
        A_{21}  & A_{22} \\
    \end{array}
\end{bmatrix}
\begin{bmatrix}
    \begin{array}{c}
        x_{1} \\
        x_{2} \\
    \end{array}
\end{bmatrix} =
\begin{bmatrix}
    \begin{array}{c}
        b_{1} \\
        b_{2} \\
    \end{array}
\end{bmatrix}
$$

Hay que transformar la matriz de la siguiente manera:

$$
\begin{bmatrix}
    \begin{array}{c|c}
        I  & A_{12}' \\
        \hline
        A_{21}  & A_{22} \\
    \end{array}
\end{bmatrix}
\begin{bmatrix}
    \begin{array}{c}
        x_{1} \\
        x_{2} \\
    \end{array}
\end{bmatrix} =
\begin{bmatrix}
    \begin{array}{c}
        b_{1}' \\
        b_{2} \\
    \end{array}
\end{bmatrix}
$$


$A_{11}$ es el pivote, si multiplicamos todos los elementos del primer renglón de la matriz
por la inversa de $A_{11}^{-1}$ y nos queda lo siguiente:


$$
\begin{align*}
    A_{12}' = A_{11}^{-1} \cdot A_{12} \\
    b_1' = A_{11}^{-1} \cdot b_1
\end{align*}
$$

Lo siguiente es hacer ceros la matriz  $A_{21}$ , para transformar la matriz en:
$$
\begin{bmatrix}
    \begin{array}{c|c}
        I  & A_{12}' \\
        \hline
        0  & A_{22}' \\
    \end{array}
\end{bmatrix}
\begin{bmatrix}
    \begin{array}{c}
        x_{1} \\
        x_{2} \\
    \end{array}
\end{bmatrix} =
\begin{bmatrix}
    \begin{array}{c}
        b_{1}' \\
        b_{2}' \\
    \end{array}
\end{bmatrix}
$$

donde:

$$
\begin{align*}

    A_{22}' = A_{22} - A_{21} \cdot A_{12}' \\
    b_2' = b_2- A_{21} \cdot b_{1}'

\end{align*}
$$


El siguiente paso es hacer unos la matriz $A_{22}' = I$, multiplicando los elementos del segundo
reglón por $A_{22}'^{-1}$, la inversa del nuevo pivote para obtener el siguiente sistema:


$$
\begin{bmatrix}
    \begin{array}{c|c}
        I  & A_{12}' \\
        \hline
        0  & I \\
    \end{array}
\end{bmatrix}
\begin{bmatrix}
    \begin{array}{c}
        x_{1} \\
        x_{2} \\
    \end{array}
\end{bmatrix} =
\begin{bmatrix}
    \begin{array}{c}
        b_{1}' \\
        b_{2}'' \\
    \end{array}
\end{bmatrix}
$$

donde:

$$
\begin{align*}
    b_{2}'' = A_{22}'^{-1} \cdot b_{2}'
\end{align*}
$$

Finalmente, hay que hacer $A_{12}' = [0]$, para obtener el siguiente sistema:


$$
\begin{bmatrix}
    \begin{array}{c|c}
        I  & 0 \\
        \hline
        0  & I \\
    \end{array}
\end{bmatrix}
\begin{bmatrix}
    \begin{array}{c}
        x_{1} \\
        x_{2} \\
    \end{array}
\end{bmatrix} =
\begin{bmatrix}
    \begin{array}{c}
        b_{1}'' \\
        b_{2}'' \\
    \end{array}
\end{bmatrix}
$$


donde:

$$
\begin{align*}

    b_{1}'' = b_{1}' - A_{12}' \cdot b_{2}''

\end{align*}
$$

La solución al sistema es:

$$
X =
\begin{bmatrix}
    \begin{array}{c}
        b_{1}'' \\
        b_{2}'' \\
    \end{array}
\end{bmatrix}
$$






## Ejemplo


Consideremos el siguiente sistema de ecuaciones lineales:

$$
\begin{align*}
2x + 3y + 4z &= 9 \\
5x + 6y + 7z &= 11 \\
8x + 9y + 10z &= 13
\end{align*}
$$

Aplicaremos el método de Gauss particionado para resolver este sistema.

### Paso 1: División de la matriz de coeficientes y el vector de términos independientes

Dividimos la matriz de coeficientes **A** en submatrices de igual tamaño:

$$
A = \begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix} =
\begin{bmatrix}
2 & 3 \\
5 & 6
\end{bmatrix}
$$

Dividimos el vector de términos independientes **b** en subvectores correspondientes:

$$
b = \begin{bmatrix}
b_1 \\
b_2
\end{bmatrix} =
\begin{bmatrix}
4 \\
7
\end{bmatrix}
$$

### Paso 2: Resolución de la submatriz $A_{11}$

Resolvemos la submatriz $A_{11}$ utilizando un método de eliminación de Gauss estándar para obtener la matriz
triangular superior $U_{11}$:

$$
U_{11} = \begin{bmatrix}
2 & 3 \\
0 & -0.6
\end{bmatrix}
$$

### Paso 3: Resolución del sistema $U_{11}x_1 = b_1$

Utilizamos la matriz triangular superior $U_{11}$ para resolver el sistema $U_{11}x_1 = b_1$ y
obtener el vector solución $x_1$:

$$
x_1 = \begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
0.8 \\
2
\end{bmatrix}
$$

### Paso 4: Cálculo del vector residual $r_2$

Calculamos el vector residual $r_2 = b_2 - A_{21}x_1$:

$$
r_2 = \begin{bmatrix}
7
\end{bmatrix} -
\begin{bmatrix}
5 & 6
\end{bmatrix} \begin{bmatrix}
0.8 \\
2
\end{bmatrix} =
\begin{bmatrix}
-2
\end{bmatrix}
$$

### Paso 5: Resolución del sistema $A_{22}x_2 = r_2$

Resolvemos el sistema $A_{22}x_2 = r_2$ utilizando un método de eliminación de Gauss estándar y obtenemos el vector
solución $x_2$:

$$
x_2 = \begin{bmatrix}
z
\end{bmatrix} =
\begin{bmatrix}
1
\end{bmatrix}
$$

### Paso 6: Obtención de la solución del sistema original

Combinamos las soluciones obtenidas en los pasos anteriores para obtener la solución del sistema original:

$$
x = \begin{bmatrix}
x \\
y \\
z
\end{bmatrix} =
\begin

{bmatrix}
0.8 \\
2 \\
1
\end{bmatrix}
$$

En este ejemplo, hemos aplicado el método de Gauss particionado para resolver un sistema de ecuaciones lineales de 3x3.
Hemos dividido la matriz de coeficientes en submatrices, resuelto cada una de ellas individualmente y
combinado las soluciones para obtener la solución del sistema original.


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

if st.button('Calcular'):
    if not(is_zero_matrix(m)):

        try:
            if sp.Matrix(m).det() == 0:
                st.write('La matriz no tiene solucion :(  **|A| = 0**')
            else:
                solucion = gauss_partitioned(np.array(m), np.array(b))
                st.write('Matriz escalonada:')
                st.latex(sp.latex(sp.Matrix(solucion[-1])))
                st.write('Solucion aproximada:')
                st.latex(r'''\hat{x} \approx ''' + sp.latex(sp.Matrix(solucion[0])))
                sol = sp.Matrix(m).inv() * sp.Matrix(b)
                st.write('Error con respecto a la solucion:')
                st.latex(' \hat{x} = ' + sp.latex(sp.Matrix(sol)))
                st.latex('error = ' + sp.latex(abs(sol-sp.Matrix(solucion[0]))))
                st.write('Pasos realizados:')
                for t in solucion[1]:
                    st.latex(t)
        except:
            if sp.Matrix(m).det() == 0:
                st.write('La matriz no tiene solucion :(')
            else:
                st.write('Algo salio mal :(')



with echo_expander(code_location="below", label="Implementación en Python"):
    import numpy as np
    import sympy as sp
    def gauss_partitioned(A, b):

        n = A.shape[0]

        steps = []

        b1  = b[:n//2]
        b2 = b[n//2:]

        # Descomposición de la matriz A
        A11 = A[:n//2, :n//2]
        A12 = A[:n//2, n//2:]
        A21 = A[n//2:, :n//2]
        A22 = A[n//2:, n//2:]


        steps.append('A_{11} = ' + sp.latex(sp.Matrix(A11)))
        steps.append('A_{12} = ' + sp.latex(sp.Matrix(A12)))
        steps.append('A_{21} = ' + sp.latex(sp.Matrix(A21)))
        steps.append('A_{22} = ' + sp.latex(sp.Matrix(A22)))
        steps.append('b_{1} = ' + sp.latex(sp.Matrix(b1)))
        steps.append('b_{2} = ' + sp.latex(sp.Matrix(b2)))


        A11_inv = np.linalg.inv(A11.astype(float))
        A12_p =np.matmul(A11_inv, A12)
        b1_p = A11_inv @ b1
        A22_p = A22 - (np.matmul(A21, A12_p))
        b2_p = b2- (np.matmul(A21 , b1_p))
        A22_inv = np.linalg.inv(A22_p.astype(float))
        b2_pp = A22_inv @ b2_p
        b1_pp = b1_p- (np.matmul(A12_p , b2_pp))

        return np.concatenate((b1_pp,b2_pp)),steps,np.eye(n)

