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
def householder_transform(A):
    m, n = A.shape
    Q = np.eye(m)  # Inicializar matriz Q como la matriz identidad

    for k in range(n):
        x = A[k:, k]  # Extraer la columna k de A
        v = np.zeros_like(x)
        v[0] = x[0] + np.sign(x[0]) * np.sqrt(np.dot(x, x))  # Calcular el vector de reflexión v

        v_norm = np.sqrt(np.dot(v, v))  # Calcular la norma del vector v
        v = v / v_norm  # Normalizar el vector v

        H = np.eye(m)  # Inicializar la matriz de transformación H como la matriz identidad
        H[k:, k:] -= 2.0 * np.outer(v, v)  # Calcular la matriz de transformación H

        Q = np.dot(Q, H)  # Acumular las matrices de transformación H en Q

    R = np.dot(Q, A)  # Calcular la matriz triangular superior R

    return Q, R



st.title('5. Cálculo de Valores y Vectores Propios')

r'''
### 5.2 Transformación de Householder

La Transformación de Householder es un método utilizado en álgebra lineal numérica para transformar una matriz en una
forma más conveniente para realizar ciertos cálculos. Esta técnica se basa en la idea de reflejar un vector sobre un
hiperplano para anular una serie de elementos en la matriz.

La Transformación de Householder se lleva a cabo en los siguientes pasos:

1. Seleccionar un vector no nulo $x$ de dimensión $n$. Este vector se conoce como el vector de Householder.

2. Calcular el vector $v$, conocido como el vector de reflexión de Householder, de la siguiente manera:

$$
v = x - \mathrm{sign}(x_1) \left\|x\right\|_2 e_1
$$

donde $x_1$ es el primer elemento de $x$, $\mathrm{sign}(x_1)$ es la función de signo de $x_1$, $\left\|x\right\|_2$ es
la norma euclidiana del vector $x$ y $e_1$ es el vector canónico de dimensión $n$ con un 1 en la primera posición y
ceros en las demás posiciones.

3. Calcular la matriz de reflexión de Householder $H$ de dimensión $n \times n$, definida por:

$$
H = I - 2 \frac{vv^T}{v^Tv}
$$

donde $I$ es la matriz identidad de dimensión $n \times n$, $v^T$ es el vector transpuesto de $v$ y $\frac{vv^T}{v^Tv}$
es la matriz resultante de la división de los productos externos de $v$.

4. Calcular la matriz transformada $B$ de dimensión $n \times n$, mediante la multiplicación de la matriz de reflexión
$H$ con la matriz original $A$:

$$
B = HA
$$

La Transformación de Householder tiene varias aplicaciones importantes en el ámbito del álgebra lineal numérica.
Se utiliza para la reducción de matrices a formas más convenientes para cálculos posteriores, como la reducción a
forma tridiagonal o forma de Hessenberg. Además, se utiliza en el cálculo de factorizaciones QR, que son útiles para
resolver sistemas de ecuaciones lineales y problemas de mínimos cuadrados. También se aplica en el cálculo eficiente
de la inversa de una matriz y en el cálculo de autovectores y autovalores.


### Algoritmo

**Entrada:** Matriz $A$ de dimensión $n \times m$

**Salida:** Matriz transformada $B$ mediante la Transformación de Householder

1. Inicializar la matriz transformada $B$ como una copia de la matriz original $A$
2. Para cada columna $j$ desde 1 hasta $m$, hacer:
   - Obtener el vector $x$ correspondiente a la columna $j$ de la matriz $B$
   - Calcular el vector de reflexión de Householder $v$ usando la fórmula:
     $$
     v = x - \mathrm{sign}(x_1) \left\|x\right\|_2 e_1
     $$
   - Calcular la norma euclidiana del vector $v$:
     $$
     \left\|v\right\|_2 = \sqrt{\sum_{i=1}^n v_i^2}
     $$
   - Inicializar la matriz de reflexión $H$ como la matriz identidad $I$ de dimensión $n \times n$
   - Para cada fila $i$ desde 1 hasta $n$, hacer:
     - Calcular el producto externo de $v$:
       $$
       vv^T = \begin{bmatrix} v_1 \cdot v_1 & v_1 \cdot v_2 & \ldots & v_1 \cdot v_n \\ v_2 \cdot v_1 & v_2 \cdot v_2 & \ldots & v_2 \cdot v_n \\ \vdots & \vdots & \ddots & \vdots \\ v_n \cdot v_1 & v_n \cdot v_2 & \ldots & v_n \cdot v_n \end{bmatrix}
       $$
     - Calcular la matriz de reflexión $H$ usando la fórmula:
       $$
       H = H - 2 \frac{vv^T}{\left\|v\right\|_2^2}
       $$
   - Actualizar la matriz transformada $B$ multiplicando $H$ por la derecha con $B$:
     $$
     B = BH
     $$
3. Devolver la matriz transformada $B$

El algoritmo de Transformación de Householder realiza una serie de cálculos iterativos para transformar la matriz
original $A$ en una forma más conveniente para ciertos cálculos posteriores. Se utiliza la reflexión de Householder
para anular ciertos elementos de la matriz y obtener una matriz más tridiagonal o en forma de Hessenberg.
Esto se logra mediante la construcción de la matriz de reflexión $H$ y su aplicación a la matriz $B$ en cada iteración.


## Supuestos de aplicación

Los supuestos de aplicación del método de Transformación de Householder son los siguientes:

1. La matriz de entrada $A$ es de tamaño $n \times m$ con $n \geq m$ (es decir, tiene al menos tantas filas como columnas).
2. La matriz de entrada $A$ es de rango completo, lo que significa que todas sus columnas son linealmente independientes.
3. Se busca transformar la matriz $A$ en una forma más conveniente para realizar ciertos cálculos posteriores, como la
factorización QR, la resolución de sistemas de ecuaciones lineales o el cálculo de autovectores y autovalores.
4. Se busca reducir la complejidad computacional al reducir el número de operaciones requeridas para realizar cálculos
numéricos en la matriz transformada.
5. El método de Transformación de Householder es especialmente útil cuando se trabaja con matrices grandes y densas,
donde se busca minimizar el tiempo y los recursos computacionales necesarios para realizar los cálculos.

Es importante tener en cuenta estos supuestos al aplicar el método de Transformación de Householder, ya que garantizan
la validez y eficiencia del algoritmo. Cumplir con estos supuestos asegura que la matriz resultante después de
la transformación de Householder tenga las propiedades y características deseadas para las aplicaciones específicas
que se pretenden realizar.

## Ejemplo

Supongamos que tenemos la siguiente matriz:

$$
A = \begin{bmatrix}
2 & 3 & 1 \\
1 & 4 & 2 \\
3 & 1 & 5 \\
\end{bmatrix}
$$

Deseamos aplicar el método de Transformación de Householder para reducir la matriz $A$ a una forma triangular superior.

**Paso 1:** Calculamos el vector de reflexión $v$ utilizando la primera columna de la matriz $A$:

$$ v = x - \text{sign}(x_1) \|x\| e_1 $$

Donde $x$ es el vector columna de la primera columna de $A$ (en este caso, $x = \begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix}$), $\text{sign}(x_1)$ es el signo de $x_1$ y $e_1$ es el vector canónico de tamaño $n$ (en este caso, $e_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$).

Aplicando los cálculos, tenemos:

$$ v = \begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix} - \text{sign}(2) \sqrt{2^2 + 1^2 + 3^2} \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} -2 \\ 1 \\ 3 \end{bmatrix} $$

**Paso 2:** Calculamos la matriz de transformación de Householder $H$ utilizando el vector $v$:

$$ H = I - 2 \frac{vv^T}{v^Tv} $$

Donde $I$ es la matriz identidad de tamaño $n \times n$, $v$ es el vector calculado en el paso anterior y $v^T$ es la transpuesta del vector $v$.

Realizando los cálculos, obtenemos:

$$ H = I - 2 \frac{\begin{bmatrix} -2 \\ 1 \\ 3 \end{bmatrix} \begin{bmatrix} -2 & 1 & 3 \end{bmatrix}}{\begin{bmatrix} -2 & 1 & 3 \end{bmatrix} \begin{bmatrix} -2 \\ 1 \\ 3 \end{bmatrix}} $$

Simplificando los productos, obtenemos:

$$ H = I - 2 \frac{\begin{bmatrix} 4 & -2 & -6 \\ -2 & 1 & 3 \\ -6 & 3 & 9 \end{bmatrix}}{\begin{bmatrix} 14 \end{bmatrix}} $$

Realizando la división y las operaciones, tenemos:

$$ H = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} - \begin{bmatrix} \frac{8}{14} & -\frac{4}{14} & -\frac{12}{14} \\ -\frac{4}{14} & \frac{2}{14} & \frac{6}{14} \\ -\frac{12}{14} & \frac{6}{14} & \frac{18}{14} \end{bmatrix} $$

Simplificando los valores, tenemos:

$$ H = \begin{bmatrix} \frac{6}{7} & \frac{4}{7} & \frac{12}{7} \\ \frac{4}{7} & \frac{9}{14} & \frac{3}{7} \\ \frac{12}{7} & \frac{3}{7} & \frac{4}{7} \end{bmatrix} $$

Por lo tanto, la matriz de transformación $H$ para la primera columna de $A$ es:

$$ H = \begin{bmatrix} \frac{6}{7} & \frac{4}{7} & \frac{12}{7} \\ \frac{4}{7} & \frac{9}{14} & \frac{3}{7} \\ \frac{12}{7} & \frac{3}{7} & \frac{4}{7} \end{bmatrix} $$

Este proceso se repite para cada columna restante de $A$ hasta obtener la forma triangular superior deseada.
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
        solucion = householder_transform(np.array(m).astype('float'))
        try:
            st.write('Matriz Q:')
            st.latex(r'''\hat{Q} \approx ''' + sp.latex(sp.Matrix(solucion[0])))
            st.write('Matriz R:')
            st.latex(r'''\hat{R} \approx ''' + sp.latex(sp.Matrix(solucion[1])))
        except:
            st.error('Algo salio mal', icon="⚠️")


with echo_expander(code_location="below", label="Implementación en Python"):
    import numpy as np

    def householder_transform(A):
        m, n = A.shape
        Q = np.eye(m)  # Inicializar matriz Q como la matriz identidad

        for k in range(n):
            x = A[k:, k]  # Extraer la columna k de A
            v = np.zeros_like(x)
            v[0] = x[0] + np.sign(x[0]) * np.sqrt(np.dot(x, x))  # Calcular el vector de reflexión v

            v_norm = np.sqrt(np.dot(v, v))  # Calcular la norma del vector v
            v = v / v_norm  # Normalizar el vector v

            H = np.eye(m)  # Inicializar la matriz de transformación H como la matriz identidad
            H[k:, k:] -= 2.0 * np.outer(v, v)  # Calcular la matriz de transformación H

            Q = np.dot(Q, H)  # Acumular las matrices de transformación H en Q

        R = np.dot(Q, A)  # Calcular la matriz triangular superior R

        return Q, R
