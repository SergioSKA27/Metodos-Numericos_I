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


def gauss_pivoteo_parcial(A, b):
    n = len(A)
    # Crear matriz aumentada
    aumentada = np.concatenate((A, np.array([b]).T), axis=1)
    steps = []

    # Eliminación gaussiana con pivoteo parcial
    for i in range(n-1):
        # Pivoteo parcial
        max_index = np.argmax(np.abs(aumentada[i:, i]))+i
        #print(aumentada[max_index])

        if max_index != i:
            aumentada[[i, max_index]] = aumentada[[max_index, i]]
            steps.append(f'R_{i} <---> R_{max_index}')
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

## 3.2.1 Método de Gauss y pivoteo parcial

El método de Gauss con pivoteo parcial es un procedimiento utilizado para resolver sistemas de ecuaciones lineales.
Este método se basa en la eliminación gaussiana y la estrategia del pivoteo parcial para mejorar la estabilidad numérica
y garantizar la existencia de soluciones.

El proceso se lleva a cabo de la siguiente manera:

1. Dado un sistema de ecuaciones lineales $Ax = b$, donde $A$ es una matriz de coeficientes de tamaño $n \times n$,
$x$ es el vector de incógnitas y $b$ es el vector de términos constantes.

2. Se crea una matriz aumentada $[A | b]$ que combina la matriz de coeficientes $A$ y el vector de términos
constantes $b$.

3. En cada paso de la eliminación gaussiana, se selecciona el elemento de mayor magnitud (pivote parcial) en la
columna actual y se intercambia la fila correspondiente con la fila actual. Esto se hace para evitar divisiones por
cero y reducir errores numéricos.

4. Después del pivoteo parcial, se realiza la eliminación gaussiana normal dividiendo cada fila por el pivote seleccionado,
de modo que se obtengan ceros debajo del pivote.

5. Este proceso se repite para las columnas restantes hasta obtener una matriz triangular superior.

6. Luego, se realiza la sustitución hacia atrás para encontrar la solución del sistema. Comenzando desde la última
ecuación, se despejan las incógnitas una por una utilizando los coeficientes de la matriz triangular superior.

7. Finalmente, se obtiene la solución del sistema de ecuaciones lineales.

El pivoteo parcial es esencial en el método de Gauss para garantizar que los pivotes sean diferentes de cero y
para reducir los errores numéricos. Al seleccionar el pivote de mayor magnitud en cada paso, se mejora la estabilidad
numérica y se minimiza la propagación de errores.

Es importante tener en cuenta que el método de Gauss con pivoteo parcial asume que la matriz de coeficientes $A$ es
no singular, lo que garantiza la existencia de una solución única para el sistema de ecuaciones. Si la matriz $A$ es
singular o el sistema es inconsistente, el método puede producir resultados incorrectos o fallar.

En resumen, el método de Gauss con pivoteo parcial es un procedimiento riguroso y confiable para resolver sistemas de
ecuaciones lineales. Al utilizar la eliminación gaussiana y el pivoteo parcial, se mejora la estabilidad numérica y se
obtienen soluciones precisas para los sistemas de ecuaciones.

## Algoritmo

**Entrada:** Una matriz de coeficientes A de tamaño n x n y un vector de términos constantes b de tamaño n.

**Salida:** El vector solución x del sistema de ecuaciones lineales Ax = b.

1. Crear una matriz aumentada [A | b].
2. Para i de 1 a n-1:
   - Encontrar el pivote parcial max en la columna i y su fila correspondiente filaMax.
   - Intercambiar la fila i con la fila filaMax tanto en la matriz aumentada como en el vector de términos constantes.
   - Para k de i+1 a n:
     - Calcular el factor multiplicativo factor = A[k][i] / A[i][i].
     - Para j de i a n:
       - Actualizar los elementos de la fila k de la matriz aumentada: A[k][j] = A[k][j] - factor * A[i][j].
     - Actualizar el elemento k del vector de términos constantes: b[k] = b[k] - factor * b[i].
3. Realizar la sustitución hacia atrás:
   - Crear un vector x de tamaño n lleno de ceros.
   - Para i de n a 1 con paso -1:
     - Inicializar la suma suma = 0.
     - Para j de i+1 a n:
       - Actualizar la suma: suma = suma + A[i][j] * x[j].
     - Calcular el valor de la incógnita x[i]: x[i] = (b[i] - suma) / A[i][i].
4. Retornar el vector solución x.

# Algoritmo

¡Por supuesto! Aquí tienes el algoritmo del Método de Gauss-Jordan con pivoteo total utilizando más notación LaTeX:

1. Construye la matriz aumentada del sistema de ecuaciones lineales:

$$
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} & b_1 \\
a_{21} & a_{22} & \cdots & a_{2n} & b_2 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} & b_m \\
\end{bmatrix}
$$

2. Para cada columna, desde la columna 1 hasta la columna $n-1$, donde $n$ es el número de incógnitas:

   a. Encuentra el pivote máximo en valor absoluto en la columna actual desde la fila actual hasta la última fila:

   $$
   \text{{Pivote}} = \max_{i=k}^{m} |a_{ik}|
   $$

   b. Realiza el intercambio de filas para que el elemento pivote se encuentre en la fila actual:

   $R_i \leftrightarrow R_j$

   c. Divide la fila del pivote por el valor del pivote para convertirlo en 1:

   $R_i = \frac{1}{{a_{ik}}}\cdot R_i$

   d. Para cada fila, excepto la fila del pivote:

      i. Multiplica la fila del pivote por el coeficiente necesario para hacer cero el elemento debajo del pivote en la columna actual:

      $R_j = R_j - a_{jk} \cdot R_i$

      ii. Resta la fila del pivote multiplicada por el coeficiente obtenido de la fila actual:

      $R_j = R_j - a_{jk} \cdot R_i$

3. La matriz resultante estará en forma escalonada reducida por filas.

4. Verifica si el sistema es compatible o incompatible:

   a. Si hay una fila con todos los coeficientes cero excepto el término independiente, y el término independiente es diferente de cero, el sistema es incompatible y no tiene solución.

   b. Si hay una fila con todos los coeficientes cero y el término independiente es cero, el sistema tiene infinitas soluciones.

5. Si el sistema es compatible y tiene solución única, lee las soluciones a partir de la matriz resultante.

Espero que este algoritmo sea útil para ti. Si tienes alguna pregunta adicional, ¡no dudes en preguntar!

## Ejemplo
Consideremos el siguiente sistema de ecuaciones lineales:

```
x + y = 3
2x - y = 2
```

Podemos representar este sistema en forma matricial como:

```
Ax = b
```

Donde:

```
A = [[1, 1],
     [2, -1]]

x = [x, y]

b = [3, 2]
```

Aplicando el método de Gauss con pivoteo parcial, el procedimiento sería el siguiente:

1. Paso inicial: La matriz aumentada es `[A | b]`:

```
Aumentada = [[1, 1, |, 3],
             [2, -1, |, 2]]
```

2. Paso de pivoteo parcial: Seleccionamos el pivote parcial como el elemento de mayor magnitud en la primera columna (2 en este caso) y se intercambia la fila correspondiente:

```
Aumentada = [[2, -1, |, 2],
             [1, 1, |, 3]]
```

3. Eliminación gaussiana: Dividimos la primera fila por el pivote:

```
Aumentada = [[1, -0.5, |, 1],
             [1, 1, |, 3]]
```

Restamos la primera fila multiplicada por el primer elemento de la segunda fila:

```
Aumentada = [[1, -0.5, |, 1],
             [0, 1.5, |, 2]]
```

4. Sustitución hacia atrás: Resolvemos la segunda ecuación para obtener el valor de `y`:

```
1.5y = 2
y = 2 / 1.5
y = 4/3
```

Sustituyendo el valor de `y` en la primera ecuación:

```
x - 4/3 = 1
x = 1 + 4/3
x = 7/3
```

Por lo tanto, la solución del sistema de ecuaciones lineales es `x = 7/3` y `y = 4/3`.
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
            solucion = gauss_pivoteo_parcial(np.array(m), np.array(b))
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




