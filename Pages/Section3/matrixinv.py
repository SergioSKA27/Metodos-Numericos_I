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









r'''
# 3.1.1 Inversión de matrices

La inversión de matrices es un concepto fundamental en el álgebra lineal que nos permite resolver sistemas de
ecuaciones lineales, encontrar soluciones únicas y realizar diversas operaciones matemáticas. Dada una matriz
cuadrada $A$ de tamaño $n \times n$, se dice que tiene una inversa si existe otra matriz $A^{-1}$ tal que el
producto de $A$ por $A^{-1}$ (y viceversa) es igual a la matriz identidad $I$:

$$
A \cdot A^{-1} = A^{-1} \cdot A = I
$$

La matriz inversa $A^{-1}$ es única para cada matriz invertible $A$. Para determinar si una matriz tiene una inversa,
es necesario evaluar su determinante. Si el determinante de la matriz $A$ es diferente de cero
($det(A) \neq 0$), entonces $A$ es invertible y tiene una matriz inversa.

Existen varios métodos para calcular la matriz inversa de una matriz invertible. Uno de los métodos más comunes es la
eliminación de Gauss-Jordan. Este método consiste en realizar operaciones elementales en las filas de la matriz
aumentada $[A | I]$ hasta reducir $A$ a su forma escalonada reducida, mientras que simultáneamente se realiza la misma
secuencia de operaciones elementales en $I$. Al finalizar, $A$ se transforma en $I$ y $I$ se transforma en $A^{-1}$.
Este proceso garantiza que el producto de $A$ por su inversa sea igual a la matriz identidad.

Otro enfoque para calcular la matriz inversa es utilizando la matriz adjunta. La matriz adjunta $adj(A)$ se obtiene
al calcular el cofactor de cada elemento de $A$ y luego transponer la matriz resultante. La matriz inversa se puede
obtener dividiendo cada elemento de $adj(A)$ por el determinante de $A$:

$$
A^{-1} = \frac{{adj(A)}}{{det(A)}}
$$

Es importante destacar que solo las matrices cuadradas invertibles tienen una matriz inversa.
Si una matriz no tiene inversa, se dice que es una matriz singular o no invertible.
Una matriz singular tiene un determinante igual a cero y no se puede invertir.



## Ejemplo

Supongamos que tenemos la siguiente matriz cuadrada $A$:

$$
\begin{align*}
A = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}
\end{align*}
$$

Para determinar si esta matriz es invertible, calculamos su determinante:

$$det(A) = 2 \cdot 3 - 1 \cdot 4 = 6 - 4 = 2 \neq 0$$

Como el determinante de $A$ es diferente de cero, podemos afirmar que $A$ es invertible.

Ahora, utilizaremos el método de eliminación de Gauss-Jordan para encontrar la matriz inversa de $A$:

1. Creamos una matriz aumentada $[A | I]$, donde $I$ es la matriz identidad de tamaño $2 \times 2$:

$$
\begin{align*}
[A | I] = \begin{bmatrix} 2 & 1 & 1 & 0 \\ 4 & 3 & 0 & 1 \end{bmatrix}
\end{align*}
$$

2. Realizamos operaciones elementales para transformar $A$ en la matriz identidad y simultáneamente transformar $I$ en $A^{-1}$:

- Dividimos la primera fila por 2: $\frac{1}{2} \cdot R_1 \rightarrow R_1$:

$$
\begin{align*}
\begin{bmatrix} 1 & \frac{1}{2} & \frac{1}{2} & 0 \\ 4 & 3 & 0 & 1 \end{bmatrix}
\end{align*}
$$

- Restamos 4 veces la primera fila de la segunda fila: $R_2 - 4R_1 \rightarrow R_2$:

$$
\begin{align*}
\begin{bmatrix} 1 & \frac{1}{2} & \frac{1}{2} & 0 \\ 0 & 1 & -2 & 1 \end{bmatrix}
\end{align*}
$$

- Multiplicamos la segunda fila por $\frac{1}{2}$: $\frac{1}{2} \cdot R_2 \rightarrow R_2$:

$$
\begin{align*}
\begin{bmatrix} 1 & \frac{1}{2} & \frac{1}{2} & 0 \\ 0 & \frac{1}{2} & -1 & \frac{1}{2} \end{bmatrix}
\end{align*}
$$

- Restamos $\frac{1}{2}$ veces la segunda fila de la primera fila: $R_1 - \frac{1}{2}R_2 \rightarrow R_1$:

$$
\begin{align*}
\begin{bmatrix} 1 & 0 & 1 & -\frac{1}{2} \\ 0 & \frac{1}{2} & -1 & \frac{1}{2} \end{bmatrix}
\end{align*}
$$

3. Obtenemos la matriz inversa $A^{-1}$, que se encuentra en la parte derecha de la matriz aumentada:

$$
\begin{align*}
A^{-1} = \begin{bmatrix} 1 & -\frac{1}{2} \\ 0 & \frac{1}{2} \end{bmatrix}
\end{align*}
$$

Por lo tanto, la matriz inversa de $A$ es:

$$
\begin{align*}
A^{-1} = \begin{bmatrix} 1 & -\frac{1}{2} \\ 0 & \frac{1}{2} \end{bmatrix}
\end{align*}
$$

Podemos comprobar que $A \cdot A^{-1} = I$ para confirmar que hemos calculado correctamente la matriz inversa.

$$
A \cdot A^{-1} = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix} \cdot \begin{bmatrix} 1 & -\frac{1}{2} \\ 0 & \frac{1}{2} \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I
$$

El producto de $A$ por su inversa da como resultado la matriz identidad, lo que confirma que hemos calculado
correctamente la matriz inversa de $A$.
'''



file1_ = open("Pages\Section3\FigMatrixinv.png", "rb")



contents1 = file1_.read()
data_url1 = base64.b64encode(contents1).decode("utf-8")
file1_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url1}" alt="fig_Matrix_inv">',
    unsafe_allow_html=True,
)



st.subheader('Calculadora de Matriz Inversa :triangular_ruler:')


r = st.number_input('Ingrese el tamño de la matriz:', min_value=1, max_value=10,value=4)

mat1 = pd.DataFrame(np.zeros((r,r)),columns=[str(i+1) for i in range(r)])
st.write('Ingrese la matriz:')
mat2 = st.data_editor(mat1, key='mat1')



st.latex('A = ' + sp.latex(sp.Matrix(mat2)))

r'# Matriz Inversa'

try:
    st.latex('A^{-1} = ' + sp.latex(sp.Matrix(mat2).inv()))
except:
    st.write('La matriz no es invertible :(')
