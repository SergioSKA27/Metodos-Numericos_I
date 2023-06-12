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

# 3.1 Condiciones necesarias y suficientes para la existencia de la solución de sistemas de ecuaciones lineales

En el contexto de los sistemas de ecuaciones lineales, es fundamental comprender las condiciones necesarias y
suficientes para determinar si un sistema tiene solución o no. Estas condiciones se pueden establecer a través de
conceptos como la consistencia y la independencia lineal.

## Consistencia del sistema

Un sistema de ecuaciones lineales se considera **consistente** si existe al menos una solución que satisface todas
las ecuaciones simultáneamente. Matemáticamente, un sistema de ecuaciones lineales se considera consistente si y solo
si existe al menos un vector solución que satisface la siguiente condición:

$$
\begin{align*}
Ax = b
\end{align*}
$$

donde A es la matriz de coeficientes, x es el vector de incógnitas y b es el vector de términos constantes.

### Independencia lineal

La independencia lineal es un concepto clave para determinar la existencia de soluciones en un sistema de ecuaciones
lineales. Se dice que un conjunto de vectores $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$ es
**linealmente independiente** si ninguna combinación lineal de los vectores puede ser igual al vector cero,
excepto cuando todos los coeficientes de la combinación lineal son cero. Matemáticamente,
esto se expresa de la siguiente manera:

$$
\begin{align*}
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \ldots + c_n\mathbf{v}_n = \mathbf{0}
\end{align*}
$$

solo si $c_1 = c_2 = \ldots = c_n = 0$, donde $c_1, c_2, \ldots, c_n$ son coeficientes escalares.

### Condición necesaria y suficiente

Para un sistema de ecuaciones lineales de coeficientes constantes, las condiciones necesarias y suficientes para la
existencia de una solución se pueden establecer de la siguiente manera:

1. Si el número de ecuaciones es igual al número de incógnitas (es decir, el sistema es cuadrado), entonces el sistema
tiene una solución si y solo si el rango de la matriz de coeficientes A es igual al número de incógnitas.
Matemáticamente, esto se expresa como:

$$
\begin{align*}
\text{rango}(A) = n
\end{align*}
$$

donde A es la matriz de coeficientes y n es el número de incógnitas.

2. Si el número de ecuaciones es mayor que el número de incógnitas, el sistema es **inconsistente** y no tiene solución.

Es importante tener en cuenta que estas condiciones son específicas para sistemas de ecuaciones lineales de coeficientes
constantes. En casos más generales, como sistemas de ecuaciones no lineales o sistemas con coeficientes variables,
pueden aplicarse diferentes condiciones para determinar la existencia de soluciones.




## Ejemplo: Condiciones para la existencia de solución en un sistema de ecuaciones lineales

Consideremos el siguiente sistema de ecuaciones lineales:

$$
\begin{align*}
2x + 3y &= 7 \\
4x - 5y &= -1 \\
\end{align*}
$$

Para determinar si este sistema tiene solución, verificaremos las condiciones necesarias y suficientes.

### Consistencia del sistema

Calculamos la matriz de coeficientes y el vector de términos constantes:

$$
A = \begin{bmatrix} 2 & 3 \\ 4 & -5 \end{bmatrix}, \quad b = \begin{bmatrix} 7 \\ -1 \end{bmatrix}
$$

Luego, podemos verificar si existe una solución al resolver la ecuación matricial:

$$
\begin{align*}
Ax = b
\end{align*}
$$

Si encontramos al menos un vector solución, el sistema es consistente.

### Independencia lineal

Podemos examinar la independencia lineal de las columnas de la matriz de coeficientes A para determinar la existencia
de soluciones únicas. Si las columnas son linealmente independientes, el sistema tendrá una única solución.

### Condición necesaria y suficiente

En este caso, el número de ecuaciones es igual al número de incógnitas (2), por lo que el sistema es cuadrado.
Para que tenga solución, debemos verificar si el rango de la matriz de coeficientes A es igual al número de incógnitas.

Calculamos el rango de A y verificamos si coincide con el número de incógnitas.

Si encontramos que $\text{rango}(A) = 2$, concluimos que el sistema es consistente y tiene una única solución.
En caso contrario, el sistema no tiene solución.

En este ejemplo, los cálculos muestran que $\text{rango}(A) = 2$, lo que significa que el sistema es consistente y
tiene una única solución.

Por lo tanto, el sistema de ecuaciones lineales dado tiene una solución única dada por $x = 2$ y $y = 1$.


'''
