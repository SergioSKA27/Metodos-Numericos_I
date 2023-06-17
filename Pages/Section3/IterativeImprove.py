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

# 3.4.1 Mejoramiento iterativo de la solución

En el análisis numérico, el mejoramiento iterativo de la solución es una técnica utilizada para aproximar la solución
de un problema matemático mediante un proceso iterativo. Esta técnica se emplea cuando no se puede obtener una solución
exacta analítica o cuando la solución exacta es computacionalmente costosa de calcular.

El objetivo del mejoramiento iterativo es encontrar una secuencia de aproximaciones $\{\mathbf{x}^{(k)}\}_{k=0}^{\infty}$
que converjan a la solución exacta $\mathbf{x}^*$ del problema. En cada iteración $k$, se calcula una nueva aproximación
$\mathbf{x}^{(k+1)}$ basada en la aproximación anterior $\mathbf{x}^{(k)}$ y una función iterativa $F$.

Sea $F: \mathbb{R}^n \rightarrow \mathbb{R}^n$ una función que define el proceso iterativo, y consideremos el problema
de encontrar $\mathbf{x}^*$ tal que $F(\mathbf{x}^*) = \mathbf{x}^*$. Entonces, el mejoramiento iterativo se puede
describir como sigue:

1. Se elige una aproximación inicial $\mathbf{x}^{(0)}$.
2. Para $k = 0, 1, 2, \ldots$:
   - Se calcula la nueva aproximación $\mathbf{x}^{(k+1)} = F(\mathbf{x}^{(k)})$.
   - Se verifica si se ha alcanzado la condición de convergencia deseada. Esto se puede hacer mediante la evaluación de
   un criterio de convergencia, como la norma del error relativo entre dos iteraciones consecutivas o la norma del
   residuo. Si se satisface la condición de convergencia, se detiene el proceso iterativo y se acepta
   $\mathbf{x}^{(k+1)}$ como la aproximación final.
   - De lo contrario, se actualiza la aproximación y se continúa con la siguiente iteración.

Un ejemplo común de mejoramiento iterativo es el Método de Jacobi para sistemas de ecuaciones lineales.
Dado un sistema de ecuaciones lineales $\mathbf{Ax} = \mathbf{b}$, donde $\mathbf{A}$ es la matriz de coeficientes de
tamaño $n \times n$, $\mathbf{x}$ es el vector de incógnitas y $\mathbf{b}$ es el vector de términos independientes,
el Método de Jacobi se basa en la siguiente iteración:

$$\mathbf{x}^{(k+1)} = \mathbf{D}^{-1}(\mathbf{b} - \mathbf{Rx}^{(k)})$$

donde $\mathbf{D}$ es la matriz diagonal de $\mathbf{A}$ y $\mathbf{R}$ es la matriz que contiene los términos no
diagonales de $\mathbf{A}$.

Es importante destacar que para garantizar la convergencia del método, se requiere que la matriz $\mathbf{A}$ sea
diagonalmente dominante o que sea simétrica y definida posit

iva. En otras palabras, se deben cumplir ciertas condiciones en los elementos de la matriz $\mathbf{A}$ para que el
método sea convergente.

La convergencia del mejoramiento iterativo se establece mediante criterios de convergencia, que dependen del problema
específico que se esté abordando. Algunos de los criterios comunes incluyen la norma del error relativo, la norma del
residuo, la tolerancia absoluta o relativa, entre otros. Estos criterios determinan cuándo detener el proceso iterativo
y aceptar la aproximación obtenida como una solución satisfactoria.

El mejoramiento iterativo ofrece varias ventajas. Permite aproximaciones sucesivas que pueden acercarse cada vez más
a la solución exacta, y es especialmente útil cuando el problema es no lineal o cuando la matriz de coeficientes no
es de fácil inversión. Sin embargo, es importante tener en cuenta que la convergencia no está garantizada en todos los
casos, y se debe realizar un análisis cuidadoso de las condiciones del problema y de las características de la función
iterativa utilizada.

En resumen, el mejoramiento iterativo de la solución es una técnica en el análisis numérico utilizada para aproximar la
solución de problemas matemáticos mediante un proceso iterativo. Se basa en la generación de una secuencia de aproximaciones
que convergen hacia la solución exacta. El Método de Jacobi es un ejemplo común de mejoramiento iterativo para sistemas de
ecuaciones lineales. La convergencia se establece mediante criterios de convergencia, y la elección adecuada de la función
iterativa y la matriz de coeficientes es fundamental para lograr una convergencia satisfactoria.
'''
