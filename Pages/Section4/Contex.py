import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math




st.header('4. Factorización LU y sus Aplicaciones')

r'''
# 4.1 Modelos de contexto y comportamiento
La factorización LU (Lower-Upper) es una técnica matemática utilizada para descomponer una matriz cuadrada $ A $ en el
producto de dos matrices: una matriz triangular inferior $ L $ y una matriz triangular superior $ U $.
Esta descomposición se puede expresar matemáticamente como $ A = LU $.



La descomposición LU proporciona modelos de contexto y comportamiento para el sistema de ecuaciones representado por
la matriz $ A $. La matriz triangular inferior $ L $ contiene los coeficientes necesarios para realizar las operaciones
de eliminación hacia adelante, mientras que la matriz triangular superior $ U $ especifica cómo se deben combinar las
variables para obtener los resultados del sistema mediante la sustitución hacia atrás.

Matemáticamente, la descomposición LU se puede representar como:

$$ A = LU $$

donde $ L $ es una matriz triangular inferior con elementos $ l_{ij} $ (donde $ 1 \leq i \leq n $ y $ 1 \leq j \leq i $)
y $ U $ es una matriz triangular superior con elementos $ u_{ij} $ (donde $ 1 \leq i \leq j \leq n $).
La matriz $ A $ es una matriz cuadrada de tamaño $ n \times n $.

## Aplicaciones de la Factorización LU

La factorización LU tiene varias aplicaciones fundamentales en el ámbito de la resolución de sistemas de ecuaciones
lineales y otros problemas numéricos. Algunas de las aplicaciones más destacadas son:

1. **Resolución de sistemas de ecuaciones lineales**: Una vez que se ha obtenido la factorización LU de una matriz
$ A $, es posible resolver eficientemente sistemas de ecuaciones lineales de la forma $ Ax = b $. Al dividir
el sistema en dos etapas (sustitución hacia adelante y sustitución hacia atrás) utilizando las matrices $ L $ y $ U $,
se puede obtener la solución del sistema de manera más eficiente en comparación con otros métodos.

2. **Cálculo de la matriz inversa**: La factorización LU también se utiliza para calcular la matriz inversa de una
matriz cuadrada $ A $. Al tener la descomposición en las matrices $ L $ y $ U $, se puede obtener la matriz inversa
mediante la resolución de sistemas lineales para cada columna de la matriz identidad.

3. **Cálculo del determinante**: A partir de la factorización LU, se puede calcular el determinante de una matriz
cuadrada $ A $. El determinante se obtiene multiplicando los elementos diagonales de la matriz $ U $. Si la matriz
$ A $ es singular, es decir, si tiene un determinante igual a cero, la factorización LU proporciona una forma eficiente
de verificar esta singularidad.

4. **Optimización de métodos iterativos**: La factorización LU también se utiliza en métodos iterativos para mejorar su
convergencia y estabilidad. Al proporcionar una matriz descompuesta $ A = LU $ como punto de partida, los métodos
iterativos, como el método de Gauss-Seidel o el método de relajación, pueden converger más rápidamente a la solución deseada.

En resumen, la factorización LU es una técnica matemática poderosa que descompone una matriz cuadrada $ A $ en
las matrices $ L $ y $ U $. Esta descomposición proporciona modelos de contexto y comportamiento para el sistema de
ecuaciones representado por $ A $. La factorización LU tiene aplicaciones fundamentales en la resolución de sistemas
de ecuaciones lineales, cálculo de la matriz inversa y determinante, así como en la mejora de la convergencia de
métodos iterativos. Su uso permite simplificar y optimizar los cálculos numéricos relacionados con matrices.
'''
