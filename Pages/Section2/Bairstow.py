import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math





st.title('2. Solución Numérica de Ecuaciones de una Sola Variable')




r'''
# 2.5 Método de Bairstow

El método de Bairstow es un algoritmo utilizado para encontrar las raíces de un polinomio de grado dos o superior. A diferencia de otros métodos, como el método de Newton o el método de la secante, el método de Bairstow puede encontrar tanto raíces reales como complejas de un polinomio.

El algoritmo del método de Bairstow se puede resumir en los siguientes pasos:

1. Dado un polinomio de grado $n$, se establecen dos valores iniciales para las raíces, $r$ y $s$, que pueden ser estimados de antemano o seleccionados al azar.

2. Se itera el siguiente proceso hasta que las raíces converjan a un valor deseado:

    a. Se calculan las derivadas del polinomio $f(x)$ con respecto a $r$ y $s$, denotadas como $f_r(x)$ y $f_s(x)$, respectivamente.

    b. Se resuelve el siguiente sistema de ecuaciones lineales utilizando el método de eliminación de Gauss-Jordan:

    $$
    \begin{bmatrix}
    f(r, s) \\
    f_r(r, s) \\
    f_s(r, s)
    \end{bmatrix}
    \begin{bmatrix}
    \Delta r \\
    \Delta s \\
    \Delta s_2
    \end{bmatrix}
    =
    \begin{bmatrix}
    0 \\
    -f_r \\
    -f_s
    \end{bmatrix}
    $$

    Donde $f(r, s)$ es el valor del polinomio evaluado en $r$ y $s$, $f_r$ y $f_s$ son los valores de las derivadas evaluadas en $r$ y $s$, respectivamente, y $\Delta r$, $\Delta s$, y $\Delta s_2$ son los incrementos en $r$, $s$, y $s^2$ respectivamente.

    c. Se actualizan los valores de $r$ y $s$ sumando los incrementos obtenidos en el paso anterior.

3. Se repite el proceso de iteración hasta alcanzar una convergencia deseada o un número máximo de iteraciones.

El método de Bairstow es especialmente útil para encontrar raíces complejas múltiples y raíces reales múltiples de un polinomio. Sin embargo, es importante tener en cuenta que este método puede ser sensible a la elección inicial de las raíces y puede requerir algunas iteraciones para obtener resultados precisos.

En resumen, el método de Bairstow es un algoritmo utilizado para encontrar las raíces de un polinomio. Es capaz de encontrar tanto raíces reales como complejas y es especialmente útil para raíces múltiples. Aunque puede requerir varias iteraciones y puede ser sensible a las elecciones iniciales, el método de Bairstow es una herramienta poderosa en el campo del análisis numérico de polinomios.


'''



