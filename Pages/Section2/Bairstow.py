import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math



import plotly.graph_objects as go
import numpy as np

def bairstow_method(coefficients, tolerance, max_iterations):
    n = len(coefficients) - 1
    roots = []

    while n >= 2:
        r = np.random.uniform(-1, 1)
        s = np.random.uniform(-1, 1)

        iterations = 0
        while iterations < max_iterations:
            b = coefficients.copy()
            c = coefficients.copy()

            for i in range(n, 1, -1):
                b[i] = coefficients[i] - r * b[i + 1] - s * b[i + 2]
                c[i] = b[i] - r * c[i + 1] - s * c[i + 2]

            dr = (b[1] * c[3] - b[2] * c[2]) / (c[2] ** 2 - c[3] * c[1])
            ds = (b[2] * c[1] - b[1] * c[2]) / (c[2] ** 2 - c[3] * c[1])

            r += dr
            s += ds

            iterations += 1

            if abs(dr) < tolerance and abs(ds) < tolerance:
                break

        roots.append(complex(r, s))

        coefficients = b[2:]
        n = len(coefficients) - 1

    if n == 1:
        roots.append(-coefficients[1] / coefficients[2])

    return roots



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


# Definir los coeficientes del polinomio
coefficients = [1, -3, -4, 12, -4]

# Definir la tolerancia y el número máximo de iteraciones
tolerance = 1e-6
max_iterations = 50

# Ejecutar el método de Bairstow
roots = bairstow_method(coefficients, tolerance, max_iterations)

# Crear un rango de valores para la variable x
x = np.linspace(-5, 5, 200)

# Evaluar el polinomio en el rango de valores de x
y = np.polyval(coefficients, x)

# Crear una figura de Plotly y agregar la gráfica del polinomio
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, name='Polinomio'))

# Marcar las raíces encontradas por el método de Bairstow
for root in roots:
    fig.add_trace(go.Scatter(x=[root.real], y=[0], mode='markers', name='Raíz', marker=dict(color='red', size=10)))

# Personalizar el diseño de la gráfica
fig.update_layout(title='Método de Bairstow', xaxis_title='x', yaxis_title='f(x)')

# Mostrar la gráfica
st.plotly_chart(fig)
