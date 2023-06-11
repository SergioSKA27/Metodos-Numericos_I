import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math



st.title('UNIDAD 1 ANÁLISIS DE ERROR')



r'''


## Orden de convergencia

El orden de convergencia es una medida que describe la rapidez con la que un método numérico se aproxima a la solución exacta de un problema. Se utiliza para evaluar y comparar la eficiencia de diferentes algoritmos y estimar la cantidad de iteraciones necesarias para obtener una aproximación precisa.

En términos generales, el orden de convergencia se refiere a cómo se reduce el error de aproximación en cada iteración del método. Se representa mediante la notación $O(h^p)$, donde $h$ es el tamaño del paso o incremento en cada iteración y $p$ es el exponente que determina la tasa de convergencia.

Existen diferentes órdenes de convergencia comunes:

1. **Convergencia lineal ($p = 1$)**: En este caso, el error disminuye en un factor constante en cada iteración. Por ejemplo, si el error inicial es $E_0$, después de una iteración, el error se reducirá a aproximadamente $\frac{{E_0}}{{h}}$. El número de iteraciones necesarias para reducir el error en un factor de $tol$ se estima como $\frac{{\log(tol)}}{{\log(h)}}$, donde $tol$ es la tolerancia deseada.

2. **Convergencia cuadrática ($p = 2$)**: En este caso, el error se reduce en un factor cuadrático en cada iteración. Después de una iteración, el error se reduce a aproximadamente $\frac{{E_0}}{{h^2}}$. La convergencia cuadrática es mucho más rápida que la lineal, lo que significa que se necesitarán muchas menos iteraciones para alcanzar la misma tolerancia.

3. **Convergencia superlineal ($p > 1$)**: En este caso, el error disminuye más rápido que linealmente, pero no tan rápido como en la convergencia cuadrática. El orden de convergencia puede ser cualquier número mayor que 1, pero menor que 2.

Es importante tener en cuenta que el orden de convergencia depende del método numérico utilizado y de la función o problema específico que se está resolviendo. En general, un orden de convergencia mayor indica un método más eficiente y más rápido para alcanzar una solución precisa.

El orden de convergencia también se utiliza para analizar el error de truncamiento en los métodos numéricos. Cuanto menor sea el orden de convergencia, mayor será el error de truncamiento y se requerirá un tamaño de paso más pequeño para obtener resultados precisos.
'''


# Tamaños de paso (h)
h_values = np.logspace(-3, -1, 100)

# Errores absolutos para diferentes órdenes de convergencia
error_linear = h_values
error_quadratic = h_values**2
error_superlinear = h_values**1.5

# Crear gráfico de dispersión
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=h_values,
    y=error_linear,
    mode='lines',
    name='Lineal (p = 1)',
))
fig.add_trace(go.Scatter(
    x=h_values,
    y=error_quadratic,
    mode='lines',
    name='Cuadrático (p = 2)',
))
fig.add_trace(go.Scatter(
    x=h_values,
    y=error_superlinear,
    mode='lines',
    name='Superlineal (p > 1)',
))

# Configurar el diseño del gráfico
fig.update_layout(
    title='Orden de Convergencia',
    xaxis_title='Tamaño de Paso (h)',
    yaxis_title='Error Absoluto',
)

# Mostrar el gráfico
st.plotly_chart(fig)
