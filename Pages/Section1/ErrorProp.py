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
## 1.2 Propagación del error en distintas operaciones aritméticas

Cuando realizamos operaciones aritméticas con números, es importante considerar cómo se propaga el error en el resultado final. Cada operación introduce cierto grado de incertidumbre, y esta incertidumbre se acumula a medida que se realizan más operaciones. A continuación, analizaremos la propagación del error en operaciones aritméticas comunes:

### Suma y resta:
En la suma y resta de dos números, el error absoluto en el resultado final es la suma de los errores absolutos de los números originales. Si los números tienen errores absolutos $E_{\text{abs1}}$ y $E_{\text{abs2}}$, respectivamente, el error absoluto en el resultado será $E_{\text{abs\_final}} = E_{\text{abs1}} + E_{\text{abs2}}$.

### Multiplicación y división:
En la multiplicación y división de dos números, el error relativo en el resultado final es la suma de los errores relativos de los números originales. Si los números tienen errores relativos $E_{\text{rel1}}$ y $E_{\text{rel2}}$, respectivamente, el error relativo en el resultado será $E_{\text{rel\_final}} = E_{\text{rel1}} + E_{\text{rel2}}$.

### Potenciación y radicación:
En las operaciones de potenciación y radicación, el error relativo en el resultado final se multiplica por el exponente. Si el número tiene un error relativo $E_{\text{rel}}$ y se eleva a un exponente $n$, el error relativo en el resultado será $E_{\text{rel\_final}} = n \times E_{\text{rel}}$.

### Funciones trigonométricas y exponenciales:
En las funciones trigonométricas y exponenciales, el error relativo en el resultado final puede depender de la función específica y del valor del argumento. Estas funciones pueden introducir errores significativos en ciertos rangos de valores y deben tenerse en cuenta al realizar cálculos precisos.

Es importante destacar que estos análisis suponen que los errores en los números originales son independientes y se propagan de manera lineal. En la práctica, la propagación del error puede ser más compleja y dependerá del método utilizado para realizar las operaciones aritméticas.

Para minimizar la propagación del error, es recomendable utilizar métodos numéricos más precisos, como el uso de aritmética de alta precisión o algoritmos que minimicen los efectos acumulativos de los errores. Además, el uso de técnicas de análisis de sensibilidad y estimación de errores puede ayudar a evaluar y controlar el impacto de la propagación del error en los resultados finales.





'''
# Datos originales con errores
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
error = np.array([1, 2, 3])  # Errores absolutos en los datos

# Propagación del error en la suma
suma = y + x
error_suma = np.sqrt(error**2 + error**2)  # Error absoluto en la suma

# Propagación del error en la multiplicación
multiplicacion = y * x
error_multiplicacion = multiplicacion * np.sqrt((error/y)**2 + (error/x)**2)  # Error relativo en la multiplicación

# Crear gráfico de dispersión con barras de error
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    name='Datos Originales',
    error_x=dict(type='data', array=error, visible=True),
    error_y=dict(type='data', array=error, visible=True),
))
fig.add_trace(go.Scatter(
    x=x,
    y=suma,
    mode='markers',
    name='Suma',
    error_y=dict(type='data', array=error_suma, visible=True),
))
fig.add_trace(go.Scatter(
    x=x,
    y=multiplicacion,
    mode='markers',
    name='Multiplicación',
    error_y=dict(type='data', array=error_multiplicacion, visible=True),
))

# Configurar el diseño del gráfico
fig.update_layout(
    title='Propagación del Error',
    xaxis_title='X',
    yaxis_title='Y',
)

# Mostrar el gráfico

st.code(r''' # Datos originales con errores
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
error = np.array([1, 2, 3])  # Errores absolutos en los datos

# Propagación del error en la suma
suma = y + x
error_suma = np.sqrt(error**2 + error**2)  # Error absoluto en la suma

# Propagación del error en la multiplicación
multiplicacion = y * x
error_multiplicacion = multiplicacion * np.sqrt((error/y)**2 + (error/x)**2)  # Error relativo en la multiplicación
''')
st.plotly_chart(fig)
