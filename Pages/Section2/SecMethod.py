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

def ddf(x):
    return x**3 - 2*x - 5

def secant_method(f,x0, x1, tolerance, max_iterations):
    x_values = []
    y_values = []
    iterations = 0
    fx = sp.lambdify(list(f.free_symbols),f)
    tab = []

    while iterations < max_iterations:
        x_values.append(x1)
        y_values.append(fx(x1))
        tab.append([x0,x1,fx(x0),fx(x1),x1 - (fx(x1) * (x1 - x0)) / (fx(x1) - fx(x0)),str(abs(fx(x1)) < tolerance)])
        if abs(fx(x1)) < tolerance:
            break

        x2 = x1 - (fx(x1) * (x1 - x0)) / (fx(x1) - fx(x0))
        x0 = x1
        x1 = x2

        iterations += 1

    return x_values, y_values,tab





st.title('2. Solución Numérica de Ecuaciones de una Sola Variable')






r'''

# 2.4 Método de la Secante

El método de la secante es un algoritmo iterativo utilizado para encontrar raíces de una función. A diferencia del método de Newton, que requiere el cálculo explícito de la derivada de la función, el método de la secante estima la derivada numéricamente utilizando la información de dos puntos cercanos. Esto hace que el método de la secante sea más flexible y aplicable a una amplia gama de funciones.

El algoritmo del método de la secante se puede resumir en los siguientes pasos:

1. Dados dos puntos iniciales $x_0$ y $x_1$ cercanos a la raíz de la función $f(x)$, se calcula la pendiente de la recta secante que pasa por los puntos $(x_0, f(x_0))$ y $(x_1, f(x_1))$. La pendiente de la recta secante se puede aproximar utilizando la fórmula:

$$
\begin{align*}
m = \frac{{f(x_1) - f(x_0)}}{{x_1 - x_0}}
\end{align*}
$$

2. La recta secante intersecta el eje $x$ en el punto $x_2$, que se convierte en una mejor aproximación de la raíz. El valor de $x_2$ se puede calcular mediante la ecuación de la recta:

$$
\begin{align*}
x_2 = x_1 - \frac{{f(x_1) \cdot (x_1 - x_0)}}{{f(x_1) - f(x_0)}}
\end{align*}
$$

3. Se actualizan los valores de $x_0$ y $x_1$ con $x_1$ y $x_2$ respectivamente.

4. Se repiten los pasos 1 y 2 hasta alcanzar una aproximación deseada o agotar un número máximo de iteraciones.

El método de la secante combina las ventajas del método de la regla falsa y el método de Newton. A diferencia de la regla falsa, no requiere el cálculo de dos valores de función en cada iteración. En cambio, utiliza la información de dos puntos anteriores para estimar la raíz. A diferencia del método de Newton, no necesita conocer la derivada de la función, lo que lo hace más general y aplicable a una amplia variedad de funciones.

La convergencia del método de la secante depende de la elección adecuada de los puntos iniciales. Es importante seleccionar dos puntos cercanos a la raíz y evitar valores que causen divisiones por cero o resultados indefinidos. Si los puntos iniciales están demasiado alejados, el método puede converger lentamente o incluso divergir. En tales casos, se recomienda ajustar los puntos iniciales o utilizar métodos alternativos.

El método de la secante tiene la ventaja de no requerir el cálculo explícito de la derivada de la función, lo que lo hace útil cuando el cálculo de la derivada es complicado o costoso. Sin embargo, el método puede presentar algunas limitaciones. Puede ser sensible a las irregularidades de la función y puede converger a raíces diferentes dependiendo de los puntos iniciales. Por lo tanto, es importante evaluar cuidadosamente las características de la función y la elección de los puntos iniciales para obtener resultados precisos.

###  :paperclip: Ejemplo

Sea

$$
\begin{align*}
f(x) = x^3-2x-5,
\end{align*}
$$

con la aproximación inicial $x_0 = 2.0$ y $x_1 = 3.0$, con un error de tolerancia $1 \times 10^{-6}$:



'''


# Definir puntos iniciales y parámetros del método de la secante
x0 = 2.0
x1 = 3.0
tolerance = 1e-6
max_iterations = 100
x = sp.symbols('x')

# Ejecutar el método de la secante
x_values, y_values,t = secant_method(x**3-2*x-5,x0, x1, tolerance, max_iterations)
table1 = pd.DataFrame(t,columns=['x_0','x_1','f(x_0)','f(x_1)','x_2','|f(c)| < '+str(1e-6)])
st.write(table1)
# Graficar la función y el proceso de iteración
x_range = [-5, 5]
xs = np.linspace(x_range[0], x_range[1], 100)
y = sp.lambdify(x,x**3-2*x-5)

fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=y(xs), name='Función f(x)'))
fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', name='Iteraciones del método',marker_color='rgba(148, 0, 1, .9)'))
fig.update_layout(title='Método de la Secante', xaxis_title='x', yaxis_title='f(x)', xaxis_range=x_range)
fig.add_hline(0)
st.plotly_chart(fig)



st.header('Método :triangular_ruler:')

fxx = st.text_input('Ingrese una funcion $f(x)$ :',value='x^2-4*x+2')
try:
    ff = sp.parse_expr(fxx,transformations='all')

    st.write('$f(x) =' + str(sp.latex(ff))+'$')


    fig2 = go.Figure()

    # Agregar la función f(x)
    a2,b2 = st.slider('Rango de la gráfica',-100,100,(-10,10))
    x2 = np.linspace(a2, b2, 1000)
    y2 = sp.lambdify(x,ff)
    fig2.add_trace(go.Scatter(x=x2, y=y2(x2), name='f(x)'))

    fig2.add_hline(0)
    # Establecer el título y etiquetas de los ejes
    fig2.update_layout(title='Grafica de la función', xaxis_title='x', yaxis_title='f(x)')

    # Mostrar la figura
    st.plotly_chart(fig2)
except :
    st.error('Error al introducir la funcion f(x)', icon="🚨")

st.write('Ingrese los puntos $x_0$ y $x_1$:')
aval = st.number_input('Ingrese $x_0$: ',-100,100,value=-1)
bval = st.number_input('Ingrese $x_1$: ',-100,100,value=1)
error = st.text_input('Ingrese el error de tolerancia: ',value='1e-6')

maxitr = st.number_input('Ingrese el máximo número de iteraciones: ',1,1000,value=10)


try:
    x_vals,y_vals,mtab = secant_method(ff,aval,bval,float(error),maxitr)
    tpd2 = pd.DataFrame(mtab,columns=['x_0','x_1','f(x_0)','f(x_1)','x_2','|f(c)| < '+str(error)])
    st.write(tpd2)
    # Crear una figura de Plotly
    fig3 = go.Figure()

    # Agregar la función f(x)

    x3 = np.linspace(aval, bval, 100)
    y3 = sp.lambdify(x,ff)
    fig3.add_trace(go.Scatter(x=x3, y=y3(x3), name='f(x)'))

    # Agregar los puntos del método de bisección
    fig3.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Puntos de aproximación',marker_color='rgba(152, 0, 0, .8)'))

    fig3.add_hline(0)
    # Establecer el título y etiquetas de los ejes
    fig3.update_layout(title='Método de Falsa Posición', xaxis_title='x', yaxis_title='f(x)')

    # Mostrar la figura
    st.plotly_chart(fig3)
except:
    st.write('')


