import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math
from streamlit_extras.stoggle import stoggle

st.title('2. Solución Numérica de Ecuaciones de una Sola Variable')



def f(x):
    return x**3 - 2*x - 5

def false_position_method(f,a, b, tolerance, max_iterations):
    """
    The false position method is a numerical method for finding the root of a function, and this Python function implements
    it with a given tolerance and maximum number of iterations.

    :param f: The function to find the root of
    :param a: The lower bound of the interval in which the root of the function is expected to be found
    :param b: The upper bound of the initial interval for the false position method
    :param tolerance: The desired level of accuracy or precision in the solution. The algorithm will stop iterating once the
    absolute value of the function evaluated at the current estimate is less than or equal to the tolerance value
    :param max_iterations: The maximum number of iterations that the false position method will perform before stopping,
    regardless of whether or not the desired tolerance has been reached
    :return: The function `false_position_method` returns three values: `x_values`, `y_values`, and `tab`. `x_values` is a
    list of the x-values of the approximations obtained by the false position method, `y_values` is a list of the
    corresponding y-values, and `tab` is a list of lists containing the values of `a`, `b`, `c`,
    """
    x_values = []
    y_values = []
    iterations = 0

    fx =sp.lambdify(list(f.free_symbols),f)
    tab = []

    while iterations < max_iterations:
        c = (a * fx(b) - b * fx(a)) / (fx(b) - fx(a))
        x_values.append(c)
        y_values.append(fx(c))

        tab.append([a,b,c,fx(c),str(abs(fx(c)) < tolerance)])

        if abs(fx(c)) < tolerance:
            break

        if fx(a) * fx(c) < 0:
            b = c
        else:
            a = c

        iterations += 1

    return x_values, y_values,tab


r'''
# 2.2 Método de falsa posición

El método de falsa posición es un algoritmo utilizado para encontrar aproximaciones de las raíces de una función en un intervalo dado. Es una variante del método de bisección que utiliza una interpolación lineal para aproximar la posición de la raíz. El método de falsa posición es iterativo y se basa en la idea de que una función continua cambiará de signo en un intervalo que contiene una raíz.

El algoritmo del método de falsa posición se puede resumir en los siguientes pasos:

- 1. Evaluación de la función en los extremos del intervalo:
   $$
   \begin{align*}
   f(a) \quad \text{y} \quad f(b)
   \end{align*}
   $$

- 2. Cálculo del punto de intersección de la recta secante:
   $$
   \begin{align*}
   c = \frac{{a \cdot f(b) - b \cdot f(a)}}{{f(b) - f(a)}}
   \end{align*}
   $$

- 3. Evaluación de la función en el punto \(c\):
   $$
   \begin{align*}
   f(c)
   \end{align*}
   $$


- 4. Verificación de los signos opuestos:
   $$
   \begin{align*}
   f(a) \cdot f(c) < 0
   \end{align*}
   $$

- 5. Se repiten los pasos 1-4 hasta que se alcance la precisión deseada o se agote el número máximo de iteraciones.

El método de falsa posición converge hacia la raíz de manera similar al método de bisección, pero puede ser más
eficiente en algunos casos, especialmente cuando la función cambia rápidamente cerca de la raíz. Sin embargo,
puede ser más lento en otros casos y puede sufrir de convergencia lenta si la función es muy plana en el intervalo.

La principal ventaja del método de falsa posición es que tiene una buena convergencia y garantiza que las aproximaciones
siempre se acerquen a la raíz. Sin embargo, puede requerir un mayor número de iteraciones que otros métodos más
sofisticados, como el método de Newton-Raphson.



## Algoritmo del Método de Falsa Posición

Dado un intervalo $[a, b]$ y una tolerancia $\varepsilon$:

1. Verificar que $f(a)$ y $f(b)$ tienen signos opuestos. Si no lo tienen, el método no se puede aplicar.
2. Inicializar $x_0 = a$ y $x_1 = b$.
3. Mientras la diferencia entre $x_1$ y $x_0$ sea mayor que $\varepsilon$, hacer:
   - Calcular el punto $x = x_1 - \frac{{f(x_1) \cdot (x_1 - x_0)}}{{f(x_1) - f(x_0)}}$.
   - Si $f(x) = 0$ o si la diferencia entre $x$ y $x_1$ es menor o igual que $\varepsilon$, retornar $x$ como la aproximación de la raíz.
   - Si $f(x)$ y $f(x_0)$ tienen signos opuestos, actualizar $x_1 = x$; de lo contrario, actualizar $x_0 = x$.
4. Retornar $x$ como la aproximación de la raíz.

El método de Falsa Posición (Regla de la Secante) es similar al Método de Bisección, pero en lugar de dividir el intervalo a la mitad, utiliza una interpolación lineal entre los puntos $x_0$ y $x_1$ para aproximar la raíz de la función $f(x)$. En cada iteración, se calcula un nuevo punto $x$ utilizando la fórmula de la secante.

La eficacia del método de Falsa Posición radica en su convergencia más rápida que el Método de Bisección, pero puede requerir más iteraciones para alcanzar la misma precisión. Es adecuado para funciones continuas y con cambios de signo en el intervalo considerado.

El algoritmo es relativamente simple de implementar y puede ser una alternativa eficiente al Método de Bisección en algunos casos. Sin embargo, al igual que con el Método de Bisección, se deben cumplir ciertos supuestos para garantizar su aplicabilidad y convergencia.

## Supuestos de aplicación

Los supuestos de aplicación del Método de Falsa Posición son los siguientes:

1. La función $f(x)$ es continua en el intervalo $[a, b]$: El Método de Falsa Posición requiere que la función sea continua en el intervalo considerado. Esto asegura que exista al menos una raíz en dicho intervalo.

2. La función $f(x)$ tiene un cambio de signo en el intervalo $[a, b]$: El método se basa en la propiedad de que si $f(a)$ y $f(b)$ tienen signos opuestos, entonces existe al menos una raíz en el intervalo. Si no hay cambio de signo, el método no puede determinar la ubicación de la raíz.

3. La función $f(x)$ es diferenciable en el intervalo $[a, b]$: Aunque el método no requiere la derivabilidad de la función, es deseable que la función sea diferenciable para obtener una convergencia más rápida y estable.

4. No hay puntos críticos en el intervalo: El método puede ser afectado por puntos críticos como máximos o mínimos locales, puntos de inflexión o discontinuidades en el intervalo. Estos puntos pueden interferir con la convergencia o hacer que el método no funcione correctamente.

Estos supuestos son importantes para garantizar la aplicabilidad y la convergencia del Método de Falsa Posición. Es fundamental tener en cuenta estas condiciones al utilizar el método y, en caso de que no se cumplan, considerar otras técnicas de búsqueda de raíces más adecuadas.

### :paperclip: Ejemplo

Sea

$$
\begin{align*}
f(x) = x^3-2x-5,
\end{align*}
$$

considerando el intervalo $[1,3]$, con un error de tolerancia $1 \times 10^{-6}$:



'''



# Definir intervalo y parámetros del método de falsa posición
a = 1
b = 3
tolerance = 1e-6
max_iterations = 100

# Ejecutar el método de falsa posición
x = sp.symbols('x')
x_values, y_values,tab1 = false_position_method(x**3-2*x-5,a, b, tolerance, max_iterations)

tabl1 = pd.DataFrame(tab1,columns=['a','b','c','f(c)','|f(c)| < 1e-6'])
st.write(tabl1)
# Graficar la función y el proceso de iteración

xs = np.linspace(a,b,100)
y = sp.lambdify(x,x**3-2*x-5)

fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=y(xs), name='Función f(x)'))
fig.add_hline(0)
fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', name='Iteraciones del método',marker_color='rgba(148, 0, 1, .9)'))
fig.update_layout(title='Método de Falsa Posición', xaxis_title='x', yaxis_title='f(x)')
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

st.write('Ingrese el intervalo $[a,b]$:')
aval = st.number_input('Ingrese $a$: ',-100,100,value=-1)
bval = st.number_input('Ingrese $b$: ',-100,100,value=1)
error = st.text_input('Ingrese el error de tolerancia: ',value='1e-6')

maxitr = st.number_input('Ingrese el máximo número de iteraciones: ',1,1000,value=10)


try:
    x_vals,y_vals,mtab = false_position_method(ff,aval,bval,float(error),maxitr)
    tpd2 = pd.DataFrame(mtab,columns=['a','b','c = a+b/2','f(c)','|f(c)| < '+str(error)])
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

stoggle('Implementación en Python',
r'''
    def false_position_method(f,a, b, tolerance, max_iterations):

        x_values = []
        y_values = []
        iterations = 0

        fx =sp.lambdify(list(f.free_symbols),f)

        while iterations < max_iterations:
            c = (a * fx(b) - b * fx(a)) / (fx(b) - fx(a))
            x_values.append(c)
            y_values.append(fx(c))

            if abs(fx(c)) < tolerance:
                break

            if fx(a) * fx(c) < 0:
                b = c
            else:
                a = c

            iterations += 1

        return x_values, y_values


'''
)
