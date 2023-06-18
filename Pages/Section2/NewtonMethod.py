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



def dwf(x):
    return x**3 - 2*x - 5

def dddf(x):
    return 3*x**2 - 2

def newton_method(f,x0, tolerance, max_iterations):
    """
    The function implements the Newton-Raphson method for finding roots of a function with a given tolerance and maximum
    number of iterations.

    :param f: The function for which we want to find the root using Newton's method
    :param x0: The initial guess for the root of the function
    :param tolerance: The desired level of accuracy or closeness to the true solution. The algorithm will stop iterating
    once the absolute value of the function evaluated at the current guess is less than or equal to the tolerance value
    :param max_iterations: The maximum number of iterations the function will perform before stopping, regardless of whether
    the desired tolerance has been reached or not
    :return: three lists: x_values, y_values, and tab. x_values contains the x values at each iteration, y_values contains
    the corresponding y values, and tab contains a table with columns for x, f(x), x - f(x)/f'(x), and a boolean indicating
    whether abs(f(x)) < tolerance.
    """
    x_values = []
    y_values = []
    iterations = 0
    tab = []

    fx = sp.lambdify(list(f.free_symbols),f)
    dfx = sp.lambdify(list(f.free_symbols),sp.diff(f))

    while iterations < max_iterations:
        x_values.append(x0)
        y_values.append(fx(x0))
        tab.append([x0,fx(x0),x0 - fx(x0) / dfx(x0),str(abs(fx(x0)) < tolerance)])

        if abs(fx(x0)) < tolerance:
            break

        x1 = x0 - fx(x0) / dfx(x0)
        x0 = x1

        iterations += 1

    return x_values, y_values,tab







st.title('2. Solución Numérica de Ecuaciones de una Sola Variable')







r'''
# 2.3 Método de Newton

El método de Newton, también conocido como el método de Newton-Raphson, es un algoritmo iterativo utilizado para
encontrar raíces de una función. Este método es altamente eficiente cuando se cuenta con una buena estimación inicial
y la función es suficientemente suave.

El algoritmo del método de Newton se puede resumir en los siguientes pasos:

- 1. Dado un punto inicial $x_0$ cercano a la raíz de la función $f(x)$, se calcula la pendiente de la función en
ese punto, es decir, su derivada $f'(x_0)$.

- 2. Se calcula la recta tangente a la curva de la función en el punto $(x_0, f(x_0))$. Esta recta intersecta el
eje $x$ en el punto $x_1$, que se convierte en una mejor aproximación de la raíz.

- 3. Se repiten los pasos 1 y 2 utilizando $x_1$ como el nuevo punto inicial, calculando su derivada $f'(x_1)$ y
encontrando la nueva aproximación $x_2$ mediante la intersección de la recta tangente.

- 4. Se continúa este proceso de forma iterativa hasta alcanzar una aproximación deseada o agotar un número máximo de iteraciones.

La fórmula general para la iteración del método de Newton es:

$$
\begin{align*}
x_{n+1} = x_n - \frac{{f(x_n)}}{{f'(x_n)}}
\end{align*}
$$

El método de Newton converge rápidamente hacia la raíz si la estimación inicial es cercana a la raíz y si la función
es suficientemente suave. La convergencia del método está determinada por la tasa de cambio de la función en las
proximidades de la raíz. Si la derivada es grande en ese punto, la convergencia será más rápida.

Es importante tener en cuenta que el método de Newton requiere el cálculo de la derivada de la función en cada iteración.
En algunos casos, esta derivada puede ser difícil o costosa de obtener analíticamente. En tales situaciones, se pueden
utilizar aproximaciones numéricas de la derivada, como el método de diferencias finitas, para calcularla.

Sin embargo, el método de Newton también tiene algunas limitaciones. Puede haber casos en los que el método no converge
o se estanque en mínimos locales o puntos de inflexión. Además, el método puede ser sensible a la elección de la estimación
inicial, y diferentes estimaciones pueden conducir a raíces diferentes o a la no convergencia del método.


## Algoritmo del Método de Newton

Dado un punto inicial $x_0$ y una tolerancia $\varepsilon$:

1. Inicializar $x = x_0$.
2. Mientras $|f(x)| > \varepsilon$, hacer:
   - Calcular la derivada de $f(x)$, $f'(x)$.
   - Calcular la corrección $\Delta x = -\frac{f(x)}{f'(x)}$.
   - Actualizar $x = x + \Delta x$.
3. Retornar $x$ como la aproximación de la raíz.

El Método de Newton-Raphson es un método iterativo utilizado para encontrar aproximaciones de raíces de una
función. El método se basa en la idea de aproximar la función por una recta tangente en cada iteración y encontrar
el punto en el cual la recta tangente cruza el eje x, que se considera una mejor aproximación de la raíz.

En cada iteración, se calcula la derivada de la función en el punto actual y se determina la corrección $\Delta x$
necesaria para acercarse a la raíz. Luego, se actualiza el valor de $x$ sumando la corrección.
Este proceso se repite hasta que se alcance una precisión deseada, determinada por la tolerancia $\varepsilon$.

El Método de Newton-Raphson es conocido por su convergencia rápida cuando se parte de un punto cercano a la raíz y
cuando la función es diferenciable y tiene una derivada no nula en la vecindad de la raíz. Sin embargo, es importante
tener en cuenta que el método puede no converger o puede converger a una raíz diferente si no se cumplen ciertas condiciones.

El algoritmo proporcionado es una implementación básica del Método de Newton-Raphson. Se parte de un punto inicial
$x_0$ y se realizan iteraciones hasta alcanzar la precisión deseada. El valor obtenido en la última iteración se retorna
como la aproximación de la raíz.

Es importante destacar que el Método de Newton-Raphson puede requerir un análisis adicional para asegurar su aplicabilidad
y convergencia, como la elección adecuada del punto inicial y la verificación de las condiciones necesarias para la
convergencia del método.


## Supuestos de aplicación

Los supuestos de aplicación del Método de Newton-Raphson son los siguientes:

1. La función $f(x)$ es continuamente diferenciable: El método se basa en el cálculo de la derivada de la función en
cada iteración. Por lo tanto, se requiere que la función $f(x)$ sea diferenciable en el intervalo considerado.
Esto asegura que la derivada exista y se pueda calcular en cada punto necesario.

2. La derivada $f'(x)$ no es igual a cero en el intervalo: El método utiliza la derivada de la función para calcular
la corrección $\Delta x$ en cada iteración. Para evitar divisiones por cero y asegurar una convergencia adecuada,
se asume que la derivada $f'(x)$ no es igual a cero en el intervalo considerado.

3. Se elige un punto inicial cercano a la raíz: El Método de Newton-Raphson converge más rápidamente cuando se parte de
un punto inicial cercano a la raíz. Por lo tanto, se asume que se ha seleccionado un punto inicial adecuado que se
encuentra en la vecindad de la raíz buscada.

Estos supuestos son importantes para garantizar la aplicabilidad y la convergencia del Método de Newton-Raphson.
Si alguno de estos supuestos no se cumple, el método puede no converger o puede converger a una raíz incorrecta.
Por lo tanto, es fundamental considerar estos supuestos al utilizar el método y, en caso de no cumplirse, evaluar
otras técnicas de búsqueda de raíces más apropiadas.


###  :paperclip: Ejemplo

Sea

$$
\begin{align*}
f(x) = x^3-2x-5,
\end{align*}
$$

con la aproximación inicial $x_0 = 2.5$, con un error de tolerancia $1 \times 10^{-6}$:

'''


# Definir punto inicial y parámetros del método de Newton
x0 = 2.5
tolerance = 1e-6
max_iterations = 10
x = sp.symbols('x')

# Ejecutar el método de Newton
x_values, y_values,tab = newton_method(x**3-2*x-5,x0, tolerance, max_iterations)

# Graficar la función y el proceso de iteración
xs = np.linspace(-5, 5, 100)
y = sp.lambdify(x,x**3-2*x-5)

fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=y(xs), name='Función f(x)'))
fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', name='Iteraciones del método',marker_color='rgba(148, 0, 1, .9)'))
fig.update_layout(title='Método de Newton', xaxis_title='x', yaxis_title='f(x)')

fig.add_hline(0)
st.plotly_chart(fig)




st.subheader('Método :triangular_ruler:')

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


xini = st.number_input('Ingrese $x_0$: ',-100,100,value=-1)

error = st.text_input('Ingrese el error de tolerancia: ',value='1e-6')

maxitr = st.number_input('Ingrese el máximo número de iteraciones: ',1,1000,value=10)


try:
    x_vals,y_vals,mtab = newton_method(ff,xini,float(error),maxitr)
    tpd2 = pd.DataFrame(mtab,columns=['x_n','f(x_n)','x_n+1','|f(c)| < '+str(error)])
    st.write(tpd2)
    # Crear una figura de Plotly
    fig3 = go.Figure()

    # Agregar la función f(x)

    x3 = np.linspace(xini-10, xini+10, 100)
    y3 = sp.lambdify(x,ff)
    fig3.add_trace(go.Scatter(x=x3, y=y3(x3), name='f(x)'))

    # Agregar los puntos del método de bisección
    fig3.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Puntos de aproximación',marker_color='rgba(152, 0, 0, .8)'))

    fig3.add_hline(0)
    # Establecer el título y etiquetas de los ejes
    fig3.update_layout(title='Método de Newton', xaxis_title='x', yaxis_title='f(x)')

    # Mostrar la figura
    st.plotly_chart(fig3)
except:
    st.write('')

stoggle('Implementación en Python',
r'''
    def newton_method(f,x0, tolerance, max_iterations):

        x_values = []
        y_values = []
        iterations = 0

        fx = sp.lambdify(list(f.free_symbols),f)
        dfx = sp.lambdify(list(f.free_symbols),sp.diff(f))

        while iterations < max_iterations:
            x_values.append(x0)
            y_values.append(fx(x0))

            if abs(fx(x0)) < tolerance:
                break

            x1 = x0 - fx(x0) / dfx(x0)
            x0 = x1

            iterations += 1

        return x_values, y_values

'''
)
