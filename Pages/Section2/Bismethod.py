import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math



def bisection_method(f,a, b, tolerance, max_iterations):
    x_values = []
    y_values = []

    fx = sp.lambdify(list(f.free_symbols),f)
    tab = []

    iteration = 1
    while iteration <= max_iterations:
        c = (a + b) / 2
        x_values.append(c)
        y_values.append(fx(c))

        tab.append([a,b,c,fx(c),str(abs(fx(c)) < tolerance)])

        if abs(fx(c)) < tolerance:
            break

        if fx(a) * fx(c) < 0:
            b = c
        else:
            a = c

        iteration += 1

    return x_values, y_values, tab



st.title('2. Solución Numérica de Ecuaciones de una Sola Variable')




r'''
## 2.1 Método de Bisección

El método de bisección es un algoritmo de búsqueda incremental utilizado para encontrar las raíces de una función en un
intervalo dado. Este método es iterativo y se basa en el teorema del valor intermedio.

El objetivo principal del método de bisección es reducir el intervalo inicial en el que se encuentra la raíz de la
función hasta alcanzar una precisión deseada. El algoritmo funciona de la siguiente manera:

- 1. Dado un intervalo inicial $[a, b]$ donde se espera encontrar una raíz y una función $f(x)$ continua en ese intervalo.
- 2. Se calcula el punto medio $c$ del intervalo:

$$
\begin{align*}
 c = \frac{{a + b}}{2}
\end{align*}
$$

- 3. Se evalúa el valor de la función en el punto medio:

$$
\begin{align*}
f(c)
\end{align*}
$$

- 4. Si $f(c)$ es cercano a cero o aproximadamente cero (dentro de una tolerancia establecida), se considera $c$ como una
aproximación de la raíz y se termina el algoritmo.
- 5. De lo contrario, se verifica en qué mitad del intervalo, $[a, c]$ o $[c, b]$, existe un cambio de signo de la función.
- 6. Se reemplaza el extremo correspondiente del intervalo con el valor de $c$, manteniendo el extremo que tiene el mismo
signo que $f(c)$.
- 7. Se repiten los pasos 2 a 6 hasta que se alcance la precisión deseada.

El método de bisección es relativamente sencillo y garantiza la convergencia hacia la raíz de la función,
siempre y cuando la función sea continua en el intervalo dado y haya un cambio de signo.
Sin embargo, puede requerir un número considerable de iteraciones para alcanzar la precisión deseada,
especialmente si la función tiene una curva suave o múltiples raíces en el intervalo.



'''
file1_ = open("Pages/Section2/Fig1_Bis.png", "rb")



contents1 = file1_.read()
data_url1 = base64.b64encode(contents1).decode("utf-8")
file1_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url1}" alt="biseccionimg">',
    unsafe_allow_html=True,
)



r'''
###  :paperclip: Ejemplo

Sea

$$
\begin{align*}
f(x) = x^3-2x-5
\end{align*}
$$

en el intervalo $[1,3]$, con un error de tolerancia $1 \times 10^{-6}$:
'''



# Parámetros del método de bisección
a = 1
b = 3
tolerance = 1e-6
max_iterations = 20
x = sp.symbols('x')
# Ejecutar el método de bisección
x_values, y_values,t = bisection_method(x**3-2*x-5,a, b, tolerance, max_iterations)
tpd = pd.DataFrame(t,columns=['a','b','c = a+b/2','f(c)','|f(c)| < '+str(tolerance)])
st.write(tpd)
# Crear una figura de Plotly
fig = go.Figure()

# Agregar la función f(x)

xs = np.linspace(a, b, 100)
y = sp.lambdify(x,x**3-2*x-5)
fig.add_trace(go.Scatter(x=xs, y=y(xs), name='f(x)'))

# Agregar los puntos del método de bisección
fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', name='Puntos de bisección',marker_color='rgba(148, 0, 1, .9)'))

fig.add_hline(0)
# Establecer el título y etiquetas de los ejes
fig.update_layout(title='Método de Bisección', xaxis_title='x', yaxis_title='f(x)')

# Mostrar la figura
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

st.write('Ingrese el intervalo $[a,b]$:')
aval = st.number_input('Ingrese $a$: ',-100,100,value=-1)
bval = st.number_input('Ingrese $b$: ',-100,100,value=1)
error = st.text_input('Ingrese el error de tolerancia: ',value='1e-6')

maxitr = st.number_input('Ingrese el máximo número de iteraciones: ',1,1000,value=10)


try:
    x_vals,y_vals,mtab = bisection_method(ff,aval,bval,float(error),maxitr)
    tpd2 = pd.DataFrame(mtab,columns=['a','b','c = a+b/2','f(c)','f(c) < '+str(error)])
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
    fig3.update_layout(title='Método de Bisección', xaxis_title='x', yaxis_title='f(x)')

    # Mostrar la figura
    st.plotly_chart(fig3)
except:
    st.write('')
