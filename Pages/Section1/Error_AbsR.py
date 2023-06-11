import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math


def calcular_error_absoluto(valor_real, valor_aproximado):
    return abs(valor_real - valor_aproximado)

def calcular_error_relativo(valor_real, valor_aproximado):
    return abs(valor_real - valor_aproximado) / abs(valor_real)

def float_to_binary(num, bits):
    if bits == 32:
        packed = struct.pack('!f', num)
    elif bits == 64:
        packed = struct.pack('!d', num)
    else:
        st.write("Bits no válidos. Por favor, seleccione 32 o 64.")
        return None

    ints = [i for i in packed]
    binary = [f'{i:08b}' for i in ints]

    return ''.join(binary)


def redondear(numero, decimales):
    factor = 10 ** decimales
    return round(numero * factor) / factor

def truncar(numero, decimales):
    factor = 10 ** decimales
    return math.trunc(numero * factor) / factor

def valor_absoluto(numero):
    return abs(numero)


st.title('UNIDAD 1 ANÁLISIS DE ERROR')

r'''
## 1.1 Errores de redondeo: aritmética del punto flotante, errores de truncamiento, absoluto y relativo

'''


r'''
### Aritmética del punto flotante

La aritmética del punto flotante es un método utilizado para representar y realizar operaciones aritméticas con números
reales en las computadoras. En este sistema, los números se expresan en una forma decimal aproximada mediante una
mantisa y un exponente. Sin embargo, debido a las limitaciones de la representación finita de los números en la
computadora, los cálculos con números de punto flotante están sujetos a errores de redondeo y pérdida de precisión.

En la representación de punto flotante, los números se expresan en base 2, utilizando una mantisa normalizada y un
exponente binario. La mantisa representa la parte fraccionaria del número, mientras que el exponente indica la posición
del punto decimal. Esta representación permite manejar un amplio rango de números, pero no puede representar todos los
números reales de manera exacta debido a la finitud de los dígitos disponibles.

La aritmética de punto flotante involucra operaciones como la suma, resta, multiplicación y división de números de punto
flotante. Estas operaciones se realizan considerando los dígitos significativos de la mantisa y ajustando el exponente
para mantener una representación normalizada. Sin embargo, debido a la limitada precisión de los números de punto
flotante, las operaciones aritméticas pueden generar errores de redondeo.

Los errores de redondeo pueden ocurrir en diferentes etapas de las operaciones aritméticas. Durante la suma o resta,
por ejemplo, puede haber una pérdida de dígitos significativos debido a la diferencia en las magnitudes de los números
a sumar o restar. En la multiplicación y división, la pérdida de precisión puede ocurrir debido a la limitada cantidad
de dígitos disponibles para representar los resultados.

Además de los errores de redondeo, la aritmética del punto flotante también está sujeta a problemas como el
desbordamiento y el subdesbordamiento. El desbordamiento ocurre cuando el resultado de una operación es mayor al valor
máximo que se puede representar, mientras que el subdesbordamiento ocurre cuando el resultado es menor al valor mínimo
representable.

Es importante tener en cuenta que los errores de redondeo y otros problemas asociados con la aritmética del punto
flotante no son exclusivos de las computadoras, sino que son inherentes a cualquier sistema de representación numérica finita.


'''


file1_ = open("Pages/Section1/Fig1.png", "rb")



contents1 = file1_.read()
data_url1 = base64.b64encode(contents1).decode("utf-8")
file1_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url1}" alt="FloatRep">',
    unsafe_allow_html=True,
)




file2_ = open("Pages/Section1/Fig2.png", "rb")



contents2 = file2_.read()
data_url2 = base64.b64encode(contents2).decode("utf-8")
file2_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url2}" alt="FloatRep32">',
    unsafe_allow_html=True,
)

r'''

### Errores de redondeo

Los errores de redondeo son un fenómeno inherente a la aritmética del punto flotante, utilizada en la representación y
manipulación de números reales en sistemas computacionales. A pesar de los avances en el diseño de algoritmos y
hardware, estos errores persisten debido a las limitaciones en la precisión de los números de punto flotante y a las
operaciones aritméticas realizadas sobre ellos.

El **error de truncamiento** es uno de los errores de redondeo más comunes. Ocurre cuando se aproxima un número de punto
flotante al redondearlo a una cantidad finita de dígitos significativos. Esta aproximación puede provocar una pérdida de
precisión, ya que los dígitos que quedan fuera de la representación truncada se descartan. En operaciones aritméticas
sucesivas, el error de truncamiento puede acumularse y tener un impacto significativo en el resultado final.

Para evaluar la precisión de un cálculo numérico, es útil considerar tanto el **error absoluto** como el
**error relativo**. El error absoluto se define como la diferencia entre el valor aproximado y el valor real de un número:

$$
\begin{align*}
E_{\text{abs}} = |x - \hat{x}|
\end{align*}
$$

donde $x$ es el valor real y $\hat{x}$ es el valor aproximado. El error absoluto representa la discrepancia directa
entre la aproximación y el valor verdadero. Cuanto menor sea el valor del error absoluto, mayor será la precisión de
la aproximación.

El error relativo, por otro lado, proporciona una medida de la precisión relativa del resultado obtenido.
Se calcula dividiendo el error absoluto por el valor real:

$$
\begin{align*}
E_{\text{rel}} = \left|\frac{{x - \hat{x}}}{{x}}\right|
\end{align*}
$$

El error relativo toma en cuenta el tamaño del número real y proporciona una medida de la magnitud del error en relación
con el valor real. Es particularmente útil cuando se comparan resultados que involucran números de diferentes magnitudes.
Un error relativo pequeño indica una mayor precisión en el cálculo.

Es importante destacar que tanto el error absoluto como el error relativo dependen del contexto y de la escala de los
números involucrados. En algunos casos, un error absoluto pequeño puede considerarse aceptable, mientras que en otros
casos, un error relativo pequeño puede ser más relevante. La elección de la medida de error apropiada depende de la
aplicación y de los requisitos específicos del problema.

'''




st.title("Medición de error absoluto y relativo")

valor_reals = st.text_input("Ingrese el valor real:")
valor_aproximados = st.text_input("Ingrese el valor aproximado:")

if st.button("Calcular",key='input1'):
    valor_real = float(sp.parse_expr(valor_reals))
    valor_aproximado = float(sp.parse_expr(valor_aproximados))
    error_absoluto = calcular_error_absoluto(valor_real, valor_aproximado)
    error_relativo = calcular_error_relativo(valor_real, valor_aproximado)

    st.write(f"El error absoluto es: {error_absoluto}")
    st.write(f"El error relativo es: {error_relativo}")




st.title("Representación en Punto Flotante")

num = st.number_input("Ingrese el número:")
bits = st.radio("Seleccione el número de bits:", (32, 64))

if st.button("Calcular",key='input2'):
    binary_representation = float_to_binary(num, bits)

    if binary_representation:
        st.write(f"Representación en binario de {num} en {bits} bits:")
        st.write(binary_representation)


st.title("Operaciones matemáticas")

numeros = st.text_input("Ingrese un número:",'0')
decimales = st.number_input("Número de decimales:", min_value=0, max_value=10, step=1)

numero = float(sp.parse_expr(numeros))
if st.button("Redondear"):
    resultado = redondear(numero, decimales)
    st.write(f"El número redondeado es: {resultado}")

if st.button("Truncar"):
    resultado = truncar(numero, decimales)
    st.write(f"El número truncado es: {resultado}")

if st.button("Valor absoluto"):
    resultado = valor_absoluto(numero)
    st.write(f"El valor absoluto es: {resultado}")
