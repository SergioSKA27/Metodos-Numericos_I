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

    Donde $f(r, s)$ es el valor del polinomio evaluado en $r$ y $s$, $f_r$ y $f_s$ son los valores de las derivadas
    evaluadas en $r$ y $s$, respectivamente, y $\Delta r$, $\Delta s$, y $\Delta s_2$ son los incrementos en $r$, $s$,
    y $s^2$ respectivamente.

    c. Se actualizan los valores de $r$ y $s$ sumando los incrementos obtenidos en el paso anterior.

3. Se repite el proceso de iteración hasta alcanzar una convergencia deseada o un número máximo de iteraciones.

El método de Bairstow es especialmente útil para encontrar raíces complejas múltiples y raíces reales múltiples de un
polinomio. Sin embargo, es importante tener en cuenta que este método puede ser sensible a la elección inicial de las
raíces y puede requerir algunas iteraciones para obtener resultados precisos.

En resumen, el método de Bairstow es un algoritmo utilizado para encontrar las raíces de un polinomio. Es capaz de
encontrar tanto raíces reales como complejas y es especialmente útil para raíces múltiples. Aunque puede requerir
varias iteraciones y puede ser sensible a las elecciones iniciales, el método de Bairstow es una herramienta poderosa
en el campo del análisis numérico de polinomios.

## Algoritmo
El algoritmo del Método de Bairstow es el siguiente:

1. Ingresar los coeficientes del polinomio $a_n, a_{n-1}, ..., a_1, a_0$ y los valores iniciales de $r$ y $s$.
2. Ingresar la tolerancia $\epsilon$ para la condición de parada.
3. Ingresar el número máximo de iteraciones $N$.
4. Inicializar las variables $i = 0$ y $j = 0$.
5. Mientras $i < N$ y $j < N$:
   1. Calcular las siguientes iteraciones para $r$ y $s$ utilizando las fórmulas recursivas:
      \[r_{k+1} = \frac{-b_{k-1}b_k + b_{k-2}c_k}{b_k^2 - b_{k-1}c_{k+1}}\]
      \[s_{k+1} = \frac{-b_k^2 + b_{k-1}c_{k+1}}{b_k^2 - b_{k-1}c_{k+1}}\]
   2. Calcular los residuos $R_r$ y $R_s$ utilizando las fórmulas:
      \[R_r = a_{n-1} - b_{n-1}r - b_n s\]
      \[R_s = a_n - b_n r - b_{n+1} s\]
   3. Si $|R_r| \leq \epsilon$ y $|R_s| \leq \epsilon$, se ha encontrado una aproximación de las raíces.
   4. Incrementar $i$ y $j$ en 1.
6. Si se alcanza el número máximo de iteraciones $N$ sin converger, se detiene el algoritmo y se considera que no
se ha encontrado una aproximación de las raíces.

En cada iteración, el método utiliza las fórmulas recursivas para actualizar los valores de $r$ y $s$. Luego,
calcula los residuos para verificar la convergencia. Si los residuos son menores o iguales a la tolerancia
$\epsilon$, se considera que se ha encontrado una aproximación de las raíces. De lo contrario, se continúa
iterando hasta alcanzar el número máximo de iteraciones.

## Supuestos de aplicación

Los supuestos de aplicación del Método de Bairstow son los siguientes:

1. El polinomio debe ser de grado mayor o igual a 2: El Método de Bairstow está diseñado para encontrar
las raíces de polinomios de grado 2 o superior. No es aplicable a polinomios de grado 1 (lineales)
ya que su solución es directa.

2. Las raíces del polinomio deben ser reales o conjugadas: El Método de Bairstow se basa en el cálculo de raíces
reales o raíces complejas conjugadas en pares. Si el polinomio tiene raíces complejas no conjugadas, el método
puede no converger correctamente.

3. Se requieren valores iniciales adecuados: El método utiliza valores iniciales para las aproximaciones de las
raíces $r$ y $s$. Estos valores deben estar lo suficientemente cerca de las raíces reales para asegurar la convergencia
del método. Una mala elección de los valores iniciales puede resultar en una falta de convergencia o en la
convergencia a raíces incorrectas.

4. El polinomio debe tener coeficientes reales: El Método de Bairstow asume que los coeficientes del polinomio
son números reales. Si los coeficientes son complejos, se debe adaptar el algoritmo para trabajar con números complejos.

Estos supuestos son importantes para garantizar la aplicabilidad y la convergencia del Método de Bairstow.
Si alguno de estos supuestos no se cumple, el método puede no funcionar correctamente. Por lo tanto, es fundamental
considerar estos supuestos y evaluar la idoneidad del método en función de las características del polinomio en cuestión.


### Ejemplo

A continuación se presenta un ejemplo del Método de Bairstow aplicado a un polinomio:

Supongamos que queremos encontrar las raíces del siguiente polinomio de grado 3:

$$p(x) = 2x^3 - 5x^2 + 3x - 1$$

Usaremos valores iniciales $r_0 = 1$ y $s_0 = -1$ para las raíces aproximadas.

**Paso 1:** Inicialización
- Grado del polinomio $n = 3$
- Coeficientes del polinomio $a = [2, -5, 3, -1]$
- Valores iniciales para las raíces aproximadas $r_0 = 1$ y $s_0 = -1$
- Tolerancia deseada $\epsilon = 0.0001$

**Paso 2:** Creación de las listas de coeficientes iniciales
- `b = [2, -5, 3]`
- `c = [0, 0, 0]`

**Paso 3:** Iteración
En cada iteración, calcularemos los coeficientes `d` y `e`, las correcciones `dr` y `ds`, y actualizaremos las raíces
aproximadas `r` y `s` hasta alcanzar la convergencia.

- **Iteración 1:**
  - `d = [2, -3, 6]`
  - `e = [0, 0, 0]`
  - `D = 0`
  - `dr = 0`
  - `ds = 0`
  - `r = 1`
  - `s = -1`

- **Iteración 2:**
  - `d = [2, -3, 6]`
  - `e = [0, 0, 0]`
  - `D = 0`
  - `dr = 0`
  - `ds = 0`
  - `r = 1`
  - `s = -1`

La iteración se detiene ya que no se produce ninguna corrección en las raíces aproximadas y no se cumple el criterio
de convergencia.

**Paso 4:** Salida
Las raíces aproximadas obtenidas son $r = 1$ y $s = -1$.

Es importante destacar que este es un ejemplo sencillo con un polinomio de grado 3 y valores iniciales adecuados,
por lo que se alcanza la convergencia en la primera iteración. En casos más complejos, puede requerirse un mayor número
de iteraciones para obtener una convergencia satisfactoria.
'''



