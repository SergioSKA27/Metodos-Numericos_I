import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64
import struct
import math


file2_ = open("fes-acatlán-unam.gif", "rb")



contents2 = file2_.read()
data_url2 = base64.b64encode(contents2).decode("utf-8")
file2_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url2}" alt="logo-fes">',
    unsafe_allow_html=True,
)


r'''
# Índice

## 1. Introducción
   - 1.1 Errores de redondeo: aritmética del punto flotante, errores de truncamiento, absoluto y relativo
   - 1.2 Propagación del error en distintas operaciones aritméticas
   - 1.3 Orden de convergencia

## 2. Solución Numérica de Ecuaciones de una Sola Variable
   - 2.1 Método de bisección
   - 2.2 Método de falsa posición
   - 2.3 Método de Newton
   - 2.4 Método de la secante
   - 2.5 Método de Bairstow

## 3. Solución de Sistemas de Ecuaciones Lineales
   - 3.1 Condiciones necesarias y suficientes para la existencia de la solución de sistemas de ecuaciones lineales
      - 3.1.1 Inversión de matrices
      - 3.1.2 Método de intercambio
   - 3.2 Métodos exactos
      - 3.2.1 Método de Gauss y pivoteo parcial
      - 3.2.2 Método de Gauss-Jordan y pivoteo total
      - 3.3.3 Gauss-Jordan particionado
   - 3.4 Métodos iterativos
      - 3.4.1 Mejoramiento iterativo de la solución
      - 3.4.2 Método de Jacobi
      - 3.4.3 Método de Gauss-Seidel
      - 3.4.4 Método de relajación

## 4. Factorización LU y sus Aplicaciones
   - 4.1 Modelos de contexto y comportamiento
   - 4.2 Método de Cholesky
   - 4.3 Método Doolittle
   - 4.4 Solución de sistemas bandados (Método de Crout)

## 5. Cálculo de Valores y Vectores Propios
   - 5.1 Método de potencias
   - 5.2 Transformación de Householder
   - 5.3 Iteración QR



'''



file1_ = open("Logo.gif", "rb")



contents1 = file1_.read()
data_url1 = base64.b64encode(contents1).decode("utf-8")
file1_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url1}" alt="logo">',
    unsafe_allow_html=True,
)
