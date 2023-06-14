import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import base64

from st_pages import Page, show_pages, add_page_title

# Optional -- adds the title and icon to the current page
#add_page_title()
st.header('MÉTODOS NÚMERICOS I')
st.subheader('Autor: Lopez Martinez Sergio Demis')
file1_ = open("Mainfig.gif", "rb")



contents1 = file1_.read()
data_url1 = base64.b64encode(contents1).decode("utf-8")
file1_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url1}" alt="cool-animation">',
    unsafe_allow_html=True,
)




file1_ = open("Fig1.gif", "rb")



contents1 = file1_.read()
data_url1 = base64.b64encode(contents1).decode("utf-8")
file1_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url1}" alt="cool-animation">',
    unsafe_allow_html=True,
)



# Specify what pages should be shown in the sidebar, and what their titles and icons
#Page("Portada.py", "Índice", "🏠"),
# should be
show_pages(
    [
        Page("indice.py", "Índice", "🏠"),


        Page("Pages/Section1/Error_AbsR.py", "1.1 Errores de redondeo ", ":books:"),
        Page("Pages/Section1/ErrorProp.py", "1.2 Propagación del error  ", ":books:"),
        Page("Pages/Section1/convorder.py", "1.3 Orden de convergencia  ", ":books:"),

        Page("Pages/Section2/Bismethod.py", "2.1 Método de bisección ", ":bookmark_tabs:"),
        Page("Pages/Section2/FalsePos.py", "2.2 Método de falsa posición ", ":bookmark_tabs:"),
        Page("Pages/Section2/NewtonMethod.py", "2.3 Método de Newton ", ":bookmark_tabs:"),
        Page("Pages/Section2/SecMethod.py", "2.4 Método de la secante ", ":bookmark_tabs:"),
        Page("Pages/Section2/Bairstow.py", "2.5 Método de Bairstow", ":bookmark_tabs:"),



        Page("Pages/Section3/Conditions.py", "3.1 Condiciones necesarias y suficientes", ":notebook_with_decorative_cover:"),
        Page("Pages/Section3/matrixinv.py", "3.1.1 Inversión de matrices", ":notebook_with_decorative_cover:"),
        Page("Pages/Section3/intercambioMethod.py", "3.1.2 Método de intercambio", ":notebook_with_decorative_cover:"),
        Page("Pages/Section3/GaussPartial.py", "3.2.1 Método de Gauss y pivoteo parcial ", ":notebook_with_decorative_cover:"),
        Page("Pages/Section3/GaussTotal.py", "3.2.2 Método de Gauss-Jordan y pivoteo total", ":notebook_with_decorative_cover:"),
        Page("Pages/Section3/GaussPartition.py", "3.3.3 Gauss-Jordan particionado", ":notebook_with_decorative_cover:"),
        Page("Pages/Section3/IterativeImprove.py", "3.4.1 Mejoramiento iterativo de la solución", ":notebook_with_decorative_cover:"),
        Page("Pages/Section3/Jacobi.py", "3.4.2 Método de Jacobi", ":notebook_with_decorative_cover:"),
        Page("Pages/Section3/GaussSeidel.py", "3.4.3 Método de Gauss-Seidel", ":notebook_with_decorative_cover:"),
        Page("Pages/Section3/Relajacion.py", "3.4.4 Método de relajación", ":notebook_with_decorative_cover:"),



    ]
)




