import streamlit as st
import numpy as np
import matplotlib as plt
import pandas as pd
import plotly.graph_objects as go
import sympy as sp

from st_pages import Page, show_pages, add_page_title

# Optional -- adds the title and icon to the current page
add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles and icons
#Page("Portada.py", "Índice", "🏠"),
# should be
show_pages(
    [
        Page("indice.py", "Índice", "🏠"),


        Page("Pages/Section1/Error_AbsR.py", "1.1 Errores de redondeo ", ":books:"),
        Page("Pages/Section1/ErrorProp.py", "1.2 Propagación del error  ", ":books:"),
        Page("Pages/Section1/convorder.py", "1.3 Orden de convergencia  ", ":books:"),

        Page("Pages/Section2/Bismethod.py", "2.1 Método de bisección ", ":books:"),
        Page("Pages/Section2/FalsePos.py", "2.2 Método de falsa posición ", ":books:"),
        Page("Pages/Section2/NewtonMethod.py", "2.3 Método de Newton ", ":books:"),
        Page("Pages/Section2/SecMethod.py", "2.4 Método de la secante ", ":books:"),
        Page("Pages/Section2/Bairstow.py", "2.5 Método de Bairstow", ":books:"),





    ]
)




