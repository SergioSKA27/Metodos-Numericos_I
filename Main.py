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
        Page("Portada.py", "Índice", "🏠"),
        Page("Pages/Section1/Error_AbsR.py", "1.1 Errores de redondeo ", ":books:"),
    ]
)




