import base64
import datetime
import io
from msilib.schema import Component
import dash_bootstrap_components as dbc
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib         
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, Input, Output, callback
import plotly.express as px
from utils import Header, Header3
import plotly.graph_objs as go         # Para la visualización de datos basado en plotly
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from dash_bootstrap_templates import load_figure_template,ThemeChangerAIO, template_from_url

# load_figure_template("plotly_white")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
theme_change = ThemeChangerAIO(
    aio_id="theme",button_props={
        "color": "danger",
        "children": "SELECT THEME",
        "outline": True,
    },
    radio_props={
        "persistence": True,
    },
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


layout = html.Div([
    Header(app),
    Header3(app),
    html.Div(
                [
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Bosques Aleatorios (Random Forests)"),
                                    html.Br([]),
                                    html.P(
                                        
                                        "Un random forest (o bosque aleatorio en español) es una técnica de Machine Learning muy popular \
                                        entre los Data Scientist y con razón : presenta muchas ventajas en comparación con otros algoritmos de datos. \
                                        Es una técnica fácil de interpretar, estable, que por lo general presenta buenas coincidencias y \
                                        que se puede utilizar en tareas de regresión o de clasificación. Cubre, por tanto, gran parte de los \
                                        problemas de Machine Learning. \
                                        En random forest primero encontramos la palabra “forest” (bosque, en español). Se entiende que ese  \
                                        algoritmo se basa en árboles que se suelen llamar árboles de decisión."
                                    ,
                                        style={"color": "#ffffff", "font-size": "10px", "font-family": "Mysecondfont", "text-align": "justify", "line-height": "1.8"},
                                        className="row",
                                    ),
                                ],
                                className="product",
                            )
                        ],
                        className="row",
                    ),
                    
                       
                ],
                className="sub_page",
            ),
    ], className="page")