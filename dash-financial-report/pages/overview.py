import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from utils import Header, make_dash_table, make_dash_table2, transformaciones_dataf_mapacalor, transformaciones_dataf_hist, dataframe_toimage

import pandas as pd
import pathlib

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()


df_prueba = pd.read_csv(DATA_PATH.joinpath("melb_data.csv"))
df_fund_facts = pd.read_csv(DATA_PATH.joinpath("df_fund_facts.csv"))
df_price_perf = pd.read_csv(DATA_PATH.joinpath("df_price_perf.csv"))
df_prueba_10 = df_prueba.head(10)

transformaciones_dataf_mapacalor(df_prueba)
transformaciones_dataf_hist(df_prueba)
df_types = pd.DataFrame(df_prueba.dtypes)
dataframe_toimage(df_types, "dataframetypes")



def create_layout(app):
    # Page layouts
    return html.Div(
        [
            html.Div([Header(app)]),
            # page 1
            html.Div(
                [
                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H5("Introducción"),
                                    html.Br([]),
                                    html.P(
                                        "¿Qué es la minería de datos? Bueno, se puede definir como el proceso de obtener información oculta de los montones de bases de datos para propósitos de análisis. \
                                    La minería de datos también se conoce como descubrimiento de conocimientos en bases de KDD. No es más que extracción de datos de grandes bases para un trabajo especializado. \
                                    La minería de datos se utiliza en gran medida en diversas aplicaciones, como la comprensión del marketing de los consumidores, el análisis de productos, la demanda y el suministro, el comercio electrónico, la tendencia de inversión en acciones y bienes raíces, las telecomunicaciones, etc. \
                                    La minería de datos se basa en algoritmos matemáticos y habilidades analíticas para impulsar los resultados deseados de la enorme colección de bases de datos.\
                                    La minería de datos tiene gran importancia en el entorno empresarial altamente competitivo de hoy en día. Un nuevo concepto de la minería de datos de inteligencia de negocios ha evolucionado ahora, que es ampliamente utilizado por las principales casas corporativas para mantenerse por delante de sus competidores. \
                                    Inteligencia de negocio (BI) puede ayudar a proporcionar la última información y utilizado para el análisis de la competición, investigación de mercado, tendencias económicas, comportamiento del consumo, investigación de la industria, análisis de información geográfica y así sucesivamente. \
                                    La minería de datos de inteligencia empresarial ayuda en la toma de decisiones."
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
                    
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Tipos de Datos"], className="subtitle padded"
                                    ),
                                    html.Img(
                                      src=app.get_asset_url("Images.png"),
                                      className="mapadecalor",
                                    ),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Valores Nulos",
                                        className="subtitle padded",
                                    ),
                                    
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row",
                        style={"margin-bottom": "35px"},
                    ),

                       html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Datos"], className="subtitle padded"
                                    ),
                                    html.Table(make_dash_table2(df_prueba_10) ),
                                ],
                                style={"maxHeight": "400px", "overflow": "scroll"},
                            ),
                            
                        ],
                        className="row",
                        style={"margin-bottom": "35px"},
                    ),
                    # Row 5
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Hypothetical growth of $10,000",
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-2",
                                        figure={
                                            "data": [
                                                go.Scatter(
                                                    x=[
                                                        "2008",
                                                        "2009",
                                                        "2010",
                                                        "2011",
                                                        "2012",
                                                        "2013",
                                                        "2014",
                                                        "2015",
                                                        "2016",
                                                        "2017",
                                                        "2018",
                                                    ],
                                                    y=[
                                                        "10000",
                                                        "7500",
                                                        "9000",
                                                        "10000",
                                                        "10500",
                                                        "11000",
                                                        "14000",
                                                        "18000",
                                                        "19000",
                                                        "20500",
                                                        "24000",
                                                    ],
                                                    line={"color": "#97151c"},
                                                    mode="lines",
                                                    name="Calibre Index Fund Inv",
                                                )
                                            ],
                                            "layout": go.Layout(
                                                autosize=True,
                                                title="",
                                                font={"family": "Raleway", "size": 10},
                                                height=200,
                                                width=340,
                                                hovermode="closest",
                                                legend={
                                                    "x": -0.0277108433735,
                                                    "y": -0.142606516291,
                                                    "orientation": "h",
                                                },
                                                margin={
                                                    "r": 20,
                                                    "t": 20,
                                                    "b": 20,
                                                    "l": 50,
                                                },
                                                showlegend=True,
                                                xaxis={
                                                    "autorange": True,
                                                    "linecolor": "rgb(0, 0, 0)",
                                                    "linewidth": 1,
                                                    "range": [2008, 2018],
                                                    "showgrid": False,
                                                    "showline": True,
                                                    "title": "",
                                                    "type": "linear",
                                                },
                                                yaxis={
                                                    "autorange": False,
                                                    "gridcolor": "rgba(127, 127, 127, 0.2)",
                                                    "mirror": False,
                                                    "nticks": 4,
                                                    "range": [0, 30000],
                                                    "showgrid": True,
                                                    "showline": True,
                                                    "ticklen": 10,
                                                    "ticks": "outside",
                                                    "title": "$",
                                                    "type": "linear",
                                                    "zeroline": False,
                                                    "zerolinewidth": 4,
                                                },
                                            ),
                                        },
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Price & Performance (%)",
                                        className="subtitle padded",
                                    ),
                                    html.Table(make_dash_table(df_price_perf)),
                                ],
                                className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "Risk Potential", className="subtitle padded"
                                    ),
                                        html.Img(
                                      src=app.get_asset_url("Images_hist.png"),
                                      className="mapadecalor",

                                    ),
                                ],
                                className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
