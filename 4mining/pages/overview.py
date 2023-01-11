import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from utils import Header, make_dash_table, make_dash_table2, transformaciones_dataf_mapacalor, transformaciones_dataf_hist, dataframe_toimage
from utils import transforminfoto_todataframe

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
df_types.columns = ['value']
dataframe_toimage(df_types, "dataframetypes")
df_null = pd.DataFrame(df_prueba.isnull().sum())
df_null.columns = ['Suma valores nulos']
df_descr = pd.DataFrame(df_prueba.describe())
df_descr_ob = pd.DataFrame(df_prueba.describe(include='object'))
dataframe_toimage(df_null, "dataframenull")
dataframe_toimage(df_descr, "dataframedescribe")
dataframe_toimage(df_descr_ob, "dataframeobject")
transforminfoto_todataframe(df_prueba)



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
                    
                       
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
