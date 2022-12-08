import dash_html_components as html
import dash_core_components as dcc
import matplotlib.pyplot as plt
import dataframe_image as dfi
import numpy as np
import pandas as pd

import seaborn as sns 



def transformaciones_dataf_mapacalor(df):
    #plt.hist(df)
    #plt.savefig('Images.png')
    corrdf = df.corr()
    sns.heatmap(corrdf, cmap='RdBu_r', annot=True)
    plt.savefig('assets/Images.png')


def transformaciones_dataf_hist(df):
    df.hist(figsize=(14,14), xrot=45)
    plt.savefig('assets/Images_hist.png')

def dataframe_toimage(df, name):
    df_styled = df.style.background_gradient() #adding a gradient based on values in cell
    dfi.export(df_styled, "assets/"+name+".png")


def Header(app):
    return html.Div([get_header(app), html.Br([]), get_menu(), get_file()])


def get_header(app):
    header = html.Div(
        [
            html.Div(
                [
                    html.A(
                        html.Img(
                            src=app.get_asset_url("FORMINNER.png"),
                            className="logo",
                        ),
                        href="https://plotly.com/dash",
                    ),
                    html.A(
                        html.Button(
                            "Documentación",
                            id="learn-more-button",
                            style={"margin-left": "-10px"},
                        ),
                        href="https://plotly.com/get-demo/",
                    ),
                    html.A(
                        html.Button("Source Code", id="learn-more-button"),
                        href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-financial-report",
                    ),
                ],
                className="row",
            ),
            html.Div(
                [
                    html.Div(
                        [html.H5("4")],
                        id="numbert", className="two columns",
                    ),
                    html.Div(
                        [html.H5("MINNING")],
                        className="five columns main-title",
                    ),
                    html.Div(
                        [
                            dcc.Link(
                                "Full View",
                                href="/dash-financial-report/full-view",
                                className="full-view-link",
                            )
                        ],
                        className="five columns",
                    ),
                ],
                className="twelve columns",
                style={"padding-left": "0"},
            ),
        ],
        className="row",
    )
    return header


def get_menu():
    menu = html.Div(
        [
            dcc.Link(
                "Vista General",
                href="/dash-financial-report/overview",
                className="tab first",
            ),
            dcc.Link(
                "EDA",
                href="/dash-financial-report/price-performance",
                className="tab",
            ),
            dcc.Link(
                "PCA",
                href="/dash-financial-report/portfolio-management",
                className="tab",
            ),
            dcc.Link(
                "Bosques y Arboles Aleatorios", href="/dash-financial-report/fees", className="tab"
            ),
            dcc.Link(
                "Segmentación y Clasificación",
                href="/dash-financial-report/distributions",
                className="tab",
            ),
            dcc.Link(
                "Regresión Lineal",
                href="/dash-financial-report/news-and-reviews",
                className="tab",
            ),
        ],
        className="row all-tabs",
    )
    return menu

def get_file():
    getfile = html.Div([
                dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Selecciona una fuente de datos (Formato .xslx, .csv) ',
                    html.A('Selecionar Datos (Unicamente uno a la vez)')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
            html.Div(id='output-data-upload'),
        ])
    return getfile

 


def make_dash_table(df):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table

def make_dash_table2(df):
    html_row = [html.Td([col]) for col in df.columns]
    table = [html.Tr(html_row)]
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table


def min_max_scaling(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

    return df_norm
