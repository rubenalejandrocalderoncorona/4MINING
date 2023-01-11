import dash_html_components as html
import dash_core_components as dcc
import matplotlib.pyplot as plt
import dataframe_image as dfi
import io
import numpy as np
import pandas as pd
import base64

import seaborn as sns 

global df




def transforminfoto_todataframe(df):
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    lines = [line.split() for line in s.splitlines()[3:-2]]
    df_info = pd.DataFrame(lines)
    df_styled = df_info.style.background_gradient() #adding a gradient based on values in cell
    dfi.export(df_styled, "assets/info.png")


def transformaciones_dataf_mapacalor(df):
    #plt.hist(df)
    #plt.savefig('Images.png')
    corrdf = df.corr()
    sns.heatmap(corrdf, cmap='RdBu_r', annot=True)
    plt.savefig('assets/Images.png')


def transformaciones_dataf_hist(df):
    df.hist(figsize=(14,14), xrot=45)
    plt.figure()
    plt.savefig('assets/Images_hist.png')

def dataframe_toimage(df, name):
    df_styled = df.style.background_gradient() #adding a gradient based on values in cell
    dfi.export(df_styled, "assets/"+name+".png")


def Header(app):
    return html.Div([get_header(app), html.Br([]), get_menu(), ])

def Header2(app):
    return html.Div(get_menu2())

def Header3(app):
    return html.Div(get_menu3())

def get_header(app):
    header = html.Div(
        [
            html.Div(
                [
                    
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
                        href="https://github.com/rubenalejandrocalderoncorona/4MINING",
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
                        className="two columns main-title",
                    ),
                    
                    html.Div(
                        [
                            dcc.Link(
                                "Full View",
                                href="/4mining/full-view",
                                className="full-view-link",
                            )
                        ],
                        className="eight columns",
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
                href="/4mining/overview",
                className="tab first",
            ),
            dcc.Link(
                "EDA",
                href="/4mining/eda2",
                className="tab",
            ),
            dcc.Link(
                "PCA",
                href="/4mining/acp",
                className="tab",
            ),
            dcc.Link(
                "Arboles de Decision", 
                href="/4mining/arboles", 
                className="tab",
                
            ),
            dcc.Link(
                "Bosques Aleatorios",
                href="/4mining/bosques",
                className="tab",
            ),
            dcc.Link(
                "Clustering",
                href="/4mining/clusters",
                className="tab",
            ),
            dcc.Link(
                "SVM",
                href="/4mining/vectores",
                className="tab",
            ),
            

        ],
        className="row all-tabs",
    )
    return menu

def get_menu2():
    menu = html.Div(
        [
            dcc.Link(
                "Clasificación",
                href="/4mining/arboles_clasificacion",
                className="tab first",
            ),
            dcc.Link(
                "Regresión",
                href="/4mining/arboles_regresion",
                className="tab",
            ),
        ],
        className="row all-tabs",
    )
    return menu

def get_menu3():
    menu = html.Div(
        [
            dcc.Link(
                "Clasificación",
                href="/4mining/bosques_clasificacion",
                className="tab first",
            ),
            dcc.Link(
                "Regresión",
                href="/4mining/bosques_regresion",
                className="tab",
            ),
        ],
        className="row all-tabs",
    )
    return menu


 

 


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
