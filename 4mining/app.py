# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from pages import (
    overview,
    eda2,
    acp,
    arboles,
    arboles_clasificacion,
    arboles_regresion,
    bosques,
    bosques_clasificacion,
    bosques_regresion,
    clusters,
    vectores
)

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    suppress_callback_exceptions=True
)
app.title = "4 MINNING"
server = app.server

CONTENT_STYLE = {
    "margin-left": "1rem",
    "margin-right": "1rem",
    "padding": "1rem 1rem",
}


# Describe the layout/ UI of the app
content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), content])

# Update page
@app.callback(Output("page-content", "children"),  [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/4mining/bosques":
        return bosques.layout
    elif pathname == "/4mining/bosques_clasificacion":
        return bosques_clasificacion.layout
    elif pathname == "/4mining/bosques_regresion":
        return bosques_regresion.layout
    elif pathname == "/4mining/arboles":
        return arboles.layout
    elif pathname == "/4mining/arboles_clasificacion":
        return arboles_clasificacion.layout
    elif pathname == "/4mining/arboles_regresion":
        return arboles_regresion.layout
    elif pathname == "/4mining/eda2":
        return eda2.layout
    elif pathname == "/4mining/acp":
        return acp.layout
    elif pathname == "/4mining/clusters":
        return clusters.layout
    elif pathname == "/4mining/vectores":
        return vectores.layout
    elif pathname == "/4mining/full-view":
        return (
            overview.create_layout(app),
            bosques.layout,
            eda2.layout,
            acp.layout,
            arboles.layout,
            bosques_clasificacion.layout,
            bosques_regresion.layout,
            arboles_clasificacion.layout,
            arboles_regresion.layout,
            clusters.layout,
            vectores.layout
        )
    else:
        return overview.create_layout(app)


if __name__ == "__main__":
    app.run_server(debug=True)
