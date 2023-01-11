import base64
import datetime
import io
from utils import Header, Header3
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
import plotly.graph_objs as go         # Para la visualización de datos basado en plotly

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from dash_bootstrap_templates import load_figure_template,ThemeChangerAIO, template_from_url

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

tabs_styles = {
    'border-radius': '10px'
}
tab_style = {
  'border-width' : '0',
  'font-size' : '15px'
}

tab_selected_style = {
    'backgroundColor': '#A8D0F3',
    'border-top-left-radius': '5px',
    'border-top-right-radius': '5px'
}

tab_style2 = {
  'border-width' : '0',
  'font-size' : '10px',
  'text-align' : 'center'
}

tab_selected_style2 = {
    'backgroundColor': '#A8D0F3',
    'border-top-left-radius': '5px',
    'border-top-right-radius': '5px',
    'font-size' : '10px',
    'text-align' : 'center'
}

layout = html.Div([
    Header(app),
    Header3(app),
    html.H1('Bosques Aleatorios (Regresión)', style={'text-align': 'center'}),
    html.Div([
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
                'margin-left': 'auto',
                'margin-right': 'auto',
                'margin-bottom': '20px',
                
            },
            # Allow multiple files to be uploaded
            multiple=True,
            accept='.csv, .txt, .xls, .xlsx'
        ),
        ], 
    className="upload", 
    ),
    html.Div(id='output-data-upload-bosques-regresion'), # output-datatable
    html.Div(id='output-div'),
], className="page")

def parse_contents(contents, filename,date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    global df
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        dbc.Toast(
            [html.P('El número de Filas del Dataframe es de: {}'.format(df.shape[0])),
            html.P('El número de Columnas del Dataframe es de: {}'.format(df.shape[1])),
            dbc.Alert('El archivo cargado es: {}'.format(filename), color="success"),],
            id="simple-toast",
            header="Información Dataframe",
            style={
                'position': 'relative',
                'border-radius': '5px',
                'box-shadow': '0 0 10px #003B64',
                'padding': '5px',
                'width' : '60%',
                'margin': 'auto',
                'margin-bottom': '5px',
                
            }
        ),
       
        dash_table.DataTable(
           data=df.to_dict('records'),
            page_size=8,
            #filter_action='native',
            #sort_action='native',
            sort_mode='multi',
            column_selectable='single',
            row_deletable=True,
            cell_selectable=True,
            editable=True,
            row_selectable='multi',
            columns=[{'name': i, 'id': i, "deletable":True} for i in df.columns],
            style_table={'height': '300px', 'padding' : '50px', 'overflowX': 'scroll', 'margin-top' : '50px'},
        ),

        html.Hr(),  
        

        html.H2(["", dbc.Badge("Análisis de Correlaciones", className="ms-1")]),
        dcc.Tab(label='Matriz de Correlación', children=[
            dcc.Graph(
                id='matriz',
                figure={
                    # Solo se despliega la mitad de la matriz de correlación, ya que la otra mitad es simétrica
                    'data': [
                        {'x': df.corr().columns, 'y': df.corr().columns, 'z': df.corr().values, 'type': 'heatmap', 'colorscale': 'RdBu'}
                    ],
                    'layout': {
                        'title': 'Matriz de correlación',
                        'xaxis': {'side': 'down'},
                        'yaxis': {'side': 'left'},
                        # Agregamos el valor de correlación por en cada celda (text_auto = True)
                        'annotations': [
                            dict(
                                x=df.corr().columns[i],
                                y=df.corr().columns[j],
                                text=str(round(df.corr().values[i][j], 4)),
                                showarrow=False,
                                font=dict(
                                    color='white' if abs(df.corr().values[i][j]) >= 0.67  else 'black'
                                )
                            ) for i in range(len(df.corr().columns)) for j in range(len(df.corr().columns))
                        ]
                    }
                }
            )
        ]),

        dcc.Tabs([
            dcc.Tab(label='Gráfico Series de Tiempo', style=tab_style, selected_style=tab_selected_style,children=[
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        html.Br(),
                        dbc.Badge("Selecciona la variable X", color="light", className="mr-1", text_color="dark"),
                        html.Br(),
                        dbc.Select(
                            options=[{'label': i, 'value': i} for i in df.columns],
                            # Selecionamos por defecto la primera columna
                            value=df.columns[0],
                            id='xaxis_column-bosque-regresion',
                            style={'width': '100%', 'className': 'mr-1'}
                        ),
                    ]),
                    dbc.Col([
                        html.Br(),
                        dbc.Badge("Selecciona la variable Y", color="light", className="mr-1", text_color="dark"),
                        html.Br(),
                        dcc.Dropdown(
                            [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                            # Seleccionamos por defecto todas las columnas numéricas, a partir de la segunda
                            value=[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2][1:3],
                            id='yaxis_column-bosque-regresion',
                            multi=True
                        ),
                    ]),

                    dcc.Graph(id='indicator_graphic_bosque_regression')
                ]),
            ]),

            dcc.Tab(label='Aplicación del algoritmo', style=tab_style, selected_style=tab_selected_style, children=[
                html.Br(),
                dbc.Badge("Selecciona la variable a predecir", color="light", className="mr-1", text_color="dark"),
                html.Br(),
                dbc.Select(
                    options=[{'label': i, 'value': i} for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                    value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns[0],
                    id='Y_Clase_Bosque_Regresion',
                    style={'width': '100%', 'className': 'mr-1'}
                ),
                html.Br(),
                dbc.Badge("Selecciona las variables predictoras", color="light", className="mr-1", text_color="dark"),
                html.Br(),
                dcc.Dropdown(
                    # En las opciones que aparezcan en el Dropdown, queremos que aparezcan todas las columnas numéricas, excepto la columna Clase
                    [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                    value=[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2][1:],
                    id='X_Clase_Bosque_Regresion',
                    multi=True,
                ),

                html.Br(),

                html.H2(["", dbc.Badge("Configuración Algoritmo", className="ms-1")]),
                html.Br(),


                dbc.Row([
                    dbc.Col([
                        dcc.Markdown('''**Criterio de División (Tamaño del test %):**'''),
                        dcc.Slider(0.2, 0.3, 0.05, value=0.2, marks={0.2: '20%', 0.25: '25%', 0.3: '30%'}, id='criterio_division_BAR'),
                    ], width=3, align='center'),

                ], justify='center', align='center'),

                html.Br(),


                dbc.Row([
                    dbc.Col([
                        dcc.Markdown('''**Criterio:**'''),
                        dbc.Select(
                            id='criterion_BAR',
                            options=[
                                {'label': 'Squared Error', 'value': 'squared_error'},
                                {'label': 'Friedman MSE', 'value': 'friedman_mse'},
                                {'label': 'Absolute Error', 'value': 'absolute_error'},
                                {'label': 'Poisson', 'value': 'poisson'},
                            ],
                            value='squared_error',
                            placeholder="Selecciona el criterio",
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**n_estimators:**'''),
                        dbc.Input(
                            id='n_estimators_BAR',
                            type='number',
                            placeholder='Ingresa el número de árboles',
                            value=100,
                            min=1,
                            max=1000,
                            step=1,
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**n_jobs:**'''),
                        dbc.Input(
                            id='n_jobs_BAR',
                            type='number',
                            placeholder='None',
                            value=None,
                            min=-1,
                            max=100,
                            step=1,
                        ),
                    ], width=2, align='center'),


                    dbc.Col([
                        dcc.Markdown('''**max_features:**'''),
                        dbc.Select(
                            id='max_features_BAR',
                            options=[
                                {'label': 'Auto', 'value': 'auto'},
                                {'label': 'sqrt', 'value': 'sqrt'},
                                {'label': 'log2', 'value': 'log2'},
                            ],
                            value='auto',
                            placeholder="Selecciona una opción",
                        ),
                    ], width=2, align='center'),

                    
                    dbc.Col([
                        dcc.Markdown('''**Max_depth:**'''),
                        dbc.Input(
                            id='max_depth_BAR',
                            type='number',
                            placeholder='None',
                            value=None,
                            min=1,
                            max=100,
                            step=1,
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**Min_samples_split:**'''),
                        dbc.Input(
                            id='min_samples_split_BAR',
                            type='number',
                            placeholder='Selecciona el min_samples_split',
                            value=2,
                            min=1,
                            max=100,
                            step=1,
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**Min_samples_leaf:**'''),
                        dbc.Input(
                            id='min_samples_leaf_BAR',
                            type='number',
                            placeholder='Selecciona el min_samples_leaf',
                            value=1,
                            min=1,
                            max=100,
                            step=1,
                        ),
                    ], width=2, align='center'),

                    dbc.Col([
                        dcc.Markdown('''**max_leaf_nodes:**'''),
                        dbc.Input(
                            id='max_leaf_nodes_BAR',
                            type='number',
                            placeholder='None',
                            value=None,
                            min=1,
                            max=1000,
                            step=1,
                        ),
                    ], width=2, align='center'),
            
                ], justify='center', align='center'),

                html.Br(),


                dbc.Button("Click para entrenar al algoritmo", color="danger", className="mr-1", id='submit-button-bosque-regresion', style={'width': '100%'}),

                html.Hr(),
                
                dcc.Tabs([
                dcc.Tab(label='Gráfica de Regresión', style=tab_style2, selected_style=tab_selected_style2, children=[
                dcc.Graph(id='matriz-bosque-regresion'),
                ]),


                dcc.Tab(label='Matriz de Clasificación', style=tab_style2, selected_style=tab_selected_style2, children=[
                html.Div(id='clasificacion-bosque-regresion'),
                ]),

                dcc.Tab(label='Importancia Variables', style=tab_style2, selected_style=tab_selected_style2, children=[
                dcc.Graph(id='importancia-bosque-regresion'),
                ]),
                ]),
            ]),

            dcc.Tab(label='Nuevos Pronósticos', style=tab_style, selected_style=tab_selected_style, children=[
                html.Div(id="output-regresion-BAR-Final"),

                html.Div(id='valor-BAR-regresion2'),
                html.Div(id='valor-BAR-regresion'),

                html.Hr(),

                dcc.Store(id='memory-output-BAR', data=df.to_dict('records')),
            ]),
        ], style={'margin-top' : '15px'})
    ],  className="sub-page", style={'padding' : '100px', 'overflow' : 'auto'})

@callback(Output('output-data-upload-bosques-regresion', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n,d) for c, n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children


@callback(
    Output('indicator_graphic_bosque_regression', 'figure'),
    Input('xaxis_column-bosque-regresion', 'value'),
    Input('yaxis_column-bosque-regresion', 'value'))
def update_graph2(xaxis_column2, yaxis_column2):
    # Conforme se van seleccionando las variables, se van agregando a la gráfica de líneas
    fig = go.Figure()
    for i in yaxis_column2:
        fig.add_trace(go.Scatter(x=df[xaxis_column2], y=df[i], mode='lines', name=i))
    fig.update_layout(xaxis_rangeslider_visible=True,showlegend=True, xaxis_title=xaxis_column2, yaxis_title='Valores',
                    font=dict(family="Courier New, monospace", size=18, color="black"))
    fig.update_traces(mode='markers+lines')

    return fig

@callback(
    Output('matriz-bosque-regresion', 'figure'),
    Output('clasificacion-bosque-regresion', 'children'),
    Output('importancia-bosque-regresion', 'figure'),
    Output('output-regresion-BAR-Final', 'children'),
    Output('valor-BAR-regresion2', 'children'),
    Input('submit-button-bosque-regresion', 'n_clicks'),
    State('X_Clase_Bosque_Regresion', 'value'),
    State('Y_Clase_Bosque_Regresion', 'value'),
    State('criterio_division_BAR', 'value'),
    State('criterion_BAR', 'value'),
    State('n_estimators_BAR', 'value'),
    State('n_jobs_BAR', 'value'),
    State('max_features_BAR', 'value'),
    State('max_depth_BAR', 'value'),
    State('min_samples_split_BAR', 'value'),
    State('min_samples_leaf_BAR', 'value'),
    State('max_leaf_nodes_BAR', 'value'),
    )
def regresion(n_clicks, X_Clase, Y_Clase, criterio_division, criterion, n_estimators, n_jobs, max_features, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes):
    if n_clicks is not None:
        X = np.array(df[X_Clase])
        Y = np.array(df[Y_Clase])

        global PronosticoBA

        from sklearn import model_selection
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                                test_size = criterio_division,
                                                                                random_state = 0,
                                                                                shuffle = True)

        #Se entrena el modelo a partir de los datos de entrada
        global PronosticoBA
        PronosticoBA = RandomForestRegressor(criterion=criterion, n_estimators=n_estimators, n_jobs=n_jobs, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, random_state=0)
        PronosticoBA.fit(X_train, Y_train)

        #Se genera el pronóstico
        Y_PronosticoBosque = PronosticoBA.predict(X_test)
        
        ValoresBosque = pd.DataFrame(Y_test, Y_PronosticoBosque)

        # Comparación de los valores reales y los pronosticados en Plotly
        fig = px.line(Y_test, color_discrete_sequence=['green'])
        fig.add_scatter(y=Y_PronosticoBosque, name='Y_Pronostico', mode='lines', line=dict(color='red'))
        fig.update_layout(title='Comparación de valores reales vs Pronosticados',xaxis_rangeslider_visible=True)
        #Cambiamos el nombre de la leyenda
        fig.update_layout(legend_title_text='Valores')
        fig.data[0].name = 'Valores Reales'
        fig.data[1].name = 'Valores Pronosticados'
        # Renombramos el nombre de las leyendas:
        fig.update_traces(mode='markers+lines') #Agregamos puntos a la gráfica
        
        
        criterio = PronosticoBA.criterion
        #MAE:
        MAEArbol = mean_absolute_error(Y_test, Y_PronosticoBosque)
        #MSE:
        MSEArbol = mean_squared_error(Y_test, Y_PronosticoBosque)
        #RMSE:
        RMSEArbol = mean_squared_error(Y_test, Y_PronosticoBosque, squared=False)
        # Score
        global ScoreArbol
        ScoreArbol = r2_score(Y_test, Y_PronosticoBosque)
        

        # Importancia de las variables
        importancia = pd.DataFrame({'Variable': list(df[X_Clase].columns),
                            'Importancia': PronosticoBA.feature_importances_}).sort_values('Importancia', ascending=False)

        # Graficamos la importancia de las variables
        fig2 = px.bar(importancia, x='Variable', y='Importancia', color='Importancia', color_continuous_scale='Bluered', text='Importancia')
        fig2.update_layout(title_text='Importancia de las variables', xaxis_title="Variables", yaxis_title="Importancia")
        fig2.update_traces(texttemplate='%{text:.2}', textposition='outside')
        fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig2.update_layout(legend_title_text='Importancia de las variables')

        # Generamos en texto el árbol de decisión
        Estimador = PronosticoBA.estimators_[1] # Se debe poder modificar
        from sklearn.tree import export_text
        r = export_text(Estimador, feature_names=list(df[X_Clase].columns))
        
        return fig, html.Div([
            html.H2(["", dbc.Badge("Metricas del Algoritmo", className="ms-1")]),
            html.Div([
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Score"),
                                html.Th("MAE"),
                                html.Th("MSE"),
                                html.Th("RMSE"),
                                html.Th("Criterion"),
                                html.Th("n_estimators"),
                                html.Th("n_jobs"),
                                html.Th("max_features"),
                                html.Th("Max_depth"),
                                html.Th("Min_samples_split"),
                                html.Th("Min_samples_leaf"),
                                html.Th("Max_leaf_nodes"),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(str(round(ScoreArbol, 6)*100) + '%', style={'color': 'green'}),
                                    html.Td(str(round(MAEArbol, 6))),
                                    html.Td(str(round(MSEArbol, 6))),
                                    html.Td(str(round(RMSEArbol, 6))),
                                    html.Td(criterio),
                                    html.Td(str(n_estimators)),
                                    html.Td(str(n_jobs)),
                                    html.Td(str(max_features)),
                                    html.Td(str(max_depth)),
                                    html.Td(min_samples_split),
                                    html.Td(min_samples_leaf),
                                    html.Td(str(max_leaf_nodes)),
                                ]
                            ),
                        ]
                    ),
                ],
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
                style={'width': '100%', 'text-align': 'center'},
                class_name='table table-hover table-bordered table-striped',
            ), ], style={'textAlign': 'center', 'width': '0px', 'overflowX' : 'scroll', 'min-width' : '100%'} ),  
            
        ]), fig2, html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Input(id='values_X1_BAR', type="number", placeholder=df[X_Clase].columns[0],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[0])),
                    dbc.Input(id='values_X2_BAR', type="number", placeholder=df[X_Clase].columns[1],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[1])),
                    dbc.Input(id='values_X3_BAR', type="number", placeholder=df[X_Clase].columns[2],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[2])),
                    dbc.Input(id='values_X4_BAR', type="number", placeholder=df[X_Clase].columns[3],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[3])),
                    dbc.Input(id='values_X5_BAR', type="number", placeholder=df[X_Clase].columns[4],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[4])),
                ], width=6),
            ])

        ]), html.Div([
                dbc.Button("Haz click para mostrar el pronóstico", id="collapse-button-BAR", className="mb-3", color="primary"),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody([
                        html.Div(id='output-container-button-BAR'),
                    ])),
                    id="collapse",
                ),
        ])
    
    elif n_clicks is None:
        import dash.exceptions as de
        raise de.PreventUpdate

# make sure that x and y values can't be the same variable
def filter_options(v):
    """Disable option v"""
    return [
        {"label": col, "value": col, "disabled": col == v}
        for col in [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]
    ]

# functionality is the same for both dropdowns, so we reuse filter_options
callback(Output("X_Clase_Bosque_Regresion", "options"), [Input("Y_Clase_Bosque_Regresion", "value")])(
    filter_options
)
callback(Output("Y_Clase_Bosque_Regresion", "options"), [Input("X_Clase_Bosque_Regresion", "value")])(
    filter_options
)

@callback(
    Output('valor-BAR-regresion', 'children'),
    Input('collapse-button-BAR', 'n_clicks'),
    # Mostar los valores de los inputs
    State('memory-output-BAR', 'data'),
    State('values_X1_BAR', 'value'),
    State('values_X2_BAR', 'value'),
    State('values_X3_BAR', 'value'),
    State('values_X4_BAR', 'value'),
    State('values_X5_BAR', 'value'),
)
def regresionFinal(n_clicks, data, values_X1, values_X2, values_X3, values_X4, values_X5):
    if n_clicks is not None:
        if values_X1 is None or values_X2 is None or values_X3 is None or values_X4 is None or values_X5 is None:
            return html.Div([
                dbc.Alert('Debe ingresar los valores de las variables', color="danger")
            ])
        else:
            XPredict = pd.DataFrame([[values_X1, values_X2, values_X3, values_X4, values_X5]])

            clasiFinal = PronosticoBA.predict(XPredict)
            return html.Div([
                dbc.Alert('El valor pronosticado con un Bosque Aleatorio que tiene una Exactitud de: ' + str(round(ScoreArbol, 6)*100) + '% es: ' + str(clasiFinal[0]), color="success", style={'textAlign': 'center'})
            ])


@callback(
    Output("modal-body-scroll-info-BAR", "is_open"),
    [
        Input("open-body-scroll-info-BAR", "n_clicks"),
        Input("close-body-scroll-info-BAR", "n_clicks"),
    ],
    [State("modal-body-scroll-info-BAR", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

