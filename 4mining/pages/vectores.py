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
from utils import Header, Header3
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table, Input, Output, callback
import plotly.express as px
import plotly.graph_objs as go         # Para la visualización de datos basado en plotly

from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from dash_bootstrap_templates import load_figure_template,ThemeChangerAIO, template_from_url

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])


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
    html.H1('Support Vector Machines (SVM)', style={'text-align': 'center'}),
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
    html.Div(id='output-data-upload-svm'), # output-datatable
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

        html.H2(["", dbc.Badge("Análisis Correlacional", className="ms-1")]),
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

            dcc.Tab(label='Grafico de Dispersión', style=tab_style, selected_style=tab_selected_style,children=[
                html.Div([
                    "Selecciona la variable X:",
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos la primera columna numérica del dataframe
                        value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns[0],
                        id='xaxis_column-svm',
                    ),

                    "Selecciona la variable Y:",
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos la segunda columna numérica del dataframe
                        value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns[1],
                        id='yaxis_column-svm',
                        placeholder="Selecciona la variable Y"
                    ),

                    "Selecciona la variable a Clasificar:",
                    dcc.Dropdown(
                        [i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)],
                        # Seleccionamos por defecto la primera columna
                        value=df[[i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)]].columns[0],
                        id='caxis_column-svm',
                        placeholder="Selecciona la variable Predictora"
                    ),
                ]),

                dcc.Graph(id='indicator_graphic-svm'),
            ]),

            dcc.Tab(label='Aplicación del algoritmo', style=tab_style, selected_style=tab_selected_style, children=[
                dbc.Badge("Selecciona las variables predictoras", color="light", className="mr-1", text_color="dark"),
                    dcc.Dropdown(
                        [i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2],
                        # Seleccionamos la segunda columna numérica del dataframe
                        value=df[[i for i in df.columns if df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) > 2]].columns,
                        id='X_Clase-svm',
                        multi=True,
                    ),

                # Seleccionamos la variable Clase con un Dropdown
                dbc.Badge("Selecciona la variable Clase", color="light", className="mr-1", text_color="dark"),
                dcc.Dropdown(
                    [i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)],
                    # Seleccionamos por defecto la primera columna
                    value=df[[i for i in df.columns if (df[i].dtype in ['float64', 'int64'] and len(df[i].unique()) == 2) or df[i].dtype in ['bool'] or (df[i].dtype in ['object'] and len(df[i].unique()) == 2)]].columns[0],
                    id='Y_Clase-svm',
                    multi=True,
                ),

                # Salto de línea
                html.Br(),

                html.H2(["", dbc.Badge("Configuración del algoritmo", className="ms-1")]),
                html.Br(),

                
                dbc.Row([
                    dbc.Col([
                        dcc.Markdown('''**Kernel**'''),
                        dbc.Select(
                            id='kernel-svm',
                            options=[
                                {'label': 'Linear', 'value': 'linear'},
                                {'label': 'Polynomial', 'value': 'poly'},
                                {'label': 'Radial Basis Function', 'value': 'rbf'},
                                {'label': 'Sigmoid', 'value': 'sigmoid'},
                            ],
                            value='linear',
                            placeholder="Selecciona el criterio",
                        ),
                    ], width=2, align='center'),
                ], justify='center', align='center'),

                html.Br(),

                dbc.Button("Click para entrenar al algoritmo", color="danger", className="mr-1", id='submit-button-clasificacion-svm',style={'width': '100%'}),

                html.Hr(),

                dcc.Tabs([
                dcc.Tab(label='Matriz de Clasificación', style=tab_style2, selected_style=tab_selected_style2, children=[
                html.Div(id='matriz-svm'),
                ]),



                dcc.Tab(label='Información Configuración', style=tab_style2, selected_style=tab_selected_style2, children=[
                html.Div(id='clasificacion-svm'),
                ]),


                dcc.Tab(label='Curva ROC', style=tab_style2, selected_style=tab_selected_style2, children=[
                html.H2(["", dbc.Badge("Gráfica Curva ROC", className="ms-1")]),
                dcc.Graph(id='roc-arbol-clasificacion-svm'),
                ]),
                 

                dcc.Tab(label='Vectores de Soporte', style=tab_style2, selected_style=tab_selected_style2, children=[
                html.H2(["", dbc.Badge("Vectores de Soporte", className="ms-1")]),
                dcc.Graph(id='vectores-svm'),
                ]),
                ]),
            ]),

            dcc.Tab(label='Nuevas Clasificaciones', style=tab_style, selected_style=tab_selected_style, children=[
                html.H2(["", dbc.Badge("Introduce los datos de las nuevas clasificaciones", className="ms-1")]),
                html.Hr(),
                html.Div(id='output-clasificacion-svm'),
                html.Hr(),
                html.Div(id='valor-clasificacion-svm'),
                html.Div(id='valor-clasificacion-svm2'),
                
            ]),
        ],  style={'margin-top' : '15px'})
    ], className="sub-page", style={'padding' : '100px', 'overflow' : 'auto'}) #Fin del layout

@callback(Output('output-data-upload-svm', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names,list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n,d) for c, n,d in
            zip(list_of_contents, list_of_names,list_of_dates)]
        return children

# CALLBACK PARA LA SELECCIÓN DEL USUARIO
@callback(
    Output('indicator_graphic-svm', 'figure'),
    Input('xaxis_column-svm', 'value'),
    Input('yaxis_column-svm', 'value'),
    Input('caxis_column-svm', 'value'))
def update_graph(xaxis_column, yaxis_column, caxis_column):
    dff = df
    dff[caxis_column] = dff[caxis_column].astype('category')
    fig = px.scatter(dff, x=xaxis_column, y=yaxis_column, color=caxis_column, title='Gráfico de dispersión')
    fig.update_layout(showlegend=True, xaxis_title=xaxis_column, yaxis_title=yaxis_column,
                    font=dict(size=18, color="blue"),legend_title_text=caxis_column)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    # str(df.groupby(caxis_column).size()[0])

    return fig

@callback(
    Output('matriz-svm', 'children'),
    Output('clasificacion-svm', 'children'),
    Output('roc-arbol-clasificacion-svm', 'figure'),
    Output('vectores-svm', 'figure'),
    Output('output-clasificacion-svm', 'children'),
    Output('valor-clasificacion-svm', 'children'),
    Input('submit-button-clasificacion-svm','n_clicks'),
    State('X_Clase-svm', 'value'),
    State('Y_Clase-svm', 'value'),
    State('kernel-svm', 'value'))
    #State(ThemeChangerAIO.ids.radio("theme"), 'value'))
def clasificacion(n_clicks, X_Clase, Y_Clase, kernel):
    if n_clicks is not None:
        global ModeloSVM

        dff = df.copy()
        Y_Clase2 = Y_Clase

        #Si la primera clase es 0, se cambia a 0, y si es 1, se cambia a 1
        if df[Y_Clase].unique()[0] == 0:
            # Reemplazamos los valores 0 por 0:
            df[Y_Clase] = df[Y_Clase].replace(df[Y_Clase].unique()[0], 0)
        elif df[Y_Clase].unique()[0] == 1:
            # Reemplazamos los valores 1 por 1:
            df[Y_Clase] = df[Y_Clase].replace(df[Y_Clase].unique()[0], 1)
        elif df[Y_Clase].unique()[1] == 0:
            # Reemplazamos los valores 0 por 0:
            df[Y_Clase] = df[Y_Clase].replace(df[Y_Clase].unique()[1], 0)
        elif df[Y_Clase].unique()[1] == 1:
            # Reemplazamos los valores 1 por 1:
            df[Y_Clase] = df[Y_Clase].replace(df[Y_Clase].unique()[1], 1)
        # Si no se cumple ninguna de las condiciones anteriores, se reemplazan los valores de la posición 0 por 0 y los de la posición 1 por 1:
        else:
            #valor1Cambio = df[Y_Clase].unique()[0]
            #valor2Cambio = df[Y_Clase].unique()[1]
            dff[Y_Clase2] = dff[Y_Clase2].replace(dff[Y_Clase2].unique()[0], 0)
            dff[Y_Clase2] = dff[Y_Clase2].replace(dff[Y_Clase2].unique()[1], 1)
            
        X = np.array(df[X_Clase])
        Y = np.array(dff[Y_Clase2])

        from sklearn import model_selection
        from sklearn.svm import SVC #Support vector classifier
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                        test_size = 0.2, 
                                                                                        random_state = 0,
                                                                                        shuffle = True)

        #Se entrena el modelo a partir de los datos de entrada
        ModeloSVM = SVC(kernel=kernel)
        
        #Se entrena el modelo a partir de los datos de entrada
        ModeloSVM.fit(X_train, Y_train)

        #Se etiquetan las clasificaciones
        Clasificaciones = ModeloSVM.predict(X_validation)

        Valores = pd.DataFrame(Y_validation, Clasificaciones)

        #Se calcula la exactitud promedio de la validación
        global exactitud
        exactitud = ModeloSVM.score(X_validation, Y_validation)
        
        #Matriz de clasificación
        ModeloClasificacion1 = ModeloSVM.predict(X_validation)
        Matriz_Clasificacion1 = pd.crosstab(Y_validation.ravel(), 
                                        ModeloClasificacion1, 
                                        rownames=['Reales'], 
                                        colnames=['Clasificación'])

        VP = Matriz_Clasificacion1.iloc[0,0]
        FP = Matriz_Clasificacion1.iloc[1,0]
        FN = Matriz_Clasificacion1.iloc[0,1]
        VN = Matriz_Clasificacion1.iloc[1,1]

        kernelSVM = kernel.upper()
        
        tasa_error = 1-exactitud

        #CURVA ROC
        Y_validation2 = pd.DataFrame(Y_validation) # Convertimos los valores de la variable Y_validation a un dataframe
        # Reeemplazamos los valores de la variable Y_validation2 por 0 y 1:
        # Checamos si el dataframe tiene dos valores únicos y si esos son 1 y 0
        if len(Y_validation2[0].unique()) == 2 and Y_validation2[0].unique()[0] == 0 and Y_validation2[0].unique()[1] == 1 or len(Y_validation2[0].unique()) == 2 and Y_validation2[0].unique()[0] == 1 and Y_validation2[0].unique()[1] == 0:
            pass
        else:
            Y_validation2 = Y_validation2.replace([Y_validation2[0].unique()[0],Y_validation2[0].unique()[1]],[1,0])
        

        # Graficamos la curva ROC con Plotly
        y_score1 = ModeloSVM.decision_function(X_validation)

        from sklearn.metrics import roc_curve, auc
        fpr1, tpr1, thresholds1 = roc_curve(Y_validation2, y_score1)

        # Graficamos la curva ROC con Plotly
        fig2 = px.area(title='Curva ROC del modelo SVM. Kernel: ' +str(kernel), x=fpr1, y=tpr1, labels={'x':'False Positive Rate', 'y':'True Positive Rate'})
        # Agregamos la curva ROC del arbol de decisión más el ROC de este modelo y agregamos un área sombreada debajo de la curva
        fig2.add_scatter(x=fpr1, y=tpr1, mode='lines', name='SVM ' +str(kernel) +', AUC = '+str(auc(fpr1, tpr1).round(4)), fill='tonexty')
        # Agregamos la diagonal de la curva ROC con líneas punteadas
        fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Black", dash="dash"))


        VectoresSoporte = ModeloSVM.support_vectors_

        # Hacemos lo mismo con Plotly
        fig3 = px.scatter()
        fig3.add_scatter(x=X_validation[:, 0], y=X_validation[:, 1], mode='markers', marker=dict(color=Clasificaciones, colorscale='temps',symbol="octagon"), name='Validación')
        fig3.add_scatter(x=X_train[:, 0], y=X_train[:, 1], mode='markers', marker=dict(color=Y_train, colorscale='temps', symbol="diamond"), name='Entrenamiento')
        fig3.add_scatter(x=X_train[ModeloSVM.support_[0:ModeloSVM.n_support_[0]], 0], y=X_train[ModeloSVM.support_[0:ModeloSVM.n_support_[0]], 1], name='Vectores de soporte 1', mode='markers', line=dict(color='green'),
                        #Contorno de los vectores de soporte (negro)
                        marker=dict(line=dict(color='black', width=1)))
        fig3.add_scatter(x=X_train[ModeloSVM.support_[ModeloSVM.n_support_[0]:], 0], y=X_train[ModeloSVM.support_[ModeloSVM.n_support_[0]:], 1], name='Vectores de soporte 2', mode='markers', line=dict(color='pink'),
                        #Contorno de los vectores de soporte (neg
                        marker=dict(line=dict(color='black', width=1)))
        fig3.data[1].name = 'Datos de Validación'
        fig3.data[2].name = 'Datos de Entrenamiento'
        fig3.data[3].name = 'Vectores de soporte 1'
        fig3.data[4].name = 'Vectores de soporte 2'

        
        return html.Div([
            html.H2(["", dbc.Badge("Matriz de clasificación", className="ms-1")]),
            #dbc.Alert("Se hicieron cambios a los valores de la variable Clase. " + str(valor2Cambio) + ': 0', color="info",style={'textAlign': 'center'}),
            dbc.Row([
                dbc.Col([
                    dbc.Alert('Verdaderos Positivos (VP): ' + str(VP), color="info"),
                    dbc.Alert('Falsos Positivos (FP): ' + str(FP), color="info"),
                ], width=4),
                dbc.Col([
                    dbc.Alert('Falsos Negativos (FN): ' + str(FN), color="info"),
                    dbc.Alert('Verdaderos Negativos (VN): ' + str(VN), color="info"),
                ], width=4),
                ], justify="center"),

        ]), html.Div([
            html.H2(["", dbc.Badge("Kernel utilizado", className="ms-1")]),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Kernel"),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(kernelSVM),
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
            ),

            html.H2(["", dbc.Badge("Métricas Algoritmos", className="ms-1")]),
            dbc.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Reporte de clasificación"),
                                html.Th("Reporte de clasificación para la clase: " + str(classification_report(Y_validation, Clasificaciones).split()[4])),
                                html.Th("Reporte de clasificación para la clase: " + str(classification_report(Y_validation, Clasificaciones).split()[9])),
                            ]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("Exactitud (Accuracy): " + str(round(exactitud*100,2)) + '%', style={'color': 'green'}),
                                    html.Td("Precisión: " + str(round(float(VP/(VP+FP))*100,5)) + '%'),
                                    html.Td("Precisión: " + str(round(float(VN/(VN+FN))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Tasa de error (Misclassification Rate): " + str(round(tasa_error*100,2)) + '%', style={'color': 'red'}),
                                    html.Td("Sensibilidad (Recall, Sensitivity, True Positive Rate): " + str(round(float(VP/(VP+FN))*100,5)) + '%'),
                                    html.Td("Sensibilidad (Recall, Sensitivity, True Positive Rate): " + str(round(float(VN/(VN+FP))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Verdaderos: " + str((Y_validation == Clasificaciones).sum()), style={'color': 'green'}),
                                    html.Td("Especificidad (Specificity, True Negative Rate): " + str(round(float(VN/(VN+FP))*100,5)) + '%'),
                                    html.Td("Especificidad (Specificity, True Negative Rate): " + str(round(float(VP/(VP+FN))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Falsos: " + str((Y_validation != Clasificaciones).sum()), style={'color': 'red'}),
                                    html.Td("F1-Score: " + str(round(float(2*VP/(2*VP+FP+FN))*100,5)) + '%'),
                                    html.Td("F1-Score: " + str(round(float(2*VN/(2*VN+FN+FP))*100,5)) + '%'),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Valores Totales: " + str(Y_validation.size)),
                                    html.Td("Número de muestras: " + str(classification_report(Y_validation, Clasificaciones).split()[8])),
                                    html.Td("Número de muestras: " + str(classification_report(Y_validation, Clasificaciones).split()[13])),
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
            ),

        ]), fig2, fig3,html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Input(id='values_X1_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[0],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[0])),
                    dbc.Input(id='values_X2_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[1],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[1])),
                    dbc.Input(id='values_X3_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[2],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[2])),
                ], width=6),
                dbc.Col([
                    dbc.Input(id='values_X4_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[3],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[3])),
                    dbc.Input(id='values_X5_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[4],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[4])),
                    dbc.Input(id='values_X6_AD_Clasificacion', type="number", placeholder=df[X_Clase].columns[5],style={'width': '100%'}),
                    dbc.FormText("Ingrese el valor de la variable: " + str(df[X_Clase].columns[5])),
                ], width=6),
            ])
        ]), html.Div([
                dbc.Button("Haz click para mostrar la clasificación obtenida", id="collapse-button-svm", className="mb-3", color="dark", style={'width': '100%', 'text-align': 'center'}),
        ]),

    elif n_clicks is None:
        import dash.exceptions as de
        raise de.PreventUpdate


@callback(
    Output('valor-clasificacion-svm2', 'children'),
    Input('collapse-button-svm', 'n_clicks'),
    State('values_X1_AD_Clasificacion', 'value'),
    State('values_X2_AD_Clasificacion', 'value'),
    State('values_X3_AD_Clasificacion', 'value'),
    State('values_X4_AD_Clasificacion', 'value'),
    State('values_X5_AD_Clasificacion', 'value'),
    State('values_X6_AD_Clasificacion', 'value'),
)
def AD_Clasificacion_Pronostico(n_clicks, values_X1, values_X2, values_X3, values_X4, values_X5, values_X6):
    if n_clicks is not None:
        if values_X1 is None or values_X2 is None or values_X3 is None or values_X4 is None or values_X5 is None or values_X6 is None:
            return html.Div([
                dbc.Alert('Debe ingresar todos los valores de las variables', color="danger")
            ])
        else:
            XPredict = pd.DataFrame([[values_X1, values_X2, values_X3, values_X4, values_X5, values_X6]])

            clasiFinal = ModeloSVM.predict(XPredict)
            # Intercambiar los valores de la clasificación
            if clasiFinal[0] == 0:
                clasiFinal[0] = 1
            else:
                clasiFinal[0] = 0
            return html.Div([
                dbc.Alert('El valor clasificado con una SVM que tiene una Exactitud de: ' + str(round(exactitud, 6)*100) + '% es: ' + str(clasiFinal[0]), color="success", style={'textAlign': 'center'})
            ])
