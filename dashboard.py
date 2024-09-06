from dash import Dash, html, dash_table, dcc, Input, Output, _dash_renderer
import dash_mantine_components as dmc 
from dash_mantine_components import MantineProvider
import pandas as pd
import requests
import plotly.express as px
import datetime
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
import os
from tempfile import NamedTemporaryFile

_dash_renderer._set_react_version("18.2.0")


def obtener_dispositivos():
    url = "https://miserably-touched-gecko.ngrok-free.app/dispositivos/"  # Cambiar la URL según donde esté alojada tu API
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error al obtener los dispositivos:", response.status_code)
        return None

def definir_ubicaciones():
    df['latitud'] = pd.to_numeric(df['latitud']).round(3)
    df['longitud'] = pd.to_numeric(df['longitud']).round(3)

    for row in data:
        df['ubicacion'] = df['latitud'].astype(str) + ', ' + df['longitud'].astype(str)

    df.loc[df['ubicacion'] == '37.166, -3.599', 'ubicacion'] = 'Palacio de Congresos'
    df.loc[df['ubicacion'] == '37.172, -3.598', 'ubicacion'] = 'Fuente de las Batallas'
    df.loc[df['ubicacion'] == '37.174, -3.599', 'ubicacion'] = 'Plaza del Carmen'
    df.loc[df['ubicacion'] == '37.175, -3.6', 'ubicacion'] = 'Plaza Bib-Rambla'
    df.loc[df['ubicacion'] == '37.177, -3.595', 'ubicacion'] = 'Plaza Santa Ana'
    df.loc[df['ubicacion'] == '37.179, -3.59', 'ubicacion'] = 'Paseo de los Tristes'

def definir_horario():
    # Por defecto es tarde
    df['jornada'] = 'Tarde'
    df['hora_primera'] = pd.to_datetime(df['hora_primera'], format='%H:%M:%S').dt.time
    df.loc[(df['hora_primera'] > datetime.time(6, 0, 0)) & (df['hora_primera'] < datetime.time(15, 0, 0)), 'jornada'] = 'Mañana'

def encontrar_mac_duplicadas(data):
    macs = []
    duplicadas = []
    for dispositivo in data:
        mac = dispositivo['hashed_mac']
        if mac in macs and mac not in duplicadas:
            duplicadas.append(mac)
        else:
            macs.append(mac)
    return duplicadas


data = obtener_dispositivos()
duplicadas = encontrar_mac_duplicadas(data)


df = pd.DataFrame(data)

definir_ubicaciones()

df['count'] = 1

df_grouped_ubicacion = df.groupby('ubicacion').size().reset_index(name='count')

df['fecha'] = df['primera_fecha_hora'].str[:10]
df['hora_primera'] = df['primera_fecha_hora'].str[11:19]
df['hora_ultima'] = df['ultima_fecha_hora'].str[11:19]
df['hora_ultima'] = pd.to_datetime(df['hora_ultima'], format='%H:%M:%S').dt.time


definir_horario()


# ANÁLISIS DE DISPOSITIVOS REPETIDOS
# Filtrar los datos para las MACs duplicadas
def definir_recorridos_duplicadas(dff,duplicadas):
    df_duplicadas = dff[dff['hashed_mac'].isin(duplicadas)]

    df_duplicadas = df_duplicadas[['hashed_mac', 'fecha', 'ubicacion', 'latitud', 'longitud']]

    # DISTINTAS UBICACIONES
    # Verificar si las MACs duplicadas se encuentran en más de una ubicación
    df_duplicadas_ubicaciones = df_duplicadas.groupby('hashed_mac')['ubicacion'].nunique().reset_index()
    df_duplicadas_ubicaciones = df_duplicadas_ubicaciones[df_duplicadas_ubicaciones['ubicacion'] > 1]

    # Obtener solo las MACs duplicadas que se encontraron en distintas ubicaciones
    macs_duplicadas_distintas_ubicaciones = df_duplicadas_ubicaciones['hashed_mac'].tolist()

    df_macs_duplicadas_distintas_ubicaciones = df_duplicadas[dff['hashed_mac'].isin(macs_duplicadas_distintas_ubicaciones)]

    # DISTINTOS DÍAS
    # Verificar si las MACs duplicadas se encuentran en más de una fecha
    df_duplicadas_fecha = df_duplicadas.groupby('hashed_mac')['fecha'].nunique().reset_index()
    df_duplicadas_fecha = df_duplicadas_fecha[df_duplicadas_fecha['fecha'] > 1]

    # Obtener solo las MACs duplicadas que se encontraron en distintas ubicaciones
    macs_duplicadas_distintas_fechas = df_duplicadas_fecha['hashed_mac'].tolist()

    df_macs_duplicadas_distintas_fechas = df_duplicadas[dff['hashed_mac'].isin(macs_duplicadas_distintas_fechas)]

    #MAPA 3

    # Crear una nueva columna 'ubicacion_siguiente' que contenga la ubicación siguiente para cada 'hashed_mac'
    df_macs_duplicadas_distintas_ubicaciones['ubicacion_siguiente'] = df_macs_duplicadas_distintas_ubicaciones.groupby('hashed_mac')['ubicacion'].shift(-1)

    # Crear las columnas 'latitud_siguiente' y 'longitud_siguiente'
    df_macs_duplicadas_distintas_ubicaciones['latitud_siguiente'] = df_macs_duplicadas_distintas_ubicaciones.groupby('hashed_mac')['latitud'].shift(-1)
    df_macs_duplicadas_distintas_ubicaciones['longitud_siguiente'] = df_macs_duplicadas_distintas_ubicaciones.groupby('hashed_mac')['longitud'].shift(-1)

    # Eliminar las filas donde 'ubicacion_siguiente', 'latitud_siguiente' o 'longitud_siguiente' son NaN
    df_macs_duplicadas_distintas_ubicaciones = df_macs_duplicadas_distintas_ubicaciones.dropna(subset=['ubicacion_siguiente', 'latitud_siguiente', 'longitud_siguiente'])

    # Agrupar por 'ubicacion', 'ubicacion_siguiente', 'latitud', 'latitud_siguiente', 'longitud' y 'longitud_siguiente'
    # y sumar el número de 'hashed_mac' únicas
    df_recorridos = df_macs_duplicadas_distintas_ubicaciones.groupby(['ubicacion', 'ubicacion_siguiente', 'latitud', 'latitud_siguiente', 'longitud', 'longitud_siguiente'])['hashed_mac'].nunique().reset_index()
    df_recorridos.columns = ['ubicacion', 'ubicacion_siguiente', 'latitud', 'latitud_siguiente', 'longitud', 'longitud_siguiente', 'num_dispositivos']
    return df_recorridos, df_macs_duplicadas_distintas_ubicaciones



app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dmc.styles.DATES])

# Layout principal
app.layout = MantineProvider(
    children=[
    dcc.Location(id='url', refresh=False),
    html.Div(    
        children=[
            #Título
            html.Div(
                style={'grid-row': '1', 'display': 'flex', 'textAlign': 'center', 'justifyContent': 'center', 'color': 'white','background-color': '#0FA3B1', 'fontSize': 30, 'fontFamily': 'Arial', 'font-weight': 'bold', 'padding': '1%'},
                children='TRABAJO FIN DE GRADO - DASHBOARD DE DISPOSITIVOS'),
            #Selector de fechas
            html.Div(
                style={'padding-top': '1%', 'padding-left': '3%', 'padding-bottom': '1%', 'width': '30%'},
                children=[
                        dmc.DatePicker(
                            id='date-picker',
                            value= [df['fecha'].min(), df['fecha'].max()],
                            type='multiple',
                            valueFormat='DD-MM-YYYY',
                            clearable=True,
                            allowDeselect=True,
                            persistence_type='session',
                            
                        )
                    ,
                ]
            ),
        ]
    ),
    html.Div(id='page-content'),
])

# Layout de la página principal
index_page_no_data = html.Div(
    children=[
        html.Div(style={'textAlign': 'center', 'fontSize': 20, 'fontFamily': 'Arial', 'font-weight':'bold','padding': '1%'},
            children=[
                'No hay datos para las fechas seleccionadas',
                html.Br(),
                'Por favor, seleccione otras fechas'
            ]    
        )
    ]
)


index_page_with_data = html.Div(
    style={'display': 'grid', 'height': '80vh', 'gridTemplateRows': '50% 50%'}, 
    children=[
        #Sección 1
        html.Div(
            style={'grid-row': '1', 'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr',}, 
            children=[
                html.Div(style={'grid-column': '1', 'display': 'grid', 'gridTemplateRows': '1fr 1fr', 'padding-top': '3%', 'padding-left': '2%' },
                children=[ 
                    html.Div(style={'grid-row': '1', 'display': 'grid', 'gridTemplateRows': '1fr 1fr', 'marginTop': '5%',},
                        children=[
                            html.Div(id='device-count', style = {'fontSize': 20, 'background-color': '#D9E5D6', 'padding-top': '3%', 'borderRadius': '5px', 'textAlign': 'center', 'border': '3px solid', 'fontFamily': 'Arial'}),
                            html.A('Consultar todos los datos', id='show-data-table-link', href='/datos', n_clicks=0, style={'textAlign': 'center', 'justifyContent': 'center', 'padding-top': '10px', 'fontFamily': 'Arial'}), 
                        ]
                    ),
                
                    html.Div(id='duplicated-count', style={'grid-row': '2','fontSize': 20, 'background-color': '#D9E5D6', 'padding-top': '3%', 'borderRadius': '5px', 'textAlign': 'center', 'border': '3px solid', 'fontFamily': 'Arial','marginTop': '7%', 'marginBottom': '10%'}),
                ]),
                dcc.Graph(id='donut-jornada', style={'grid-column': '2', 'alignSelf': 'center', 'width': '100%', 'height': '100%'}),
                html.Iframe(id='folium-map', width='95%', height='95%'), 

            ]),
        html.Hr(),
        #Sección 2
        html.Div(style={'grid-row': '2', 'display': 'grid', 'gridTemplateColumns': '1fr 1fr',},
            children=[
                dcc.Graph(id='mapa_recorridos', style={'grid-column': '1', 'width': '100%', 'height': '95%', 'padding-left': '1%'}),  
                dcc.Graph(id='histogram-ubicacion', style={'grid-column': '2', 'width': '90%', 'height': '90%', 'padding-left': '5%'}),

        ])
])

# Layout de la página de datos
data_page = html.Div(
    style={'display':'inline'},
    children=[
        #Tabla de datos
        html.Div(style={'padding': '1%'}, 
                 children=[
                    dash_table.DataTable(
                        id='table-data',
                        columns=[{"name": i, "id": i} for i in ['hashed_mac', 'fecha', 'hora_primera', 'hora_ultima', 'latitud', 'longitud', 'ubicacion','jornada']],
                        data=[],
                        page_size=10,
                        style_cell={'textAlign': 'center', 'fontFamily': 'Arial'},
                    ),
                    html.A('Volver a la página principal', href='/', style={'fontFamily': 'Arial'}),
        ]),
])

# Callback para actualizar los datos de la tabla
@app.callback(
    Output('table-data', 'data'),
    Input('date-picker', 'value'),
)
def update_table_data(dates):
    if isinstance(dates, str):
        dates = [dates]

    dff = df[df['fecha'].isin(dates)]
    dff['fecha'] = pd.to_datetime(dff['fecha']).dt.strftime('%d-%m-%Y')

    return dff.to_dict('records')


# Callback para cambiar el contenido de la página según la URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    Input('date-picker', 'value'),
)
def display_page(pathname, dates):
    if isinstance(dates, str):
        dates = [dates]

    dff = df[df['fecha'].isin(dates)]    
    if pathname == '/':
        if dff.empty:
            return index_page_no_data
        else:
            return index_page_with_data
    elif pathname == '/datos':
        return data_page


# Callback para actualizar gráficos y contadores
@app.callback(
    [Output('device-count', 'children'),
     Output('duplicated-count', 'children'),
     Output('histogram-ubicacion', 'figure'),
     Output ('donut-jornada', 'figure'),
     Output('folium-map', 'srcDoc'),
     Output('mapa_recorridos', 'figure')],
     [Input('date-picker', 'value'),
      Input('show-data-table-link', 'n_clicks')]
)

def update_output(dates,n_clicks):
    if isinstance(dates, str):
        dates = [dates]

    dff = df[df['fecha'].isin(dates)]
    if dff.empty:
        return ('', '', '', '', '', '')
    else:
        dff_grouped_ubicacion = dff.groupby('ubicacion').size().reset_index(name='count')
        device_count = len(dff)
        duplicadas = encontrar_mac_duplicadas(dff.to_dict('records'))
        duplicated_count = len(duplicadas)

        dff['fecha'] = pd.to_datetime(dff['fecha']).dt.strftime('%d-%m-%Y')
        fig_ubicacion = px.histogram(dff, x='ubicacion', y='count', color='fecha', histfunc='sum', nbins=20, title='Histograma de ubicación de los dispositivos', color_discrete_sequence=['#0FA3B1', '#FF9B42', '#EDDEA4'])
        fig_ubicacion.update_layout(
            title={
                'text': 'Histograma de ubicación de los dispositivos',
                'font': {'size': 16, 'family': 'Arial'},  
                'x': 0.5  
            },
            xaxis_title={
                'text': '',
                'font': {'size': 1}  
            },
            yaxis_title={
                'text': '',
                'font': {'size': 1}  
            },
            font={
                'size': 10  
            },
            legend={
                'title': 'Fecha'
            },
            margin=dict(t=50, b=0, l=40, r=0)

        )
        fig_jornada= px.pie(dff, names='jornada', hole=0.4, title='Distribución de dispositivos por jornada', color_discrete_sequence=['#0FA3B1', '#FF9B42'])
        fig_jornada.update_layout(
            title={
                'text': 'Distribución de dispositivos por jornada',
                'font': {'size': 16, 'family': 'Arial'},  
                'x': 0.5  
            },
            margin=dict(t=50, b=50, l=70, r=40)
        )

        mapa = generar_mapa_calor(dff)
        with open(mapa, 'r') as f:
            folium_map_html = f.read()
        
        os.remove(mapa)

        mapa_recorridos = go.Figure()
        df_recorridos, df_macs_duplicadas_distintas_ubicaciones= definir_recorridos_duplicadas(dff, duplicadas)
        for i, row in df_recorridos.iterrows():
            mapa_recorridos.add_trace(go.Scattermapbox(
                lat=[row['latitud'], row['latitud_siguiente']],
                lon=[row['longitud'], row['longitud_siguiente']],
                mode='lines',
                line=dict(width=row['num_dispositivos']),
                name=f"{row['ubicacion']} -> {row['ubicacion_siguiente']}"
            ))
        mapa_recorridos.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=13,
            mapbox_center_lat=df_macs_duplicadas_distintas_ubicaciones['latitud'].mean(),
            mapbox_center_lon=df_macs_duplicadas_distintas_ubicaciones['longitud'].mean(),
            margin={"r":0,"t":0,"l":0,"b":0},
            font={
                'size': 10  
            }
        )


        table_style = {'display': 'block'} if n_clicks else {'display': 'none'}

        
        return (      
            f'Dispositivos detectados:\n {device_count}',
            f'Dispositivos repetidos:\n {duplicated_count}',
            fig_ubicacion,
            fig_jornada,
            folium_map_html,
            mapa_recorridos
        )

def generar_mapa_calor(dff):
    # Crea un mapa centrado en la media de las coordenadas de dff
    mapa = folium.Map(location=[dff['latitud'].mean(), dff['longitud'].mean()], zoom_start=10)

    # Convierte los datos en una lista de listas
    heat_data = [[row['latitud'], row['longitud']] for index, row in dff.iterrows()]

    # Agrega el mapa de calor al mapa
    HeatMap(heat_data).add_to(mapa)

    # Crea un archivo temporal para guardar el mapa
    temp_file = NamedTemporaryFile(delete=False, suffix=".html")
    temp_file.close()
    
    # Guarda el mapa en el archivo temporal
    mapa.save(temp_file.name)
    
    return temp_file.name

if __name__ == '__main__':
    app.run(debug=True)