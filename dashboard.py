from dash import Dash, html, dash_table, dcc, Input, Output
import pandas as pd
import requests
import plotly.express as px
import datetime
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
import os
from tempfile import NamedTemporaryFile


def obtener_dispositivos():
    url = "https://miserably-touched-gecko.ngrok-free.app/dispositivos/"  # Asegúrate de cambiar la URL según donde esté alojada tu API
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
            #print("Duplicada:", mac)
            #print("Longitud: ", dispositivo['longitud'])
            #print("Latitud: ", dispositivo['latitud'])
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

# df_grouped_fecha = df.groupby('fecha').size().reset_index(name='count')

# df_mañana_tarde = df[df['fecha'] == '2024-05-08']
# df_distinta_semana = df[((df['fecha'] == '2024-05-08') & (df['jornada'] == 'Tarde')) | (df['fecha'] == '2024-05-01')]
# df_misma_semana = df[((df['fecha'] == '2024-05-01') & ((df['ubicacion'] == 'Plaza Santa Ana') | (df['ubicacion'] == 'Plaza del Carmen'))) | ((df['fecha'] == '2024-05-03') & ((df['ubicacion'] == 'Plaza Santa Ana') | (df['ubicacion'] == 'Plaza del Carmen')))]

# # Crear una nueva columna que es True si 'hora_primera' y 'hora_ultima' son iguales, y False de lo contrario
# df_misma_semana['horas_coinciden'] = df_misma_semana['hora_primera'] == df_misma_semana['hora_ultima']

# # Contar cuántas veces cada valor ocurre en la nueva columna
# conteo_horas_coinciden = df_misma_semana['horas_coinciden'].value_counts()



# ANÁLISIS DE DISPOSITIVOS REPETIDOS
# Filtrar los datos para las MACs duplicadas
df_duplicadas = df[df['hashed_mac'].isin(duplicadas)]

df_duplicadas = df_duplicadas[['hashed_mac', 'fecha', 'ubicacion', 'latitud', 'longitud']]

# DISTINTAS UBICACIONES
# Verificar si las MACs duplicadas se encuentran en más de una ubicación
df_duplicadas_ubicaciones = df_duplicadas.groupby('hashed_mac')['ubicacion'].nunique().reset_index()
df_duplicadas_ubicaciones = df_duplicadas_ubicaciones[df_duplicadas_ubicaciones['ubicacion'] > 1]

# Obtener solo las MACs duplicadas que se encontraron en distintas ubicaciones
macs_duplicadas_distintas_ubicaciones = df_duplicadas_ubicaciones['hashed_mac'].tolist()

df_macs_duplicadas_distintas_ubicaciones = df_duplicadas[df['hashed_mac'].isin(macs_duplicadas_distintas_ubicaciones)]

# DISTINTOS DÍAS
# Verificar si las MACs duplicadas se encuentran en más de una fecha
df_duplicadas_fecha = df_duplicadas.groupby('hashed_mac')['fecha'].nunique().reset_index()
df_duplicadas_fecha = df_duplicadas_fecha[df_duplicadas_fecha['fecha'] > 1]

# Obtener solo las MACs duplicadas que se encontraron en distintas ubicaciones
macs_duplicadas_distintas_fechas = df_duplicadas_fecha['hashed_mac'].tolist()

df_macs_duplicadas_distintas_fechas = df_duplicadas[df['hashed_mac'].isin(macs_duplicadas_distintas_fechas)]

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

# # Crear un mapa vacío
# mapa_3 = go.Figure()

# # Para cada recorrido, agregar una traza al mapa
# for i, row in df_recorridos.iterrows():
#     mapa_3.add_trace(go.Scattermapbox(
#         lat=[row['latitud'], row['latitud_siguiente']],
#         lon=[row['longitud'], row['longitud_siguiente']],
#         mode='lines',
#         line=dict(width=row['num_dispositivos']),  # el grosor de la línea es proporcional al número de dispositivos
#         name=f"{row['ubicacion']} -> {row['ubicacion_siguiente']}"
#     ))

# # Configurar el layout del mapa
# mapa_3.update_layout(
#     mapbox_style="open-street-map",
#     mapbox_zoom=10,
#     mapbox_center_lat = df_macs_duplicadas_distintas_ubicaciones['latitud'].mean(),
#     mapbox_center_lon = df_macs_duplicadas_distintas_ubicaciones['longitud'].mean(),
#     margin={"r":0,"t":0,"l":0,"b":0}
# )


app = Dash(__name__, suppress_callback_exceptions=True)

# Layout principal
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Layout de la página principal
index_page = html.Div(
    style={'height': '100vh', 'display': 'grid', 'gridTemplateRows': '10% 45% 45%'}, 
    children=[
        #Título
        html.Div(
            style={'grid-row': '1', 'display': 'flex', 'textAlign': 'center', 'justifyContent': 'center', 'color': 'white','background-color': '#0FA3B1', 'fontSize': 30, 'fontFamily': 'Arial', 'font-weight': 'bold', 'padding': '20px'},
            children='TRABAJO FIN DE GRADO - DASHBOARD DE DISPOSITIVOS'),
        html.Hr(),
        #Sección 1
        html.Div(
            style={'grid-row': '2', 'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '10px', 'padding': '10px'},
            children=[
                html.Div(#style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px', 'width': '50%'}, 
                style={'grid-column': '1', 'display': 'grid', 'gridTemplateRows': '1fr 1fr 1fr', 'gap': '10px'},
                children=[ 
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=df['fecha'].min(),
                        end_date=df['fecha'].max(),
                        display_format='YYYY-MM-DD',
                        style={'margin': 'auto', 'grid-row': '1'}
                    ),
        
                #html.Div(id='output-container-date-picker-range'),
                    html.Div(style={'grid-row': '2', 'display': 'grid', 'gridTemplateRows': '1fr 1fr'},
                        children=[
                            html.Div(id='device-count', style = {'fontSize': 20, 'background-color': '#D9E5D6', 'padding': '10px', 'borderRadius': '5px', 'textAlign': 'center', 'border': '3px solid', 'fontFamily': 'Arial'}),#style={'display': 'table-caption', 'marginBottom': '20px', }),
                            html.A('Consultar todos los datos', id='show-data-table-link', href='/datos', n_clicks=0, style={'textAlign': 'center', 'justifyContent': 'center', 'padding-top': '10px', 'fontFamily': 'Arial'}), #style = {'grid-row': '2'}),
                        ]
                    ),
                
                    html.Div(id='duplicated-count', style={'grid-row': '3','fontSize': 20, 'background-color': '#D9E5D6', 'padding': '10px', 'borderRadius': '5px', 'textAlign': 'center', 'border': '3px solid', 'fontFamily': 'Arial','marginBottom': '50px' }),#style={'display': 'table-caption', 'marginBottom': '20px', }),
                ]),
                dcc.Graph(id='donut-jornada', style={'grid-column': '2', 'alignSelf': 'center', 'width': '100%', 'height': '100%'}),
                html.Iframe(id='folium-map', width='100%', height='100%'), #style={'grid-column': '3', 'alignSelf': 'center', 'width': '100%', 'height': '100%'}),

            ]),
        html.Hr(),
        #Sección 2
        html.Div(style={'grid-row': '3', 'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '10px', 'padding': '10px'},
            children=[
                dcc.Graph(id='mapa_recorridos', style={'grid-column': '1', 'width': '100%', 'height': '100%'}),  
                dcc.Graph(id='histogram-ubicacion', style={'grid-column': '2', 'width': '100%', 'height': '100%'}),
 
        ])
])

# Layout de la página de datos
data_page = html.Div([
    dash_table.DataTable(
        id='table-data',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=[],
        page_size=10
    ),
    html.A('Volver a la página principal', href='/'),
])

@app.callback(
    Output('table-data', 'data'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_table_data(start_date, end_date):
    dff = df[(df['fecha'] >= start_date) & (df['fecha'] <= end_date)]
    return dff.to_dict('records')


# Callback para cambiar el contenido de la página según la URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))

def display_page(pathname):
    if pathname == '/datos':
        return data_page
    else:
        return index_page

# Callback para actualizar gráficos y contadores
@app.callback(
    [#Output('output-container-date-picker-range', 'children'),
     #Output('table-data', 'style'),
     #Output('table-grouped-ubicacion', 'data'),
     Output('device-count', 'children'),
     Output('duplicated-count', 'children'),
     Output('histogram-ubicacion', 'figure'),
     Output ('donut-jornada', 'figure'),
     #Output('mapa-calor', 'figure'),
     Output('folium-map', 'srcDoc'),
     Output('mapa_recorridos', 'figure')],
     [Input('date-picker-range', 'start_date'),
      Input('date-picker-range', 'end_date'),
      Input('show-data-table-link', 'n_clicks')]
)

def update_output(start_date, end_date,n_clicks):
    dff = df[(df['fecha'] >= start_date) & (df['fecha'] <= end_date)]
    if dff.empty:
        return 'No hay datos para las fechas seleccionadas', '', '', '', ''
    else:
        dff_grouped_ubicacion = dff.groupby('ubicacion').size().reset_index(name='count')
        device_count = len(dff)
        duplicadas = encontrar_mac_duplicadas(dff.to_dict('records'))
        duplicated_count = len(duplicadas)
        
        fig_ubicacion = px.histogram(dff, x='ubicacion', y='count', color='fecha', histfunc='sum', nbins=20, title='Histograma de ubicación de los dispositivos', color_discrete_sequence=['#0FA3B1', '#FF9B42', '#EDDEA4'])
        #fig_fecha = px.histogram(dff, x='fecha', y='count', color='ubicacion', histfunc='sum', nbins=20, title='Histograma de fecha de los dispositivos')
        #fig_jornada = px.histogram(dff, x='jornada', y='count', color='ubicacion', histfunc='sum', nbins=20, title='Histograma de hora de los dispositivos')
        fig_ubicacion.update_layout(
            title={
                'text': 'Histograma de ubicación de los dispositivos',
                'font': {'size': 16},  # Cambia el tamaño del título
                'x': 0.5  # Opcional: Centra el título
            },
            xaxis_title={
                'text': '',
                'font': {'size': 1}  # Cambia el tamaño del título del eje X
            },
            yaxis_title={
                'text': '',
                'font': {'size': 1}  # Cambia el tamaño del título del eje Y
            },
            font={
                'size': 10  # Cambia el tamaño del resto del texto (leyendas, etiquetas, etc.)
            }
        )
        fig_jornada= px.pie(dff, names='jornada', hole=0.4, title='Distribución de dispositivos por jornada', color_discrete_sequence=['#0FA3B1', '#FF9B42'])
        fig_jornada.update_layout(
            title={
                'text': 'Distribución de dispositivos por jornada',
                'font': {'size': 16},  # Cambia el tamaño del título
                'x': 0.5  # Opcional: Centra el título
            },
        )
        # fig_mapa_calor = px.density_mapbox(dff, lat='latitud', lon='longitud',
        #     radius=10, center=dict(lat=dff['latitud'].mean(), lon=dff['longitud'].mean()), zoom=10,
        #     mapbox_style="carto-positron", title='Mapa de Calor Espacial', color_continuous_scale='rainbow'
        # )
        mapa = generar_mapa_calor(dff)
        with open(mapa, 'r') as f:
            folium_map_html = f.read()
        
        os.remove(mapa)

        mapa_recorridos = go.Figure()
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
                'size': 10  # Cambia el tamaño del resto del texto (leyendas, etiquetas, etc.)
            }
        )


        table_style = {'display': 'block'} if n_clicks else {'display': 'none'}

        
        return (
            #f'Fechas seleccionadas: {start_date} - {end_date}',
            #dff.to_dict('records'),
            #dff_grouped_ubicacion.to_dict('records'),        
            f'Dispositivos detectados:\n {device_count}',
            f'Dispositivos repetidos:\n {duplicated_count}',
            fig_ubicacion,
            fig_jornada,
            #fig_mapa_calor
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