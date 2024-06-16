from dash import Dash, html, dash_table, dcc
import pandas as pd
import requests
import plotly.express as px
import datetime
import plotly.graph_objects as go


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
df_grouped_fecha = df.groupby('fecha').size().reset_index(name='count')

df_mañana_tarde = df[df['fecha'] == '2024-05-08']
df_distinta_semana = df[((df['fecha'] == '2024-05-08') & (df['jornada'] == 'Tarde')) | (df['fecha'] == '2024-05-01')]
df_misma_semana = df[((df['fecha'] == '2024-05-01') & ((df['ubicacion'] == 'Plaza Santa Ana') | (df['ubicacion'] == 'Plaza del Carmen'))) | ((df['fecha'] == '2024-05-03') & ((df['ubicacion'] == 'Plaza Santa Ana') | (df['ubicacion'] == 'Plaza del Carmen')))]

# Crear una nueva columna que es True si 'hora_primera' y 'hora_ultima' son iguales, y False de lo contrario
df_misma_semana['horas_coinciden'] = df_misma_semana['hora_primera'] == df_misma_semana['hora_ultima']

# Contar cuántas veces cada valor ocurre en la nueva columna
conteo_horas_coinciden = df_misma_semana['horas_coinciden'].value_counts()



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

#MAPA 1

""" fig = px.scatter_mapbox(df_macs_duplicadas_distintas_ubicaciones, lat='latitud', lon='longitud', color="hashed_mac", 
                        color_continuous_scale=px.colors.cyclical.IceFire, 
                        size_max=100, zoom=10)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}) """


#MAPA 2

# Ordenar el DataFrame
df_macs_duplicadas_distintas_ubicaciones = df_macs_duplicadas_distintas_ubicaciones.sort_values(['hashed_mac', 'fecha'])

# Crear un mapa vacío
mapa_2 = go.Figure()

# Obtener una lista de todas las 'hashed_mac' únicas
macs = df_macs_duplicadas_distintas_ubicaciones['hashed_mac'].unique()

# Para cada 'hashed_mac' única, agregar una traza al mapa
for mac in macs:
    df_mac = df_macs_duplicadas_distintas_ubicaciones[df_macs_duplicadas_distintas_ubicaciones['hashed_mac'] == mac]
    mapa_2.add_trace(go.Scattermapbox(
        lat=df_mac['latitud'],
        lon=df_mac['longitud'],
        mode='lines+markers+text',
        name=mac,
        text=df_mac['ubicacion'],
        textposition='bottom right'
    ))

# Configurar el layout del mapa
mapa_2.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=10,
    mapbox_center_lat = df_macs_duplicadas_distintas_ubicaciones['latitud'].mean(),
    mapbox_center_lon = df_macs_duplicadas_distintas_ubicaciones['longitud'].mean(),
    margin={"r":0,"t":0,"l":0,"b":0}
)

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

# Crear un mapa vacío
mapa_3 = go.Figure()

# Para cada recorrido, agregar una traza al mapa
for i, row in df_recorridos.iterrows():
    mapa_3.add_trace(go.Scattermapbox(
        lat=[row['latitud'], row['latitud_siguiente']],
        lon=[row['longitud'], row['longitud_siguiente']],
        mode='lines',
        line=dict(width=row['num_dispositivos']),  # el grosor de la línea es proporcional al número de dispositivos
        name=f"{row['ubicacion']} -> {row['ubicacion_siguiente']}"
    ))

# Configurar el layout del mapa
mapa_3.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=10,
    mapbox_center_lat = df_macs_duplicadas_distintas_ubicaciones['latitud'].mean(),
    mapbox_center_lon = df_macs_duplicadas_distintas_ubicaciones['longitud'].mean(),
    margin={"r":0,"t":0,"l":0,"b":0}
)


app = Dash()

app.layout = [
    html.Div(children='TRABAJO FIN DE GRADO - DASHBOARD DE DISPOSITIVOS', style={'textAlign': 'center', 'color': 'blue', 'fontSize': 30}),
    html.Hr(),
    html.Div(children='Datos obtenidos'),
    html.Hr(),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    html.Div(children='Ubicaciones'),
    html.Hr(),
    dash_table.DataTable(data=df_grouped_ubicacion.to_dict('records'), page_size=50),
    html.Div(children=f'Número de dispositivos detectados: {len(df)}'),
    html.Hr(),
    dcc.Graph(figure=px.histogram(df, x='ubicacion', y='count',color='fecha', histfunc='sum', nbins=20, title='Histograma de ubicación de los dispositivos')),
    dcc.Graph(figure=px.histogram(df, x='fecha', y='count',color='ubicacion', histfunc='sum', nbins=20, title='Histograma de fecha de los dispositivos')),
    dcc.Graph(figure=px.histogram(df, x='jornada', y='count', color='ubicacion', histfunc='sum', nbins=20, title='Histograma de hora de los dispositivos')),
    dcc.Graph(figure=px.histogram(df_mañana_tarde, x='jornada', y='count', color='ubicacion', histfunc='sum', nbins=20, title='Comparando mañana y tarde un mismo día')),
    dcc.Graph(figure=px.histogram(df_distinta_semana, x='fecha', y='count', color='ubicacion', histfunc='sum', nbins=20, title='Comparando el mismo día y hora en semanas distintas')),
    dcc.Graph(figure=px.histogram(df_misma_semana, x='fecha', y='count', color='ubicacion', histfunc='sum', nbins=20, title='Comparando distintos días en la misma semana')),
    dcc.Graph(figure=px.histogram(df_misma_semana, x='horas_coinciden', y='count', color='fecha', histfunc='sum', nbins=20, title='Comparando si las horas coinciden en un mismo día')),
    html.Div(children=f'Número de dispositivos repetidos: {len(duplicadas)}'),
    html.Hr(),
    dash_table.DataTable(data=df_duplicadas.to_dict('records'), page_size=10),
    html.Hr(),
    html.Div(children=f'Número de dispositivos repetidos en distintas ubicaciones: {len(macs_duplicadas_distintas_ubicaciones)}'),
    dash_table.DataTable(data=df_macs_duplicadas_distintas_ubicaciones.to_dict('records'), page_size=10),
    html.Div(children='Mapa de dispositivos repetidos en distintas ubicaciones'),
    dcc.Graph(figure=mapa_2),
    html.Div(children='Mapa de dispositivos repetidos en distintas ubicaciones, cambiando el grosor de la línea según el número de dispositivos repetidos en ese recorrido'),
    dcc.Graph(figure=mapa_3),
    html.Div(children=f'Número de dispositivos repetidos en distintas fechas: {len(macs_duplicadas_distintas_fechas)}'),
    dash_table.DataTable(data=df_macs_duplicadas_distintas_fechas.to_dict('records'), page_size=10),

]

if __name__ == '__main__':
    app.run(debug=True)