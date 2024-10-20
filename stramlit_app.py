import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

# Configuración y carga de datos
def cargar_datos(file_path, delimiter=','):
    """
    Carga un archivo CSV en un DataFrame.
    """
    return pd.read_csv(file_path, delimiter=delimiter)

# Ruta del archivo (relativa)
file_path_colombia = os.path.join(os.getcwd(), 'Org_Datos_Colombia.csv')
df_colombia = cargar_datos(file_path_colombia)

# Identificación de género
def determinar_genero(nombre, nombres_femeninos):
    primer_nombre = nombre.split()[0].upper()
    return 'Femenino' if primer_nombre in nombres_femeninos else 'Masculino'

# Lista de nombres femeninos
nombres_femeninos = ['DANIELA', 'GLORIA', 'JULIETH', 'LILIANA', 'YINETH', 'YENNY', 
                     'LEIDY', 'YURLEY', 'DIVIANA', 'SAIDA', 'DEISY', 'JENNIFFER', 
                     'DEICY', 'NANCY', 'ANYELA', 'ERIKA', 'NORVY', 'CECILIA', 'ARACELY', 
                     'CARMEN', 'MIRIAM', 'YORLEY', 'LAURA', 'YANEIRA', 'LISBETH', 
                     'ALEXANDRA', 'DIANA', 'CLAUDIA', 'RUBY', 'LUCIA']

# Aplicar la función para identificar género
df_colombia['GENERO'] = df_colombia['NOMBRES Y APELLIDOS'].apply(lambda x: determinar_genero(x, nombres_femeninos))

# Asignar valores numéricos al género
df_colombia['GENERO_NUMERICO'] = np.where(df_colombia['GENERO'] == 'Femenino', 0, 1)

# Función para convertir escala salarial
SMLV = 1_300_000
def convertir_smlv(escala, smlv=SMLV):
    try:
        escala = escala.strip().upper()
        if re.match(r'^\d+\s*-\s*SMLV$', escala):
            numero = re.search(r'\d+', escala)
            return int(numero.group(0)) * smlv if numero else None
        elif '-' in escala:
            numeros = re.findall(r'\d+', escala)
            if len(numeros) == 2:
                val_min, val_max = map(int, numeros)
                promedio = (val_min + val_max) / 2
                return promedio * smlv
        else:
            numero = re.search(r'\d+', escala)
            return int(numero.group(0)) * smlv if numero else None
    except Exception as e:
        print(f"Error procesando la escala: {escala} - {e}")
    return None

# Aplicar la conversión de escala salarial
df_colombia['SALARIO_CALCULADO'] = df_colombia['ESCALA SALARIAL SEGÚN LAS CATEGORÍAS PARRA SERVIDORES PÚBLICOS Y/O EMPREADOS DEL SECTOR PRIVADO.'].apply(lambda x: convertir_smlv(str(x)))

# Renombrar columnas
df_colombia.rename(columns={
    'ESCALA SALARIAL SEGÚN LAS CATEGORÍAS PARRA SERVIDORES PÚBLICOS Y/O EMPREADOS DEL SECTOR PRIVADO.': 'ESCALA_SALARIO',
}, inplace=True)

# Lista de niveles de educación en el orden deseado
niveles_educacion = ['PRIMARIA', 'BACHILLER', 'MEDIA BOCACIONAL', 'TÉCNICO', 
                     'TECNOLOGO', 'PASANTE SENA', 'PASANTE UNIVERSITARIO', 'PROFESIONAL']

# Crear un diccionario que asigne valores del 1 al 8 a cada nivel
educacion_dict = {nivel: valor for valor, nivel in enumerate(niveles_educacion, start=1)}

# Crear una nueva columna con el valor numérico correspondiente al nivel de educación
df_colombia['NIVEL EDUCACION ORDENADO'] = df_colombia['FORMACIÓN ACADEMICA'].map(educacion_dict)

# Filtrar columnas importantes
df_colombia = df_colombia[['NUMERO', 'NOMBRES Y APELLIDOS', 'GENERO', 'GENERO_NUMERICO', 'FORMACIÓN ACADEMICA', 'NIVEL EDUCACION ORDENADO', 'ESCALA_SALARIO', 'SALARIO_CALCULADO']]

# Análisis y agrupación de datos
def calcular_estadisticas_genero(df):
    return df.groupby('GENERO')['SALARIO_CALCULADO'].agg(promedio='mean').reset_index()

# Calcular las estadísticas de salario por género
estadisticas_genero = calcular_estadisticas_genero(df_colombia)

# Obtener los salarios promedio de hombres y mujeres
salario_promedio_mujeres = estadisticas_genero[estadisticas_genero['GENERO'] == 'Femenino']['promedio'].values[0]
salario_promedio_hombres = estadisticas_genero[estadisticas_genero['GENERO'] == 'Masculino']['promedio'].values[0]

# Calcular la diferencia absoluta y el porcentaje
diferencia_salario = salario_promedio_hombres - salario_promedio_mujeres
diferencia_porcentual = (diferencia_salario / salario_promedio_mujeres) * 100

# Mostrar resultados de salario promedio
st.markdown("<h1 style='text-align: center;'>Resultados de Colombia:</h1>", unsafe_allow_html=True)
st.header("Estadísticas de Colombia:")
st.metric("Salario promedio de mujeres", f"{salario_promedio_mujeres:.2f}")
st.metric("Salario promedio de hombres", f"{salario_promedio_hombres:.2f}")
st.metric("Diferencia absoluta de salario (Hombres - Mujeres)", f"{diferencia_salario:.2f}")
st.metric("Diferencia porcentual", f"{diferencia_porcentual:.2f}%")

st.header("Correlograma Colombia:")
def generar_correlograma(df, title):
    st.write("Columnas del DataFrame:", df.columns.tolist())
    try:
        df_selected = df[['GENERO_NUMERICO', 'NIVEL EDUCACION ORDENADO', 'SALARIO_CALCULADO']]
    except KeyError as e:
        st.error(f"Error: {str(e)}")
        return

    correlation_matrix = df_selected.corr()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    
    st.pyplot(plt)
    plt.clf()

# Generar correlograma
generar_correlograma(df_colombia, 'Correlograma de Datos Completos')

# Cargar otro archivo CSV
file_path_espana = os.path.join(os.getcwd(), 'datos.csv')
df_espana = cargar_datos(file_path_espana, delimiter=';')  # Cambiar el delimitador a punto y coma

# Verificar las columnas disponibles en df_espana
st.write("Columnas en df_espana:", df_espana.columns.tolist())

# Filtrar las filas donde 'Sexo/Brecha de género' no contenga "Cociente" y donde 'Tipo de jornada' no sea 'Total'
if 'Sexo/Brecha de género' in df_espana.columns and 'Tipo de jornada' in df_espana.columns:
    df_filtered = df_espana[
        (~df_espana['Sexo/Brecha de género'].str.contains('Cociente', case=False, na=False)) &
        (df_espana['Tipo de jornada'] != 'Total')
    ]

    # Limpiar la columna 'Total', reemplazar comas por puntos y convertir a numérico
    df_filtered['Total'] = pd.to_numeric(df_filtered['Total'].str.replace(',', '.'), errors='coerce')

    # Agrupar por 'Tipo de jornada' y 'Sexo/Brecha de género', y calcular el promedio de 'Total'
    promedio_total = df_filtered.groupby(['Tipo de jornada', 'Sexo/Brecha de género'])['Total'].mean().reset_index()

    # Asignar valores numéricos al género
    promedio_total['GENERO_NUMERICO'] = np.where(promedio_total['Sexo/Brecha de género'] == 'Mujeres', 0, 1)

    # Mostrar resultados de España
    st.markdown("<h1 style='text-align: center;'>Resultados de España:</h1>", unsafe_allow_html=True)
    st.header("Estadísticas de España:")

    # Cálculo de la diferencia salarial por jornada
    def calcular_diferencias_salariales(promedio_total):
        diferencias = []

        for jornada in promedio_total['Tipo de jornada'].unique():
            jornada_data = promedio_total[promedio_total['Tipo de jornada'] == jornada]
            salario_hombres = jornada_data[jornada_data['Sexo/Brecha de género'] == 'Hombres']['Total'].values[0]
            salario_mujeres = jornada_data[jornada_data['Sexo/Brecha de género'] == 'Mujeres']['Total'].values[0]

            diferencia_salarial = salario_hombres - salario_mujeres
            diferencia_porcentual = (diferencia_salarial / salario_mujeres) * 100

            diferencias.append({
                'Tipo de jornada': jornada,
                'Salario Hombres': salario_hombres,
                'Salario Mujeres': salario_mujeres,
                'Diferencia Salarial': diferencia_salarial,
                'Diferencia Porcentual (%)': diferencia_porcentual
            })

        return pd.DataFrame(diferencias)

    # Calcular las diferencias salariales
    diferencias_salariales = calcular_diferencias_salariales(promedio_total)

    # Mostrar los resultados en una tabla
    st.subheader("Diferencias Salariales por Tipo de Jornada")
    st.dataframe(diferencias_salariales)

    st.header("Correlograma España:")

    def generar_correlograma_filtrado(df, title):
        correlation_matrix = df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(title)
        st.pyplot(plt)
        plt.clf()

    # Correlograma del DataFrame filtrado
    st.write("Columnas del DataFrame filtrado:", df_filtered.columns.tolist())
    generar_correlograma_filtrado(df_filtered[['Total', 'GENERO_NUMERICO']], 'Correlograma de Datos Filtrados por Género')
else:
    st.error("Las columnas 'Sexo/Brecha de género' o 'Tipo de jornada' no se encuentran en el DataFrame.")
