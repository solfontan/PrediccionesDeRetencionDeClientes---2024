import pickle
from flask import Flask, render_template, request,session, redirect, url_for
import pandas as pd
import numpy as np
from io import StringIO


app = Flask(__name__, static_folder="static",template_folder="templates")
app.secret_key = "your_secret_key"

# Función para cargar los modelos
def cargar_modelos():
    modelos = {}
    paises = ['france', 'spain', 'germany']
    for pais in paises:
        with open(f'app/models/{pais}.pkl', 'rb') as f:
            modelos[pais] = pickle.load(f)
    return modelos

# Función para realizar la predicción
def predecir(modelo, datos):
    # Realizar la predicción
    prediccion = modelo.predict(datos)
    return prediccion

print('modelos guardados')
modelos = cargar_modelos()

@app.route('/')
def index():
    return render_template("archivotest.html")

def leer_csv(df):
    tabla_html = df.to_html(index=False, classes=['table', 'table-striped', 'table-bordered', 'table-hover'])
    tabla_html = tabla_html.replace(',', '')
    return tabla_html


@app.route('/procesar',  methods=['POST'])
def procesar():
    archivo_csv = request.files['csv_file']
    df = pd.read_csv(archivo_csv, sep=';')
    tabla_html = leer_csv(df)
        
    # Realizar la predicción para Francia
    df_france = df[df['Geography'] == 'France']
    df_spain = df[df['Geography'] == 'Spain']
    df_germany = df[df['Geography'] == 'Germany']
    churn_counts = 0
    churn_counts_spain = 0
    churn_counts_germany = 0
    
    print(df_france)
    print(df_spain)
    print(df_germany)

    if not df_france.empty:
        df_france.loc[:, 'CreditCardOwnerTenure'] = df_france['HasCrCard'] * df_france['Age']
        columns_to_drop = ['HasCrCard', 'Geography', 'EstimatedSalary']  # Eliminar 'id' y otras columnas
        df_france.drop(columns=columns_to_drop, inplace=True)
        df_france.set_index('id', inplace=True)
        print(df_france)
        
        columns_ordered = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'CreditCardOwnerTenure', 'IsActiveMember', 'Gender']
        df_france = df_france.reindex(columns=columns_ordered)
        print(df_france)
        
        
        prediction_array = predecir(modelos['france'], df_france)  
        churn_counts = (prediction_array == 1).sum() # Contar el número de predicciones igual a 1
        print(prediction_array)
        print(churn_counts)

        # Convertir DataFrame a un formato adecuado para la sesión
        if 'datos_usuario' not in session:
            session['datos_usuario'] = []

        for index, row in df_france.iterrows():
            data_to_store = {
                'pre': prediction_array.tolist(),
                'id': index,  # Añadir el índice como parte de los datos
                'datos_usuario': row.to_dict()  # Convertir fila a diccionario
            }
            session['datos_usuario'].append(data_to_store)
            
    if not df_spain.empty:
        df_spain['is_male'] = df_spain['Gender'].map(lambda x: 1 if x == "Male" else 0)
        df_spain.drop(columns=['HasCrCard', 'Geography', 'EstimatedSalary', 'Gender'], inplace=True)
        df_spain.set_index('id', inplace=True)
        print(df_spain)
        
        columns_ordered = ['CreditScore','Age','is_male','Tenure','Balance','NumOfProducts','IsActiveMember']
        df_spain = df_spain.reindex(columns=columns_ordered)
        print(df_spain)
        
        prediction_array_spain = predecir(modelos['spain'], df_spain)  
        churn_counts_spain = (prediction_array_spain == 1).sum() # Contar el número de predicciones igual a 1
        print(prediction_array_spain)
        print(churn_counts_spain)

        # Convertir DataFrame a un formato adecuado para la sesión
        if 'datos_usuario_spain' not in session:
            session['datos_usuario_spain'] = []

        for index_s, row_s in df_spain.iterrows():
            data_to_store_spain = {
                'pre_s': prediction_array_spain.tolist(),
                'id': index_s,  # Añadir el índice como parte de los datos
                'datos_usuario_spain': row_s.to_dict()  # Convertir fila a diccionario
            }
            session['datos_usuario_spain'].append(data_to_store_spain)
            
    if not df_germany.empty:
        df_germany.loc[:,'Balance_Tenure_Ratio'] = df_germany['Balance'] / (df_germany['Tenure'] + 1e-6)
        df_germany.drop(columns=['HasCrCard', 'Geography', 'EstimatedSalary'], inplace=True)
        df_germany.set_index('id', inplace=True)
        print(df_germany)
        
        columns_ordered_g = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts','Balance_Tenure_Ratio', 'IsActiveMember', 'Gender']
        df_germany = df_germany.reindex(columns=columns_ordered_g)
        print(df_germany)
        
        prediction_array_germany = predecir(modelos['germany'], df_germany)  
        churn_counts_germany = (prediction_array_germany == 1).sum() # Contar el número de predicciones igual a 1
        print(prediction_array_germany)
        print(churn_counts_germany)

        # Convertir DataFrame a un formato adecuado para la sesión
        if 'datos_usuario_germany' not in session:
            session['datos_usuario_germany'] = []

        for index_g, row_g in df_germany.iterrows():
            data_to_store_germany = {
                'pre_g': prediction_array_germany.tolist(),  # Obtener la predicción correspondiente para el índice actual
                'id': index_g,  # Añadir el índice como parte de los datos
                'datos_usuario_germany': row_g.to_dict()  # Convertir fila a diccionario
            }
            session['datos_usuario_germany'].append(data_to_store_germany)
            print(session['datos_usuario_germany'])

    return render_template('archivotest.html', churn_counts=churn_counts, churn_counts_spain=churn_counts_spain, churn_counts_germany=churn_counts_germany, tabla_html=tabla_html)


@app.route('/france_page', methods=['GET'])
def france_page():
    datos_usuario_2 = session.get('datos_usuario')  # Corregido de 'datos_usuarios' a 'datos_usuario'
    pre = session.get('pre', [])
    print("datos_usuario:", datos_usuario_2)  # Print to debug
    print("pre:", pre)  # Print to debug

    # Comprueba si hay datos de usuario y si la clave 'pre' está presente en el diccionario
    if datos_usuario_2 is not None and 'pre' in datos_usuario_2[0]:
        # Filtra los datos de usuario donde 'pre' es igual a 1
        datos_usuario_filtrados = [datos_usuario_2[i] for i, d in enumerate(datos_usuario_2) if d['pre'][i] == 1]
        print("datos_usuario_filtrados:", datos_usuario_filtrados)  # Print to debug
    else:
        datos_usuario_filtrados = {}

    return render_template('france_page.html', datos_usuario=datos_usuario_filtrados)


@app.route('/spain_page', methods=['GET'])
def spain_page():
    datos_usuario_spain = session.get('datos_usuario_spain')  # Corregido de 'datos_usuario_spains' a 'datos_usuario_spain'
    pre_s = session.get('pre_s', [])
    print("datos_usuario_spain:", datos_usuario_spain)  # Print to debug
    print("pre_s:", pre_s)  # Print to debug

    # Comprueba si hay datos de usuario y si la clave 'pre' está presente en el diccionario
    if datos_usuario_spain is not None and 'pre_s' in datos_usuario_spain[0]:
        datos_usuario_filtrados_spain = [datos_usuario_spain[i] for i, d in enumerate(datos_usuario_spain) if d['pre_s'][i] == 1]
        print("datos_usuario_filtrados:", datos_usuario_filtrados_spain)  # Print to debug
    else:
        datos_usuario_filtrados_spain = {}

    return render_template('spain_page.html', datos_usuario_spain=datos_usuario_filtrados_spain)


@app.route('/germany_page', methods=['GET'])
def germany_page():
    datos_usuario_germany = session.get('datos_usuario_germany', [])  # Corregido de 'datos_usuario_spains' a 'datos_usuario_spain'
    pre_g = session.get('pre_g', [])
    print("datos_usuario_germany:", datos_usuario_germany)  # Print to debug
    print("pre_g:", pre_g)  # Print to debug

    # Comprueba si hay datos de usuario y si la clave 'pre_g' está presente en el diccionario
    if datos_usuario_germany is not None and 'pre_g' in datos_usuario_germany[0]:
        datos_usuario_filtrados_germany = [datos_usuario_germany[i] for i, d in enumerate(datos_usuario_germany) if d['pre_g'][i] == 1]
        print("datos_usuario_filtrados:", datos_usuario_filtrados_germany)  # Print to debug
    else:
        datos_usuario_filtrados_germany = []

    return render_template('germany_page.html', datos_usuario_germany=datos_usuario_filtrados_germany)




if __name__ == '__main__':
    app.run(debug=True)

