import pickle
from flask import Flask, render_template, request,session, redirect, flash, url_for
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
    try:
        archivo_csv = request.files['csv_file']
    except KeyError:
        flash('Por favor, sube un archivo CSV.', 'error')
        return redirect('/')

    try:
        df = pd.read_csv(archivo_csv, sep=';')
        tabla_html = leer_csv(df)
    except pd.errors.EmptyDataError:
        flash('El archivo CSV está vacío o no contiene datos.', 'error')
        return render_template('archivotest.html', alert='El archivo CSV está vacío o no contiene datos.')
        
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
        probabilidades_predichas_france = modelos['france'].predict_proba(df_france)
        probabilidades_france =probabilidades_predichas_france[:, 1]
        churn_counts = (prediction_array == 1).sum() # Contar el número de predicciones igual a 1
        print(prediction_array)
        print(probabilidades_france)
        print(churn_counts)
        
        # Convertir DataFrame a un formato adecuado para la sesión
        if 'datos_usuario' not in session:
            session['datos_usuario'] = []

        for index, row in df_france.iterrows():
            data_to_store = {
                'pre': prediction_array.tolist(),
                'pro': probabilidades_france.tolist(),
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
        probabilidades_predichas_spain = modelos['spain'].predict_proba(df_spain)
        probabilidades_spain =probabilidades_predichas_spain[:, 1]
        churn_counts_spain = (prediction_array_spain == 1).sum() # Contar el número de predicciones igual a 1
        print(prediction_array_spain)
        print(probabilidades_spain)
        print(churn_counts_spain)


        # Convertir DataFrame a un formato adecuado para la sesión
        if 'datos_usuario_spain' not in session:
            session['datos_usuario_spain'] = []

        for index_s, row_s in df_spain.iterrows():
            data_to_store_spain = {
                'pre_s': prediction_array_spain.tolist(),
                'pro_s': probabilidades_spain.tolist(),
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
        probabilidades_predichas_germany = modelos['germany'].predict_proba(df_germany)
        probabilidades_germany = np.round(probabilidades_predichas_germany[:, 1],3)
        churn_counts_germany = (prediction_array_germany == 1).sum() # Contar el número de predicciones igual a 1
        print(prediction_array_germany)
        print(probabilidades_germany)
        print(churn_counts_germany)

        # Convertir DataFrame a un formato adecuado para la sesión
        if 'datos_usuario_germany' not in session:
            session['datos_usuario_germany'] = []
            
        for index_g, row_g in df_germany.iterrows():
            data_to_store_germany = {
                'pre_g':  prediction_array_germany.tolist(),  # Obtener la predicción correspondiente para el índice actual
                'pro_g': probabilidades_germany.tolist(),
                'id': index_g,  # Utilizar el ID original
                'datos_usuario_germany': row_g.to_dict()  # Convertir fila a diccionario
            }
            session['datos_usuario_germany'].append(data_to_store_germany)
            print(session['datos_usuario_germany'])

    return render_template('archivotest.html', churn_counts=churn_counts, churn_counts_spain=churn_counts_spain, churn_counts_germany=churn_counts_germany, tabla_html=tabla_html)


#### PÁGINAS CON PREDICCIONES ------------------------------------------------------------

@app.route('/france_page', methods=['GET'])
def france_page():
    datos_usuario = session.get('datos_usuario')  # Corregido de 'datos_usuarios' a 'datos_usuario'
    pre = session.get('pre', [])
    pro = session.get('pro', [])
    print("datos_usuario:", datos_usuario)  # Print to debug
    print("pre:", pre) 
    print("pro:", pro) 


    if datos_usuario is not None and 'pre' in datos_usuario[0]:     
        datos_usuario_filtrados = [
            {
            'id': d['id'],
            'datos_usuario': d['datos_usuario'],
            'pro': round(d['pro'][index], 3)
        }
        for index, d in enumerate(datos_usuario)
        if d['pre'][index] == 1
        ]

    else:
        datos_usuario_filtrados = {}

    return render_template('france_page.html', datos_usuario=datos_usuario_filtrados)




@app.route('/spain_page', methods=['GET'])
def spain_page():
    datos_usuario_spain = session.get('datos_usuario_spain')  # Corregido de 'datos_usuario_spains' a 'datos_usuario_spain'
    pre_s = session.get('pre_s', [])
    pro_s = session.get('pro_s', [])
    print("datos_usuario_spain:", datos_usuario_spain)  
    print("pre_s:", datos_usuario_spain[0]['pre_s'][0])  
    print("pro_s:", pro_s) 

    if datos_usuario_spain is not None and all('pre_s' in d for d in datos_usuario_spain):
       datos_usuario_filtrados_spain = [
            {
                'id': d['id'],
                'datos_usuario_spain': d['datos_usuario_spain'],
                'pro_s': round(d['pro_s'][index], 3)  # Redondear el primer elemento de pro_s
            }
            for index, d in enumerate(datos_usuario_spain)
            if d['pre_s'][index] == 1  # Verificar si el valor 1 está presente en la lista pre_s
        ]
    else:
        datos_usuario_filtrados_spain = {}

    return render_template('spain_page.html', datos_usuario_spain=datos_usuario_filtrados_spain)




@app.route('/germany_page', methods=['GET'])
def germany_page():
    datos_usuario_germany = session.get('datos_usuario_germany', [])  
    pre_g = session.get('pre_g', [])
    pro_g = session.get('pro_g', [])
    print("datos_usuario_germany:", datos_usuario_germany)  # Print to debug
    print("pre_g:", pre_g)  # Print to debug
    print("pro_g:", pro_g) 

        # Comprueba si hay datos de usuario y si la clave 'pre_g' está presente en el diccionario
    if datos_usuario_germany is not None and 'pre_g' in datos_usuario_germany[0]:
        datos_usuario_filtrados_germany = [
        {
            'id': d['id'],
            'datos_usuario_germany': d['datos_usuario_germany'],
            'pro_g': round(d['pro_g'][index], 3)  
        }
        for index, d in enumerate(datos_usuario_germany)
        if d['pre_g'][index] == 1 
    ]
        
    else:
        datos_usuario_filtrados_germany = []


    return render_template('germany_page.html', datos_usuario_germany=datos_usuario_filtrados_germany)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=2424)

