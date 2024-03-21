from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Función para cargar los modelos
def cargar_modelos():
    modelos = {}
    paises = ['france', 'spain', 'germany']
    for pais in paises:
        with open(f'app/models/{pais}.pkl', 'rb') as f:
            modelos[pais] = pickle.load(f)
    return modelos

modelos = cargar_modelos()

# Función para realizar la predicción
def predecir(modelo, datos):
    # Realizar la predicción
    prediccion = modelo.predict(datos)[0]
    print(prediccion)
    return prediccion

# Ruta para mostrar el formulario
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar el formulario y mostrar la predicción
@app.route('/prediccion', methods=['POST'])
def prediccion():
    # Obtener los datos del formulario
    age = request.form['age']
    country = request.form['country']
    gender_code = request.form['gender']
    is_active = request.form['is_active']
    tenure = request.form['tenure']
    balance = request.form['balance']
    credit_score = request.form['credit_score']
    num_of_products = request.form['num_of_products']

    # Guardar los datos en la sesión (opcional)
    has_cr_card = 1
    user_data_dict = {
        'credit_score': int(credit_score),
        'age': age,
        'tenure': int(tenure),
        'balance': float(balance),
        'num_of_products': int(num_of_products),
        'has_cr_card': has_cr_card, 
        'is_active': is_active,
        'gender': gender_code,
        'country': country,
    }
    
    user_df = pd.DataFrame([user_data_dict])
    

    if country.lower() == 'france':
        
        modelo = modelos.get(country)
        user_df.rename(columns={
        'credit_score': 'CreditScore',
        'age': 'Age',
        'tenure': 'Tenure',
        'balance': 'Balance',
        'num_of_products': 'NumOfProducts',
        'has_cr_card' : 'has_cr_card',
        'is_active': 'IsActiveMember',
        'gender': 'Gender'
        }, inplace=True)
        
        user_df['CreditCardOwnerTenure'] = user_df['has_cr_card'] * user_df['Age']
        
        user_df.drop(columns='country', inplace=True)
        
        columns_ordered = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'CreditCardOwnerTenure', 'IsActiveMember', 'Gender']
        user_df = user_df.reindex(columns=columns_ordered)
        print(user_df)
        
        prediction = predecir(modelo, user_df)
        print( prediction)

    elif country.lower() == 'spain':
        user_df['gender'] = 1 if gender_code.lower == 'male' else 0
        
        modelo = modelos.get(country)
        user_df.rename(columns={
        'credit_score': 'CreditScore',
        'age': 'Age',
        'tenure': 'Tenure',
        'balance': 'Balance',
        'num_of_products': 'NumOfProducts',
        'is_active': 'IsActiveMember',
        'gender': 'is_male'
        }, inplace=True)
        
        user_df.drop(columns='country', inplace=True)
        print(user_df.columns)
        prediction = predecir(modelo, user_df)
        print( prediction)
        
    elif country.lower() == 'germany':
        modelo = modelos.get(country)
        
        user_df.rename(columns={
        'credit_score': 'CreditScore',
        'age': 'Age',
        'tenure': 'Tenure',
        'balance': 'Balance',
        'num_of_products': 'NumOfProducts',
        'is_active': 'IsActiveMember',
        'gender': 'Gender'
        }, inplace=True)
        
        user_df['Balance_Tenure_Ratio'] = user_df['Balance'] / (user_df['Tenure'] + 1e-6)
        
        columns_ordered = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts','Balance_Tenure_Ratio', 'IsActiveMember', 'Gender']
        user_df = user_df.reindex(columns=columns_ordered)
        print(user_df)
        
        prediction = predecir(modelo, user_df)
        print( prediction)

    
    else:
        prediction = "No se encontró un modelo para el país especificado"
        

    return render_template('predict_page.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)






