from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Cargar el modelo y el umbral óptimo desde el archivo
with open(r'app\france.pkl', 'rb') as f:
    loaded_model_data = pickle.load(f)

# Extraer el modelo y el umbral óptimo del diccionario cargado
loaded_france_model = loaded_model_data['model']
loaded_optimal_threshold = loaded_model_data['threshold']

# Cargar el modelo y el umbral óptimo desde el archivo
with open(r'app\spain.pkl', 'rb') as f:
    loaded_model_data_spain = pickle.load(f)

# Extraer el modelo y el umbral óptimo del diccionario cargado
loaded_spain_model = loaded_model_data_spain['model']
loaded_optimal_threshold_spain = loaded_model_data_spain['threshold']

# Cargar el modelo y el umbral óptimo desde el archivo
with open(r'app\german.pkl', 'rb') as f:
    loaded_model_data_german = pickle.load(f)

# Extraer el modelo y el umbral óptimo del diccionario cargado
loaded_german_model = loaded_model_data_german['model']
loaded_optimal_threshold_german = loaded_model_data_german['threshold']


# Definir la ruta para la página de inicio
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index', methods=['GET', 'POST'])
def modelo():
    print("Accediendo a la ruta '/index'")
    if request.method == 'POST':
        # Obtener los datos del formulario
        data = {
            'CreditScore': float(request.form['credit_score']),
            'Age': float(request.form['age']),
            'Tenure': float(request.form['tenure']),
            'Balance': float(request.form['balance']),
            'NumOfProducts': float(request.form['num_of_products']),
            'EstimatedSalary': float(request.form['estimated_salary']),
            'HasCrCard': 1 if request.form['has_cr_card'] == 'si' else 0,
            'IsActiveMember': 1 if request.form['is_active'] == 'si' else 0
        }
                
        # Crear un DataFrame con los datos del formulario
        input_data = pd.DataFrame([data])
        print(input_data)

        # Seleccionar el modelo adecuado según el país seleccionado
        country = request.form['country']
        if country == 'france':
            gender = request.form['gender'].lower()
            if gender == 'female':
                input_data['Gender'] = 'Female'
            elif gender == 'male':
                input_data['Gender'] = 'Male'
                
            loaded_model = loaded_france_model
            optimal_threshold = loaded_optimal_threshold
            print(input_data)  
            print(loaded_model.predict(input_data))
        elif country == 'spain':
            gender = request.form['gender'].lower()
            if gender == 'female':
                input_data['Gender'] = 0
            elif gender == 'male':
                input_data['Gender'] = 1
                
            loaded_model = loaded_spain_model
            optimal_threshold = loaded_optimal_threshold_spain
            print(loaded_model)     
            print(input_data)         
            print(loaded_model.predict(input_data))
        elif country == 'germany':
            gender = request.form['gender'].lower()
            if gender == 'female':
                input_data['Gender'] = 'Female'
            elif gender == 'male':
                input_data['Gender'] = 'Male'
                
            loaded_model = loaded_german_model
            optimal_threshold = loaded_optimal_threshold_german
            print(input_data)  
            print(loaded_model.predict(input_data))
        else:
            return render_template('index.html', error='País no válido')  # Manejo de errores si el país no es válido
        
        # Realizar la predicción utilizando el modelo seleccionado
        y_probs_loaded = loaded_model.predict_proba(input_data)[:, 1]
        y_pred_optimal_loaded = (y_probs_loaded >= optimal_threshold).astype(int)
        
        print(optimal_threshold)
        print(y_probs_loaded)
        print(y_pred_optimal_loaded)
        
        if y_pred_optimal_loaded[0] == 1:
            return redirect(url_for('alerta_abandono'))
        else:
            return redirect(url_for('home'))

    return render_template('index.html')


@app.route('/alerta_abandono')  # Cambia '/bank_page' a '/alerta_abandono'
def alerta_abandono():
    # Renderiza la página de alerta de abandono
    return render_template('bank_page.html')


if __name__ == '__main__':
    app.run(debug=True)
