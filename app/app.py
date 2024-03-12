from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Cargar el modelo y el umbral óptimo desde el archivo
with open(r'scr\modelos\prof\paises\france.pkl', 'rb') as f:
    loaded_model_data = pickle.load(f)

# Extraer el modelo y el umbral óptimo del diccionario cargado
loaded_france_model = loaded_model_data['model']
loaded_optimal_threshold = loaded_model_data['threshold']

# Definir la ruta para la página de inicio
@app.route('/')
def home():
    return render_template('index.html')

# # Definir la ruta para procesar los datos del formulario
@app.route('/france_page', methods=['GET', 'POST'])
def modelo():
    print("Accediendo a la ruta '/france_page'")
    if request.method == 'POST':
        # Obtener los datos del formulario
        data = {
            'CreditScore': float(request.form['credit_score']),
            'Age': float(30),
            'Tenure': float(request.form['tenure']),
            'NumOfProducts': float(request.form['num_of_products']),
            'Balance': float(request.form['balance']),
            'EstimatedSalary': float(2000),
            'HasCrCard': 1 if request.form['has_cr_card'] == 'si' else 0,
            'IsActiveMember': 1 if request.form['is_active'] == 'si' else 0,
            'Gender': 1
        }
        
        # Crear un DataFrame con los datos del formulario
        input_data = pd.DataFrame([data])
        print(input_data)

        y_probs_loaded = loaded_france_model.predict_proba(input_data)[:, 1]
        y_pred_optimal_loaded = (y_probs_loaded >= loaded_optimal_threshold).astype(int)

        # Renderizar el template con el resultado de la predicción
        return render_template('/france_page.html', prediction=y_pred_optimal_loaded) 
    
    return render_template('/france_page.html')

if __name__ == '__main__':
    app.run(debug=True)
