import streamlit as st
import pandas as pd
import numpy as np
import pickle

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
    prediccion = modelo.predict(datos)
    return prediccion

# # Barra lateral con opciones de países
# st.sidebar.title('Opciones')

# # Selección de país
# pais = st.sidebar.selectbox('Seleccione un país', ['France', 'Spain', 'Germany'])

# # Opciones de entrada para el usuario
# credit_score = st.sidebar.slider('Puntaje de crédito', 300, 850, 500)
# age = st.sidebar.slider('Edad', 18, 100, 40)
# tenure = st.sidebar.slider('Antigüedad', 0, 20, 5)
# balance = st.sidebar.slider('Balance', 0, 250000, 50000)
# num_of_products = st.sidebar.slider('Número de productos', 1, 4, 2)
# gender = st.sidebar.radio('Género', ['Masculino', 'Femenino'])
# is_active_member = st.sidebar.radio('Le gustaría recibir notificaciones de futuras promociones?', ['Si', 'No'])

 
# # Crear DataFrame con datos del usuario
# user_df = pd.DataFrame({
#     'credit_score': [credit_score],
#     'age': [age],
#     'tenure': [tenure],
#     'balance': [balance],
#     'num_of_products': [num_of_products],
#     'is_active': [1 if is_active_member == 'Si' else 0],
#     'gender' : ['Male' if gender == 'Masculino' else 'Female']
# })


# if pais.lower() == 'france':
#     modelo = modelos[pais.lower()]
#     user_df['has_cr_card'] = 1
#     user_df.rename(columns={
#     'credit_score': 'CreditScore',
#     'age': 'Age',
#     'tenure': 'Tenure',
#     'balance': 'Balance',
#     'num_of_products': 'NumOfProducts',
#     'has_cr_card' : 'has_cr_card',
#     'is_active': 'IsActiveMember',
#     'gender': 'Gender'
#     }, inplace=True)
    
#     user_df['CreditCardOwnerTenure'] = user_df['has_cr_card'] * user_df['Age']
#     user_df.drop(columns='has_cr_card', inplace=True)
    
#     columns_ordered = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'CreditCardOwnerTenure', 'IsActiveMember', 'Gender']
#     user_df = user_df.reindex(columns=columns_ordered)
        
# # Ajustar datos según el país seleccionado
# elif pais.lower() == 'spain':
#     gender_code = 1 if gender == 'Masculino' else 0
#     user_df['gender'] = gender_code
#     modelo = modelos[pais.lower()]
    
#     user_df.rename(columns={
#     'credit_score': 'CreditScore',
#     'age': 'Age',
#     'tenure': 'Tenure',
#     'balance': 'Balance',
#     'num_of_products': 'NumOfProducts',
#     'is_active': 'IsActiveMember',
#     'gender': 'is_male'
#     }, inplace=True)
        
# elif pais.lower() == 'germany':
#     modelo = modelos[pais.lower()]
        
#     user_df.rename(columns={
#     'credit_score': 'CreditScore',
#     'age': 'Age',
#     'tenure': 'Tenure',
#     'balance': 'Balance',
#     'num_of_products': 'NumOfProducts',
#     'is_active': 'IsActiveMember',
#     'gender': 'Gender'
#     }, inplace=True)
    
#     user_df['Balance_Tenure_Ratio'] = user_df['Balance'] / (user_df['Tenure'] + 1e-6)
        
#     columns_ordered = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts','Balance_Tenure_Ratio', 'IsActiveMember', 'Gender']
#     user_df = user_df.reindex(columns=columns_ordered)
#     print(user_df)


# # Realizar predicción
# if st.sidebar.button('Predecir Churn'):
#     # Realizar predicción
#     prediction = predecir(modelo, user_df)
#     # Mostrar resultado de la predicción en la barra lateral
#     if prediction == 1:
#         st.sidebar.write(f'El modelo predice que este cliente está en riesgo de churn en {pais}.')
#     else:
#         st.sidebar.write(f'El modelo predice que este cliente no está en riesgo de churn en {pais}.')

# ---------------------------------------------------
# Obtener modelos cargados
modelos = cargar_modelos()

# Obtener datos del usuario
st.title('Predicción de Churn')

if 'df' not in st.session_state:
    st.session_state.df = None

st.subheader("Upload Your CSV File", anchor=False)
uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])
st.divider()

if uploaded_file:
    # Leer el archivo CSV
    if st.session_state.df is None:
        df = pd.read_csv(uploaded_file, sep=';')
        st.session_state.df = df.copy()  # Guardar una copia del DataFrame original
    else:
        df = st.session_state.df.copy()  # Utilizar la copia guardada del DataFrame

    # Mostrar el DataFrame
    st.subheader("🎬 Dataset", anchor=False)
    st.dataframe(df, use_container_width=True)

    # Inicializar contadores para cada país
    churn_counts = {}
    non_churn_counts = {}

    # Dividir los datos por país y aplicar transformaciones
    for pais in df['Geography'].unique():
        pais_df = df[df['Geography'] == pais].copy()

        if pais.lower() == 'france':
            # Aplicar transformaciones para Francia
            pais_df['CreditCardOwnerTenure'] = pais_df['HasCrCard'] * pais_df['Age']
            pais_df.drop(columns=['HasCrCard', 'Geography', 'EstimatedSalary'], inplace=True)
            
            columns_ordered = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'CreditCardOwnerTenure', 'IsActiveMember', 'Gender']
            pais_df = pais_df.reindex(columns=columns_ordered)
        
        elif pais.lower() == 'spain': 
            # Convertir género a valores numéricos (1 para Male, 0 para Female)
            pais_df['Gender'] = pais_df['Gender'].map(lambda x: 1 if x == "Male" else 0)
            pais_df.drop(columns=['HasCrCard', 'Geography', 'EstimatedSalary'], inplace=True)
            
        elif pais.lower() == 'germany':
            # Aplicar transformaciones para Alemania
            pais_df['Balance_Tenure_Ratio'] = pais_df['Balance'] / (pais_df['Tenure'] + 1e-6)
            pais_df.drop(columns=['HasCrCard', 'Geography', 'EstimatedSalary'], inplace=True)
                    
            columns_ordered = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts','Balance_Tenure_Ratio', 'IsActiveMember', 'Gender']
            pais_df = pais_df.reindex(columns=columns_ordered)
        
        # Obtener modelo correspondiente al país
        modelo_pais = modelos[pais.lower()]
        
        # Realizar predicciones y actualizar contadores
        churn_count = 0
        non_churn_count = 0

        for index, row in pais_df.iterrows():
            prediction = predecir(modelo_pais, [row])
            if prediction == 1:
                churn_count += 1
            else:
                non_churn_count += 1
        
        churn_counts[pais] = churn_count
        non_churn_counts[pais] = non_churn_count

    for pais, churn_count in churn_counts.items():
        if churn_count > 0:
            if st.sidebar.button(f'{pais}: {churn_count} Ver registros de clientes en riesgo', key=f'{pais}_churn_button'):
                st.subheader(f'Registros en riesgo de churn en {pais}')
                churn_df = df[df['Geography'] == pais]
                st.dataframe(churn_df)

    
    for pais, non_churn_count in non_churn_counts.items():
        if non_churn_count > 0:
            button_color = 'green'
            st.sidebar.markdown(f'<button style="background-color:{button_color}; color:white">{pais}: {non_churn_count} clientes no están en riesgo de churn</button>',
                        unsafe_allow_html=True)
