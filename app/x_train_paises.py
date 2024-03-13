### Manipulación de Datos
import pandas as pd
import numpy as np

### Visualización de Datos
import seaborn as sns
import matplotlib.pyplot as plt

### Machine Learning
# Preparación de datos 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from sklearn.decomposition import PCA

# Modelos
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import  classification_report, precision_recall_curve, f1_score
import pickle

# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('scr\data\processed\Churn_Modelling.csv')

df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

df_france = df[df['Geography'] == 'France']
df_spain = df[df['Geography'] == 'Spain']
df_germany = df[df['Geography'] == 'Germany']


# ---- FRANCE------------------------------------------------------------------------------

# Definir las variables X e y
X = df_france.drop(columns=['Exited']) 
y = df_france['Exited']

numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',  'EstimatedSalary']
categ = ['Gender']

# Definir transformadores para características numéricas y categóricas
numeric_transformer = MinMaxScaler()

# Crear un ColumnTransformer para aplicar transformaciones a diferentes columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)

# Crear el pipeline con el preprocesador
pipeline = make_pipeline(preprocessor)

X_processed = pipeline.fit_transform(X)

# Crear DataFrame con los datos procesados y los nombres de las columnas
processed_df = pd.DataFrame(X_processed, columns=numeric_features)
concatenated_series  = pd.concat((processed_df, X[[ 'HasCrCard', 'IsActiveMember', 'Gender']].reset_index(drop=True)), axis=1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(concatenated_series, y, test_size=0.2, random_state=24)

# # Aplicar SMOTE solo al conjunto de entrenamiento
smote = SMOTENC(random_state=24, categorical_features= categ)
X_train_resampled_france, y_train_resampled_france = smote.fit_resample(X_train, y_train) 

# Inicializar el modelo de Random Forest
france_model = CatBoostClassifier(silent=True, iterations= 500, bootstrap_type = 'MVS', bagging_temperature= 10, eval_metric='Precision', learning_rate=0.01)

# Ejecutar el grid search en los datos de entrenamiento
france_model.fit(X_train_resampled_france, y_train_resampled_france, cat_features=categ)

# Obtener las probabilidades de predicción para la clase positiva
y_probs = france_model.predict_proba(X_test)[:, 1]

# Calcular la precisión, el recall y los umbrales utilizando precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Calcular el F1-score para cada umbral
f1_scores = 2 * (precision * recall) / (precision + recall)

# Encontrar el índice del umbral que maximiza el F1-score
best_threshold_index = f1_scores.argmax()

# Seleccionar el umbral óptimo
optimal_threshold = thresholds[best_threshold_index]

# Aplicar el umbral óptimo para convertir las probabilidades de predicción en etiquetas de clase
y_pred_optimal = (y_probs >= optimal_threshold).astype(int)

# Mostrar el reporte de clasificación
print("Reporte de clasificación para Francia:")
print(classification_report(y_test, y_pred_optimal))


print(concatenated_series.head(2))

# Guardar el modelo y el umbral óptimo en un diccionario
model_data = {'model': france_model, 'threshold': optimal_threshold}

# Guardar el modelo y el umbral óptimo en un archivo
with open('app/france.pkl', 'wb') as f:
    pickle.dump(model_data, f)  

# ------ Spain ----------------------------------------------------------------------------------------------------

df_spain['is_male'] = df_spain['Gender'].replace({'Male': 1, 'Female': 0})

numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
variables_sin_modificacion = ['HasCrCard', 'IsActiveMember', 'Exited', 'is_male']

# Definir transformadores para características numéricas y categóricas
numeric_transformer = MinMaxScaler()

# Crear un ColumnTransformer para aplicar transformaciones a diferentes columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)

# Crear el pipeline con el preprocesador
pipeline = make_pipeline(preprocessor)
# Aplicar el preprocesador a los datos
X_processed = pipeline.fit_transform(df_spain)

# # Crear DataFrame con los datos procesados y los nombres de las columnas
processed_df_spain = pd.DataFrame(X_processed, columns=numeric_features)
concatenated_series_spain  = pd.concat((processed_df_spain, df_spain[['HasCrCard', 'IsActiveMember', 'Exited', 'is_male']].reset_index(drop=True)), axis=1)

X = concatenated_series_spain.drop(columns='Exited')
y = concatenated_series_spain['Exited']

# # # Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# # # Aplicar SMOTE solo al conjunto de entrenamiento
smote = SMOTE(random_state=24)
X_train_resampled_spain, y_train_resampled_spain = smote.fit_resample(X_train , y_train)

model_spain = lgb.LGBMClassifier(verbosity= -1, num_leaves=70, learning_rate=0.005,  min_child_samples=20, max_depth=40, is_unbalance=True, min_split_gain=0.1)

# Ejecutar el grid search en los datos de entrenamiento
model_spain.fit(X_train_resampled_spain, y_train_resampled_spain)

# Obtener las probabilidades de predicción para la clase positiva
y_probs = model_spain.predict_proba(X_test)[:, 1]

# Calcular la precisión, el recall y los umbrales utilizando precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Calcular el F1-score para cada umbral
f1_scores = 2 * (precision * recall) / (precision + recall)

# Encontrar el índice del umbral que maximiza el F1-score
best_threshold_index = f1_scores.argmax()

# Seleccionar el umbral óptimo
optimal_threshold_spain = thresholds[best_threshold_index]

# Aplicar el umbral óptimo para convertir las probabilidades de predicción en etiquetas de clase
y_pred_optimal = (y_probs >= optimal_threshold_spain).astype(int)

print("Reporte de clasificación para Spain:")
print(classification_report(y_test, y_pred_optimal))

print(concatenated_series_spain.head(2))

# Guardar el modelo y el umbral óptimo en un diccionario
model_data_spain = {'model': model_spain, 'threshold': optimal_threshold_spain}

# Guardar el modelo y el umbral óptimo en un archivo
with open('app/spain.pkl', 'wb') as f:
    pickle.dump(model_data_spain, f)  
    

# ------------ Germany ---------------------------------------------------------------------------------------------


X = df_germany.drop(columns='Exited')
y = df_germany['Exited']

numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
variables_sin_modificacion =  df_germany[[ 'HasCrCard', 'IsActiveMember', 'Gender']]
categ = ['Gender']

# Definir transformadores para características numéricas y categóricas
numeric_transformer = MinMaxScaler()

# Crear un ColumnTransformer para aplicar transformaciones a diferentes columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)
# Crear el pipeline con el preprocesador
pipeline = make_pipeline(preprocessor)
# Aplicar el preprocesador a los datos
X_processed = pipeline.fit_transform(X)

# Crear DataFrame con los datos procesados y los nombres de las columnas
processed_df = pd.DataFrame(X_processed, columns=numeric_features)
concatenated_series_germany  = pd.concat([processed_df, variables_sin_modificacion.reset_index(drop=True) ], axis=1) #

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(concatenated_series_germany, y, test_size=0.2, random_state=24)

# Aplicar SMOTE solo al conjunto de entrenamiento
smote = SMOTENC(random_state=24, categorical_features= categ)
X_train_resampled_germany, y_train_resampled_germany = smote.fit_resample(X_train, y_train)

model_german = CatBoostClassifier(silent=True, iterations= 500, bagging_temperature= 10, eval_metric='Recall', learning_rate=0.01)

model_german.fit(X_train_resampled_germany, y_train_resampled_germany, cat_features = categ)

# Obtener las probabilidades de predicción para la clase positiva
y_probs = model_german.predict_proba(X_test)[:, 1]

# Calcular la precisión, el recall y los umbrales utilizando precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Calcular el F1-score para cada umbral
f1_scores = 2 * (precision * recall) / (precision + recall)

# Encontrar el índice del umbral que maximiza el F1-score
best_threshold_index = f1_scores.argmax()

# Seleccionar el umbral óptimo
optimal_threshold_german = thresholds[best_threshold_index]

# Aplicar el umbral óptimo para convertir las probabilidades de predicción en etiquetas de clase
y_pred_optimal = (y_probs >= optimal_threshold_german).astype(int)

print("Reporte de clasificación para Germany:")
print(classification_report(y_test, y_pred_optimal))

print(concatenated_series_germany.head(2))

# Guardar el modelo y el umbral óptimo en un diccionario
model_data_german = {'model': model_german, 'threshold': optimal_threshold_german}

# Guardar el modelo y el umbral óptimo en un archivo
with open('app/german.pkl', 'wb') as f:
    pickle.dump(model_data_german, f)  