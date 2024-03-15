### Manipulación de Datos
import pandas as pd
import numpy as np

### Machine Learning
# Preparación de datos 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC

# Modelos
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import  classification_report
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

df_france['CreditCardOwnerTenure'] = df_france.HasCrCard * df_france.Age

# Definir las variables X e y
X = df_france.drop(columns=['Exited']) 
y = df_france['Exited']

numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'CreditCardOwnerTenure']
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
concatenated_series  = pd.concat((processed_df, X[['IsActiveMember', 'Gender']].reset_index(drop=True)), axis=1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(concatenated_series, y, test_size=0.2, random_state=24)

# # Aplicar SMOTE solo al conjunto de entrenamiento
smote = SMOTENC(random_state=24, categorical_features= categ)
X_train_resampled_france, y_train_resampled_france = smote.fit_resample(X_train, y_train) 

# Inicializar el modelo de Random Forest
france_model = CatBoostClassifier(silent=True, iterations= 300, bagging_temperature= 5, eval_metric='Precision', learning_rate=0.05, auto_class_weights='Balanced', reg_lambda=0.01)

# Ejecutar el grid search en los datos de entrenamiento
france_model.fit(X_train_resampled_france, y_train_resampled_france, cat_features=categ)

# Mostrar el reporte de clasificación
print("Reporte de clasificación para Francia:")
print(classification_report(y_test, france_model.predict(X_test)))

# # Guardar el modelo y el umbral óptimo en un archivo
with open(r'scr\modelos\mejores_modelos\france.pkl', 'wb') as f:
    pickle.dump(france_model, f)  

# ------ Spain ----------------------------------------------------------------------------------------------------

df_spain['is_male'] = df_spain['Gender'].replace({'Male': 1, 'Female': 0})

numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts']
variables_sin_modificacion = ['IsActiveMember', 'Exited', 'is_male']

# Definir transformadores para características numéricas y categóricas
numeric_transformer = StandardScaler()

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
concatenated_series_spain  = pd.concat((processed_df_spain, df_spain[['IsActiveMember', 'Exited', 'is_male']].reset_index(drop=True)), axis=1)

X = concatenated_series_spain.drop(columns='Exited')
y = concatenated_series_spain['Exited']

# # # Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# # # Aplicar SMOTE solo al conjunto de entrenamiento
smote = SMOTE(random_state=24)
X_train_resampled_spain, y_train_resampled_spain = smote.fit_resample(X_train , y_train)

model_spain = CatBoostClassifier(silent=True, iterations= 200, bagging_temperature= 10, eval_metric='Precision', learning_rate=1, auto_class_weights='Balanced')

# Ejecutar el grid search en los datos de entrenamiento
model_spain.fit(X_train_resampled_spain, y_train_resampled_spain)

print("Reporte de clasificación para Spain:")
print(classification_report(y_test, model_spain.predict(X_test)))

# # Guardar el modelo y el umbral óptimo en un archivo
with open(r'scr\modelos\mejores_modelos\spain.pkl', 'wb') as f:
    pickle.dump(model_spain, f)  
    

# ------------ Germany ---------------------------------------------------------------------------------------------

df_germany['Balance_Tenure_Ratio'] = df_germany['Balance'] / (df_germany['Tenure'] + 1e-6)

X = df_germany.drop(columns='Exited')
y = df_germany['Exited']

numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',  'Balance_Tenure_Ratio']
variables_sin_modificacion =  df_germany[[ 'IsActiveMember', 'Gender']]
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

model_german = CatBoostClassifier(silent=True, iterations= 300, bagging_temperature= 5, eval_metric='Precision', learning_rate=0.005,  auto_class_weights='Balanced', reg_lambda=0.001)

model_german.fit(X_train_resampled_germany, y_train_resampled_germany, cat_features = categ)

print("Reporte de clasificación para Germany:")
print(classification_report(y_test, model_german.predict(X_test)))

# # Guardar el modelo y el umbral óptimo en un archivo
with open(r'scr\modelos\mejores_modelos\germany.pkl', 'wb') as f:
    pickle.dump(model_german, f)  