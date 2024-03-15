### Manipulación de Datos
import pandas as pd
import numpy as np

### Visualización de Datos
import seaborn as sns
import matplotlib.pyplot as plt

### Machine Learning
# Preparación de datos 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

# Modelos

from sklearn.ensemble import  RandomForestClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import  classification_report

from sklearn.model_selection import GridSearchCV
import pickle


# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans

df = pd.read_csv('scr\data\processed\Churn_Modelling.csv')

df['is_male'] = df['Gender'].replace({'Female' : 0, 'Male': 1})

numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
categorical_features = ['Geography']

# Definir transformadores para características numéricas y categóricas
numeric_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder()

# Crear un ColumnTransformer para aplicar transformaciones a diferentes columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Crear el pipeline con el preprocesador
pipeline = make_pipeline(preprocessor)
# Aplicar el preprocesador a los datos
X_processed = pipeline.fit_transform(df)

# Obtener los nombres de las columnas después de aplicar OneHotEncoder
encoded_categorical_columns = preprocessor.named_transformers_['cat']\
    .get_feature_names_out(input_features=categorical_features)

# Combinar los nombres de las columnas numéricas y categóricas
processed_columns = numeric_features + list(encoded_categorical_columns)

# Crear DataFrame con los datos procesados y los nombres de las columnas
processed_df = pd.DataFrame(X_processed, columns=processed_columns)
concatenated_series  = pd.concat((processed_df, df[['HasCrCard', 'IsActiveMember', 'Exited', 'is_male']].reset_index(drop=True)),  axis=1)

# Inicializar y ajustar el modelo K-Means
n_clusters = 3  # Lo voy a dividir en tres
kmeans = KMeans(n_clusters=n_clusters, random_state=24)
kmeans.fit(concatenated_series)

# Obtener las etiquetas de cluster asignadas a cada punto de datos en el conjunto de entrenamiento
train_labels = kmeans.labels_

# # # Inicializar un diccionario para almacenar los DataFrames de cada cluster
dict_cluster_dfs = {}

# Iterar sobre cada cluster
for cluster_num in range(n_clusters):
    # Obtener los índices de las filas asignadas al cluster actual
    indices_current_cluster = [i for i, label in enumerate(train_labels) if label == cluster_num]
    # Crear un nuevo DataFrame con las filas del cluster actual
    df_current_cluster = concatenated_series.iloc[indices_current_cluster]
    # Almacenar el DataFrame del cluster actual en el diccionario
    dict_cluster_dfs[cluster_num] = df_current_cluster
    
    
X = dict_cluster_dfs[0].drop(columns='Exited')
y = dict_cluster_dfs[0]['Exited']

X_train0, X_test0, y_train0, y_test0 = train_test_split(X, y, test_size=0.2, random_state=24)

# Aplicar SMOTE solo al conjunto de entrenamiento
smote = ADASYN(random_state=24)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train0 , y_train0)

clust0_xgb = RandomForestClassifier(n_estimators=500, max_depth=50, min_samples_split=2, min_samples_leaf=1, bootstrap=True, class_weight='balanced', random_state=24)
clust0_xgb.fit(X_train_resampled, y_train_resampled)

print(classification_report(y_test0, clust0_xgb.predict(X_test0) ))


X = dict_cluster_dfs[1].drop(columns='Exited')
y = dict_cluster_dfs[1]['Exited']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=24)

# Aplicar SMOTE solo al conjunto de entrenamiento
smote = BorderlineSMOTE(random_state=24)
X_train_resampled_clust1, y_train_resampled_clust1 = smote.fit_resample(X_train1 , y_train1)

clust1_rnd = CatBoostClassifier(iterations= 500, silent=True, bagging_temperature = 10, learning_rate=0.01, eval_metric='Precision', auto_class_weights='Balanced')
clust1_rnd.fit(X_train_resampled_clust1, y_train_resampled_clust1)

print(classification_report(y_test1, clust1_rnd.predict(X_test1) ))

X = dict_cluster_dfs[2].drop(columns='Exited')
y = dict_cluster_dfs[2]['Exited']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=24)

# Aplicar SMOTE solo al conjunto de entrenamiento
smote = ADASYN(random_state=24)
X_train_resampled_clust2, y_train_resampled_clust2 = smote.fit_resample(X_train2 , y_train2)

clus2_cat = CatBoostClassifier(bagging_temperature = 20, iterations = 400 , silent=True, learning_rate=0.01, eval_metric='Precision',  auto_class_weights='Balanced')
clus2_cat.fit(X_train_resampled_clust2, y_train_resampled_clust2)
print(classification_report(y_test2, clus2_cat.predict(X_test2) ))