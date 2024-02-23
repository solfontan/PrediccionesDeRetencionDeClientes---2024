### Manipulación de Datos
import pandas as pd
import numpy as np

### Visualización de Datos
import seaborn as sns
import matplotlib.pyplot as plt

### Tratamiento de datos
from utils.funciones import extended_describe

### Machine Learning
# Preparación de datos 
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import  RandomForestClassifier,  GradientBoostingClassifier, VotingClassifier,  AdaBoostClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  classification_report

from sklearn.model_selection import GridSearchCV
import pickle
from utils.funciones import BaseLine

# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('./data/processed/Churn_Modelling.csv')

from sklearn.cluster import KMeans

df= pd.read_csv('./data/processed/Churn_Modelling.csv')

df['CreditCardOwnerTenure'] = df.HasCrCard * df.Age
df['Saldo_Salario_Ratio'] = df['Balance'] / df['EstimatedSalary']

numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary','CreditCardOwnerTenure', 'Saldo_Salario_Ratio']
variables_sin_modificacion = ['HasCrCard', 'IsActiveMember', 'Exited']
categorical_features = ['Geography', 'Gender']

# Definir transformadores para características numéricas y categóricas
numeric_transformer = StandardScaler()
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
concatenated_series  = pd.concat((processed_df, df[[ 'HasCrCard', 'IsActiveMember', 'Exited']]), axis=1)

# Inicializar y ajustar el modelo K-Means
n_clusters = 3  # Lo voy a dividir en tres
kmeans = KMeans(n_clusters=n_clusters, random_state=24)
kmeans.fit(concatenated_series)

# Obtener las etiquetas de cluster asignadas a cada punto de datos en el conjunto de entrenamiento
train_labels = kmeans.labels_
# Inicializar un diccionario para almacenar los DataFrames de cada cluster
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Aplicar SMOTE solo al conjunto de entrenamiento
smote = SMOTE(random_state=24)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train , y_train)

clust0_xgb = xgb.XGBClassifier(colsample_bytree= 0.9, learning_rate= 0.05, max_depth= 8, n_estimators= 200, subsample= 0.7)
clust0_xgb.fit(X_train_resampled, y_train_resampled)
print(classification_report(y_test, clust0_xgb.predict(X_test) ))

X = dict_cluster_dfs[1].drop(columns='Exited')
y = dict_cluster_dfs[1]['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Aplicar SMOTE solo al conjunto de entrenamiento
smote = SMOTE(random_state=24)
X_train_resampled_clust1, y_train_resampled_clust1 = smote.fit_resample(X_train , y_train)

clust1_rnd = RandomForestClassifier(bootstrap = False, max_depth = 12, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 4, n_estimators = 300)
clust1_rnd.fit(X_train_resampled_clust1, y_train_resampled_clust1)
print(classification_report(y_test, clust1_rnd.predict(X_test) ))

X = dict_cluster_dfs[2].drop(columns='Exited')
y = dict_cluster_dfs[2]['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Aplicar SMOTE solo al conjunto de entrenamiento
smote = SMOTE(random_state=24)
X_train_resampled_clust2, y_train_resampled_clust2 = smote.fit_resample(X_train , y_train)

clus2_cat = CatBoostClassifier(bagging_temperature = 20, depth = 7, iterations = 200, l2_leaf_reg =  7, learning_rate =  0.05, silent=True)
clus2_cat.fit(X_train_resampled_clust2, y_train_resampled_clust2)
print(classification_report(y_test, clus2_cat.predict(X_test) ))