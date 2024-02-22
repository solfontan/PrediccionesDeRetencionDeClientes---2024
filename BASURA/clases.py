#Manipulación de datos
import pandas as pd
import numpy as np
# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from collections import Counter


class CategoricalReport:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def generate_report(self, column_name, color='skyblue'):
        if column_name not in self.dataframe.columns:
            print(f"La columna {column_name} no está en el DataFrame.")
            return

        # Seleccionar la columna específica
        column_data = self.dataframe[column_name]

        # Obtener los valores únicos ordenados por frecuencia
        unique_values_counts = column_data.value_counts().sort_values(ascending=False)

        # Mostrar solo los 10 valores únicos más comunes
        top_10_unique_values = unique_values_counts.head(10).index.tolist()

        print("\n", "***********" * 10, "\n")
        print(f'Variable : {column_name.upper()}\n')

        # Mostrar valores únicos usando PrettyTable
        unique_table = PrettyTable(["Valores Únicos", "Frecuencia de Mayor Valor"])
        for value in top_10_unique_values:
            frequency = unique_values_counts[value]
            unique_table.add_row([value, frequency])
        print(f"{unique_table}")

        # Verificar si hay valores nulos y mostrar la proporción en porcentaje
        null_count = column_data.isnull().sum()
        null_percentage = (null_count / len(column_data)) * 100
        print(f"\nNulos: {null_count} ({null_percentage:.2f}%) sobre el total: {column_data.shape[0]} rows\n")

        # Mostrar medidas centrales
        mode_value = column_data.mode().to_numpy()[0]
        mode_count = column_data.mode().count()

        # Mostrar la moda usando PrettyTable
        mode_table = PrettyTable(["Moda", "Frecuencia de la moda"])
        mode_table.add_row([mode_value, mode_count])
        print(f"{mode_table}")

        # Crear un gráfico de barras con estilo profesional para los 10 valores más comunes
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        ax.set_facecolor('#F0F0F0')  # Fondo gris claro
        column_data[column_data.isin(top_10_unique_values)].value_counts().plot(kind='bar', color=color, ax=ax)
        plt.title(f'Distribución de {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=0, ha='center')
        plt.show()

class NumericReport:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def generate_report(self, column_name, visualization_type, color='skyblue'):
        if column_name not in self.dataframe.columns:
            print(f"La columna {column_name} no está en el DataFrame.")
            return

        # Seleccionar la columna específica
        column_data = self.dataframe[column_name]

        print("\n","***********" * 10, "\n")
        print(f"Variable : {column_name.upper()}")
        
        # Validar si la columna es numérica
        if pd.api.types.is_numeric_dtype(column_data):
            # Mostrar valores únicos usando PrettyTable
            # Mostrar valores únicos usando PrettyTable
            unique_table = PrettyTable(["Valores Únicos", "Frecuencia"])
            unique_values_count = Counter(column_data)

            # Mostrar solo los 10 valores únicos más comunes
            for value, frequency in unique_values_count.most_common(10):
                unique_table.add_row([value, frequency])

            # Verificar si hay valores nulos y mostrar la proporción en porcentaje
            null_count = column_data.isnull().sum()
            null_percentage = (null_count / len(column_data)) * 100

            # Mostrar medidas centrales y de dispersión
            mean_value = column_data.mean()
            median_value = column_data.median()
            std_dev_value = column_data.std()

            # Mostrar curtosis y asimetría usando PrettyTable
            kurtosis_value = column_data.kurtosis()
            skewness_value = column_data.skew()

            # Crear PrettyTable para medidas centrales y forma de la distribución
            central_table = PrettyTable(["Medidas Centrales", "Valor"])
            central_table.add_row(["Media", f"{mean_value:.2f}"])
            central_table.add_row(["Mediana", f"{median_value:.2f}"])
            central_table.add_row(["Desviación Estándar", f"{std_dev_value:.2f}"])

            distribution_table = PrettyTable(["Forma de la Distribución", "Valor"])
            distribution_table.add_row(["Curtosis", f"{kurtosis_value:.2f}"])
            distribution_table.add_row(["Asimetría", f"{skewness_value:.2f}"])

            # Imprimir las tablas
            print(f"\n{unique_table}")
            print(f"\nNulos: {null_count} ({null_percentage:.2f}%) sobre el total: {column_data.shape[0]} rows")
            print(f"\n{central_table}")
            print(f"\n{distribution_table}")
            print("\n","***********" * 10, "\n")


            # Visualización según la elección del usuario
            plt.figure(figsize=(10, 6))
            if visualization_type.lower() == 'boxplot':
                sns.boxplot(x=column_data, color=color)
                plt.title(f'Boxplot de {column_name}')
            elif visualization_type.lower() == 'histograma':
                column_data.plot(kind='hist', color=color, edgecolor='black')
                plt.title(f'Histograma de {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Frecuencia')
            elif visualization_type.lower() == 'densidad':
                sns.kdeplot(column_data, fill=True, color=color)
                plt.title(f'Gráfico de Densidad de {column_name}')
                plt.xlabel(column_name)
                plt.ylabel('Densidad')
            else:
                print("Tipo de visualización no reconocido. Por favor, elija entre 'boxplot', 'histograma' o 'densidad'.")

            plt.show()
        else:
            print(f"La columna {column_name} no es numérica.")
