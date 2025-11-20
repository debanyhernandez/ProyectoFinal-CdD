#----------------------------------------------------------
#Proyecto final de Ciencia de Datos 
#Segmentación de clientes por Perfil de Endeudamiento 
#Debany Jazmín Hernández Camacho 
#debany.hernandez@cimat.mx
#----------------------------------------------------------

#Exploración inicial:
#   - Carga y vista general del DataFrame
#   - Tipos de datos, NA, duplicados, resumen estadístico
#   - Exploración gráfica básica de algunas variables clave
#   - Exploración inicial e imputación de datos faltantes
#


#---------------------------------------------------------
#Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
#---------------------------------------------------------

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option("display.width", None)
pd.set_option('display.max_colwidth', None)



#---------------------------------------------------------
#Carga del DataFrame
file_path = "./data/cs-training.csv"
df = pd.read_csv(file_path, index_col=0)
print("\n --- Vista general del DataFrame ---")
print(df.head())
print("\n --- Dimensión del DataFrame (filas, columnas) ---", df.shape)
#---------------------------------------------------------

#---------------------------------------------------------
#Eliminamos el índice artificial si es necesario
if 'Unnamed: 0' in df.columns:
    df = df.rename(columns={'Unnamed: 0': 'ID'})
    df = df.drop(columns=['ID'])

print("\n --- Columnas después de eliminar índice artificial ---")
print(df.columns)
#---------------------------------------------------------

#---------------------------------------------------------
#Exploración inicial de la base de datos
print("\n --- Tipos de datos ---")
print(df.dtypes)

#Verificamos si existen valores faltantes en las columnas
print("\n --- Valores faltantes por columna ---")
na_counts = df.isnull().sum()
print(na_counts)

#Porcentaje de valores faltantes por columna
print("\n --- Porcentaje de valores faltantes por columna ---")
na_percentage = (df.isnull().sum() / len(df)) * 100
print(na_percentage)

#Verificamos si existen filas duplicadas
print("\n ---Número de filas duplicadas:", df.duplicated().sum())

#Resumen estadístico de las variables numéricas
print("\n --- Resumen estadístico de las variables numéricas ---")
print(df.describe())

#Dimensión final de la base de datos
print("\n --- Dimensión final de la base de datos (filas, columnas) ---", df.shape)

#---------------------------------------------------------

#---------------------------------------------------------
#Exploración gráfica inicial 

sns.set(style="whitegrid")

#Seleccionamos variables de interés para la visualización
variables_interes = ['RevolvingUtilizationOfUnsecuredLines', 
                     'DebtRatio',
                     'MonthlyIncome',
                     'NumberOfOpenCreditLinesAndLoans']

print("\n --- Exploración gráfica inicial de variables clave ---")
print("\n Variables de interés:", variables_interes)

#Gráfica compuesta: Histograma, Densidad y Boxplot
fig, axes = plt.subplots(3, len(variables_interes), figsize=(4 * len(variables_interes), 12))
if len(variables_interes) == 1:
    axes = axes.reshape(3,1)  # Asegura que axes sea 2D incluso si hay una sola variable
for i, var in enumerate(variables_interes):
    col_data = df[var].dropna()

    # Histograma
    sns.histplot(col_data, bins=40, kde=False, ax=axes[0, i], color='skyblue')
    axes[0, i].set_title(f'Histograma de {var}')
    axes[0, i].set_xlabel("")
    
    # Densidad
    sns.kdeplot(col_data, ax=axes[1, i], fill=True, color='purple')
    axes[1, i].set_title(f'Densidad de {var}')
    axes[1, i].set_xlabel("")
    
    # Boxplot
    sns.boxplot(x=col_data, ax=axes[2, i], color='lightgreen')
    axes[2, i].set_title(f'Boxplot de {var}')
    axes[2, i].set_xlabel(var)
#Guardamos el gráfico
plt.savefig('./Figures/variables_clave_exploracion.png', dpi=300)
plt.tight_layout()
plt.show()
plt.close()

#---------------------------------------------------------

#---------------------------------------------------------
#Matriz de correlación
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', center =0)
plt.title('Matriz de Correlación')
plt.savefig('./Figures/matriz_correlacion.png', dpi=300)
plt.tight_layout()
plt.show()
plt.close()
#---------------------------------------------------------
