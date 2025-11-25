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
#Diseño de las gráficas
sns.set(style="whitegrid")

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


#Seleccionamos variables de interés para la visualización
variables_interes = ['RevolvingUtilizationOfUnsecuredLines', 
                     'DebtRatio',
                     'MonthlyIncome',
                     'NumberOfOpenCreditLinesAndLoans']

print("\n --- Exploración gráfica inicial de variables clave ---")
print("\n Variables de interés:", variables_interes)

#Gráfica compuesta: Histograma, Densidad y Boxplot
fig, axes = plt.subplots(3, len(variables_interes),figsize=(5 * len(variables_interes), 10))
if len(variables_interes) == 1:
    axes = axes.reshape(3,1)  # Asegura que axes sea 2D incluso si hay una sola variable
for i, var in enumerate(variables_interes):
    col_data = df[var].dropna()

    # Histograma 
    sns.histplot(col_data, bins=40, kde=False, ax=axes[0, i],
                 color="#9BD4FF")  # azul pastel
    axes[0, i].set_title(f'Histograma de {var}')
    axes[0, i].set_xlabel("")
    
    # Densidad 
    sns.kdeplot(col_data, ax=axes[1, i],
                fill=True, color="#BB88BB")  # lila pastel
    axes[1, i].set_title(f'Densidad de {var}')
    axes[1, i].set_xlabel("")
    
    # Boxplot 
    sns.boxplot(x=col_data, ax=axes[2, i],
                color="#A8D1AE")  # verde menta pastel
    axes[2, i].set_title(f'Boxplot de {var}')
    axes[2, i].set_xlabel(var)


#Titulo general
plt.suptitle('Exploración Gráfica Inicial de Variables Clave', fontsize=16, fontweight='bold', y=1.02)
#Guardamos el gráfico
plt.savefig('./Figures/variables_clave_exploracion.png', dpi=300)
plt.tight_layout()
plt.close()

#---------------------------------------------------------

#---------------------------------------------------------
#Matriz de correlación
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación', fontsize=16, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.savefig('./Figures/matriz_correlacion.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.close()
#---------------------------------------------------------

#---------------------------------------------------------
#Limpieza y tratamiento de datos problemáticos

#Eliminamos filas duplicadas 
num_duplicados = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"\n --- Filas duplicadas eliminadas: {num_duplicados} ---")
#Dimensión después de eliminar duplicados
print("\n --- Dimensión después de eliminar duplicados (filas, columnas) ---", df.shape)

#Revisamos datos anómalos y outliers en variables clave
#---------------------Age------------------------- 
#Justificación: Edad mínima legal para tener crédito es 18 años
#Los valores negativos o menores a 18 son erróneos
filas_iniciales = len(df)
df = df[df['age'] > 0].reset_index(drop=True)
filas_removidas = filas_iniciales - len(df)
print(f"\n --- Tratamiento de datos de Age = 0 ---")
print(f"Filas eliminadas (Age=0): {filas_removidas}")
print(f"Dimensión actual: {df.shape}")

#Imputación de valores faltantes (NaN)
#----------------------MonthlyIncome-------------------------
#Imputación de MonthlyIncome (~19.8% faltante)
#Utilizamos la mediana para imputar los valores faltantes
#La mediana es robusta frente a outliers y al fuerte sesgo de la distribución de ingresos
median_income = df['MonthlyIncome'].median()
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(median_income)
print(f"\n Imputación de valores faltantes en MonthlyIncome: {median_income:.2f}")

#----------------------NumberOfDependents-------------------------
#Imputación de NumberOfDependents (~2.6% faltante)
#Es una variable de conteo (discreta), por lo que la moda es la medida más adecuada
mode_dependents = df['NumberOfDependents'].mode()[0]
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(mode_dependents)
print(f"\n Imputación de valores faltantes en NumberOfDependents: {mode_dependents:.0f}")
#Verificamos que no queden nulos
print("\n Verificación de valores nulos después de la imputación:")
print(f"Total de valores nulos en el DataFrame: {df.isnull().sum().sum()}")

#Tratamiento de outliers por truncamiento 
#----------------------RevolvingUtilizationOfUnsecuredLines-------------------------
#Truncamiento de RevolvingUtilizationOfUnsecuredLines (Rango esperado de 0 a 50,000)
#Valores por encima de 1.0 se consideran atípicos en este contexto
#Truncamos al percentil 99 para limitar el impacto extremo en PCA/k-means 
p99_util = df['RevolvingUtilizationOfUnsecuredLines'].quantile(0.99)
df['RevolvingUtilizationOfUnsecuredLines'] = np.where(
    df['RevolvingUtilizationOfUnsecuredLines'] > p99_util,
    p99_util,
    df['RevolvingUtilizationOfUnsecuredLines']
)
print(f"\n Truncamiento de RevolvingUtilizationOfUnsecuredLines al percentil 99:")
print(f"Valores > P99 truncados a: {p99_util:.4f}")

#----------------------DebtRatio-------------------------
#Truncamiento de DebtRatio (Rango esperado de 0 a 329,664+)
#Valores extremadamente altos distorsionan el clustering
#Truncamos al percentil 99.5 para limitar el impacto extremo en PCA/k-means
p995_debt = df['DebtRatio'].quantile(0.995)
df['DebtRatio'] = np.where(
    df['DebtRatio'] > p995_debt,
    p995_debt,
    df['DebtRatio']
)
print(f"\n Truncamiento de DebtRatio al percentil 99.5:")
print(f"Valores > P99.5 truncados a: {p995_debt:.4f}")

#Excluimos la variable objetivo 'SeriousDlqin2yrs' del DataFrame
df_clustering = df.drop(columns=['SeriousDlqin2yrs'])

print("\n --- Dimensión final del DataFrame para clustering (filas, columnas) ---", df_clustering.shape)
print("Todas las variables están limpias y listas")
#-------------------------------------------------------------

#Guardamos el DataFrame limpio para análisis posteriores
df_clustering.to_csv('./data/df_limpio_para_clustering.csv', index=False)
print("\n DataFrame limpio guardado en './data/df_limpio_para_clustering.csv'")

