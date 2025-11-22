#----------------------------------------------------------
#Proyecto final de Ciencia de Datos 
#Segmentación de clientes por Perfil de Endeudamiento 
#Debany Jazmín Hernández Camacho 
#debany.hernandez@cimat.mx
#----------------------------------------------------------

#----------------------------------------------------------
#Librerías 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
import warnings
warnings.filterwarnings('ignore')

#Configuración de pandas para visualización
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
#----------------------------------------------------------

#----------------------------------------------------------
#Carga de datos con el Cluster Asignado 
df_resultado = pd.read_csv("./data/df_resultado_clusters.csv")

k_optimo = df_resultado['Cluster'].nunique()

#-----------------------------------------------------------


#-----------------------------------------------------------
#--- Perfilamiento por medias y tamaño---
perfil_clusters = df_resultado.drop(columns=['Target'], errors='ignore').groupby('Cluster').mean()
#Calculamos el tamaño de cada cluster (conteo y porcentaje)
tamaño_clusters = df_resultado['Cluster'].value_counts().rename('Conteo').to_frame()
tamaño_clusters['Porcentaje'] = df_resultado['Cluster'].value_counts(normalize=True).mul(100)
perfil_final = perfil_clusters.join(tamaño_clusters).sort_values(by='Porcentaje', ascending=False)

print("\n --- Tabla de perfilamiento final (Medias por cluster)---")
print(perfil_final.T.to_markdown(floatfmt=".3f"))

variables_clave = [
    'RevolvingUtilizationOfUnsecuredLines',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'MonthlyIncome',
    'DebtRatio'
]

#Creamos boxplots para comparar la distribución de las variables clave por cluster
fig, axes = plt.subplots(2, 2, figsize=(14,10))
axes = axes.flatten()
for i, col in enumerate(variables_clave):
    sns.boxplot(x= 'Cluster', y =col, data=df_resultado, ax=axes[i], palette='tab10')
    axes[i].set_title(f'Distribución de {col} por CLuster')
    axes[i].grid(axis='y', alpha=0.5)

plt.tight_layout()
plt.savefig('./Figures/perfil_boxplots_clusters.png', dpi=300)
plt.show()
plt.close()

