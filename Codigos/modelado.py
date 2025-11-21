#----------------------------------------------------------
#Proyecto final de Ciencia de Datos 
#Segmentación de clientes por Perfil de Endeudamiento 
#Debany Jazmín Hernández Camacho 
#debany.hernandez@cimat.mx
#----------------------------------------------------------

#---------------------------------------------------------
#Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Confuguración de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set(style="whitegrid")
#---------------------------------------------------------

#---------------------------------------------------------
#Carga del DataFrame
df_clustering = pd.read_csv("./data/df_limpio_para_clustering.csv")
print("\n --- Dimensión del DataFrame para modelado (filas, columnas) ---", df_clustering.shape)
#---------------------------------------------------------


#---------------------------------------------------------
#--- Escalado de datos (StandardScaler) ---
#PCA y KMeans son sensibles a la escala de las variables, por lo que es importante escalar los datos antes de aplicar estos métodos.
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)
df_scaled = pd.DataFrame(df_scaled, columns=df_clustering.columns)
print("\n --- Datos escalados (primeras filas) ---")
print(df_scaled.head())

#--- Reducción de dimensionalidad con PCA ---
#Aplicamos PCA para reducir la dimensionalidad de los datos y facilitar la visualización y el clustering.
pca = PCA() 
pca.fit(df_scaled)

#Calculamos la varianza explicada acumulada
radio_var_explicada = np.cumsum(pca.explained_variance_ratio_)

#Gráfica de la varianza explicada acumulada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(radio_var_explicada) + 1), radio_var_explicada, marker='o', linestyle='--')
plt.title('Varianza Explicada Acumulada por Componentes Principales (PCA)')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulada')
plt.grid(True)
plt.axhline(y=0.85, color='r', linestyle='-', label='85% de Varianza Explicada')
plt.legend()
plt.savefig('./Figures/varianza_explicada_pca.png', dpi=300)
plt.show()
plt.close()

#Decisión del número de componentes principales a retener
#Retenemos los componentes que explican al menos el 85% de la varianza
num_componentes = np.where(radio_var_explicada >= 0.85)[0][0] + 1
print(f"\n Número de componentes principales retenidos para explicar al menos el 85% de la varianza: {num_componentes}")