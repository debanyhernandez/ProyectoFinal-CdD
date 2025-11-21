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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

#Configuración de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set(style="whitegrid")
#----------------------------------------------------------

#----------------------------------------------------------
#Carga de datos transformados 
df_pca = pd.read_csv("./data/df_pca_para_clustering.csv")
df_clustering = pd.read_csv("./data/df_limpio_para_clustering.csv")
print("\n Datos PCA y original cargados exitosamente.")

#----------------------------------------------------------

#---------------------------------------------------------
#--- Clustering con KMeans ---
#Determinamos el número óptimo de clusters usando el método del codo y el coeficiente de silhouette
inercia = []
k_range = range(2, 11)
silhouette_values = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(df_pca)
    inercia.append(kmeans.inertia_)
    