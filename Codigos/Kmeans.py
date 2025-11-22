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
sns.set(style="whitegrid", context="talk", font_scale=0.9)
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
    score = silhouette_score(df_pca, kmeans.labels_)
    silhouette_values.append(score)

#Gráfica del método del codo
plt.figure(figsize=(12, 5))
plt.rcParams["font.family"]="serif"
plt.plot(k_range, inercia, marker='o', linestyle='--', markerfacecolor="#E03B69" , markeredgecolor="#F4ACB7", color="#F4ACB7")
plt.title('Método del Codo para KMeans', fontsize=16)
plt.xlabel('Número de clusters (k)', fontsize=12)
plt.ylabel('Inercia', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(k_range)
plt.savefig('./Figures/metodo_del_codo_kmeans.png', dpi=300)
plt.tight_layout()
plt.show()
plt.close()


#Gráfica del coeficiente de silhouette vs k 
plt.figure(figsize=(12, 5))
plt.plot(k_range, silhouette_values, marker='o', linestyle='--', markerfacecolor="#D1A5F3E7"  , markeredgecolor= "#896BA3", color = "#D1A5F3E7" )
plt.title('Coeficiente de Silhouette vs Número de Clusters (k)', fontsize=16)
plt.xlabel('Número de clusters (k)', fontsize=12)
plt.ylabel('Coeficiente de Silhouette', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(k_range)
plt.savefig('./Figures/silhouette_vs_k_kmeans.png', dpi=300)
plt.tight_layout()  
plt.show()
plt.close()

print("\n --- Resultados del método del codo y coeficiente de silhouette score")
for k, s in zip(k_range, silhouette_values):
    print(f"Clusters: {k}, Silhouette Score: {s:.4f}")

#Basándonos en el método del codo y la justificación metodológica, seleccionamos k=5
#Aunque silhoutte dé k=2, justificamos k=5 por la interpretabilidad y el método del codo
k_optimo = 5
print(f"\n --- Se elige k={k_optimo} para el clustering KMeans ---")

#Ajuste final de Kmeans con k óptimo
kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init='auto')
clusters_final = kmeans_final.fit_predict(df_pca)

#Añadimos la asignación de clusters al DataFrame original limpio
df_resultado = df_clustering.copy()
df_resultado['Cluster'] = clusters_final

#Visualización de los clusters en los dos primeros componentes principales
var_pc1 = 0.3010
var_pc2 = 0.1622
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    df_pca['PC1'],
    df_pca['PC2'], 
    c=clusters_final, 
    cmap='Set1', 
    alpha=0.6
)
plt.xlabel(f'PC1 ({var_pc1:-2%} varianza)')
plt.ylabel(f'PC2 ({var_pc2: .2%} varianza)')
plt.title(f'Clusters identificados por KMeans (k={k_optimo}) en PCA')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('./Figures/clusters_kmeans_pca.png', dpi=300)
plt.tight_layout()
plt.show()
plt.close()

#Guardamos la base de datos con la asignación de clusters 
df_resultado.to_csv('./data/df_resultado_clusters.csv' , index=False)
