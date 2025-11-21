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
from sklearn.metrics import silhouette_score

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
pca.fit_transform(df_scaled)

#Calculamos la varianza explicada acumulada
radio_var_explicada = np.cumsum(pca.explained_variance_ratio_)
print("\n --- Varianza explicada acumulada por componentes principales ---")
for i, v in enumerate(radio_var_explicada, start=1):
    print(f"Componente {i}: {v:.4f}")

#Gráfica de la varianza explicada acumulada
plt.figure(figsize=(10, 6))
#Barras: varianza explicada por cada componente
plt.bar(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_,
    alpha=0.5,
    label='Varianza explicada por componente'
)
#Línea: varianza explicada acumulada
plt.plot(
    range(1, len(radio_var_explicada) + 1),
    radio_var_explicada,
    marker='o',
    color='purple',
    label='Varianza explicada acumulada'
)
plt.title('Varianza Explicada por Componentes Principales (PCA)')
plt.xlabel('Componente principal')
plt.ylabel('Proporción de varianza')
plt.axhline(y=0.85, color='green', linestyle='--', label='85% de varianza explicada')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('./Figures/varianza_explicada_pca.png', dpi=300)
plt.tight_layout()
plt.show()
plt.close()


#Decisión del número de componentes principales a retener
#Retenemos los componentes que explican al menos el 85% de la varianza
umbral_varianza = 0.85
num_componentes = np.where(radio_var_explicada >= umbral_varianza)[0][0] + 1
print(f"\n Número de componentes principales retenidos para explicar al menos el 85% de la varianza: {num_componentes}")

#Aplicamos PCA con el número seleccionado de componentes
pca_final = PCA(n_components=num_componentes)
df_pca = pca_final.fit_transform(df_scaled)
df_pca = pd.DataFrame(df_pca, columns=[f'PC{i}' for i in range(1,num_componentes+1)])
var_seleccionada = pca_final.explained_variance_ratio_.sum()
print(f"Datos reducidos a {num_componentes} componentes principales (PCA). Varianza explicada total: {var_seleccionada:.4f}")

#Contribución de las variables originales a los componentes principales
loadings = pca_final.components_.T * np.sqrt(pca_final.explained_variance_)
df_loadings = pd.DataFrame(loadings, index=df_clustering.columns, columns=[f'PC{i}' for i in range(1,num_componentes+1)])

print("\n --- Loadings de las primeras componentes principales ---")
print(df_loadings.iloc[:, min(4, num_componentes)])

#Gráficos de los loadings para PC1 y PC2
if num_componentes >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Loadings para PC1
    axes[0].barh(df_clustering.columns, df_loadings['PC1'], color='skyblue')
    axes[0].set_title('Loadings de las Variables para PC1')
    axes[0].set_xlabel('Contribución')

    # Loadings para PC2
    axes[1].barh(df_clustering.columns, df_loadings['PC2'], color='lightgreen')
    axes[1].set_title('Loadings de las Variables para PC2')
    axes[1].set_xlabel('Contribución')

    plt.tight_layout()
    plt.savefig('./Figures/loadings_pc1_pc2.png', dpi=300)
    plt.show()
    plt.close()

#--- Clustering con KMeans ---
#Determinamos el número k óptimo de clusters usando el método del codo
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
plt.figure(figsize=(10, 6))
plt.plot(k_range, inercia, marker='o', linestyle='--')
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters (KMeans)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.grid(True, alpha=0.3)
plt.xticks(k_range)
plt.tight_layout()
plt.savefig('./Figures/metodo_del_codo_kmeans.png', dpi=300)
plt.show()
plt.close()

#Gráfica del silhouette score vs k 
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_values, marker='o', linestyle='--', color='deeppink')
plt.title('Silhouette Score vs Número de Clusters (KMeans)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True, alpha=0.3)
plt.xticks(k_range)
plt.tight_layout()
plt.savefig('./Figures/silhouette_score_kmeans.png', dpi=300)
plt.show()
plt.close()

print("\n --- Resultados del método del codo y silhouette score ---")
for k, s in zip(k_range, silhouette_values):
    print(f"k={k}: Silhouette Score={s:.4f}")

#Elección automática del número óptimo de clusters según Silhouette máximo
optimal_k = k_range[int(np.argmax(silhouette_values))]
print(f"\n Número óptimo de clusters seleccionado automáticamente según Silhouette Score: k={optimal_k}")


#Ajuste final de KMeans con el número óptimo de clusters
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
clusters_final = kmeans_final.fit_predict(df_pca)

#Añadimos la asignación de clusters al DataFrame original
df_resultado = df_clustering.copy()
df_resultado['Cluster'] = clusters_final

#Visualización de los clusters en los primeros dos componentes principales
if num_componentes >= 2:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df_pca['PC1'], df_pca['PC2'],
        c=clusters_final,
        cmap='tab10',
        alpha=0.6
    )
    plt.xlabel(f'PC1({pca_final.explained_variance_ratio_[0]:.2%} varianza)')
    plt.ylabel(f'PC2({pca_final.explained_variance_ratio_[1]:.2%} varianza)')
    plt.title('Clusters Identificados por KMeans en el Espacio PCA(PC1 vs PC2)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig('./Figures/clusters_kmeans_pc1_pc2.png', dpi=300)
    plt.tight_layout()
    plt.show()
    plt.close()

