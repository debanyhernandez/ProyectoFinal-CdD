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
plt.style.use('seaborn-v0_8-whitegrid')
paleta_color_dict= {
    '0': '#FF69B4',
    '1': '#87CEEB',
    '2': '#98FB98',
    '3': "#DDA7F3",
    '4': '#FFDAB9'
}
paleta_color_list = list(paleta_color_dict.values())
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
    sns.boxplot(x= 'Cluster', y =col, data=df_resultado, ax=axes[i], palette=paleta_color_dict)
    axes[i].set_title(f'Distribución de {col} por Cluster', fontsize=14)
    axes[i].grid(axis='y', alpha=0.5)

plt.tight_layout()
plt.savefig('./Figures/perfil_boxplots_clusters.png', dpi=300)
plt.show()
plt.close()

#Gráfica de Radar
df_perfil_grafica = perfil_final.drop(columns=['Conteo', 'Porcentaje'])

#Eliminamos el Cluster 1 (Outlier) ya que distorsiona la escala completamente
df_perfil_grafica = df_perfil_grafica.drop(index=1, errors='ignore')

#Estandarizamos los valores usando el escalado Min-Max
#Esto nos ayuda a que todas las variables estén en una escala de 0 a 1 para poder compararlas
df_norm = df_perfil_grafica.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

def create_radar(df, title, palette):
    '''
    Función para crear el gráfico de radar
    '''
    categories = list(df.columns)
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(polar=True))


    for cluster_id in df.index:
        values = df.loc[cluster_id].values.flatten().tolist()
        values += values[:1]
        color = paleta_color_dict[str(cluster_id)]
        ax.plot(angles, values, linewidth =2, linestyle='solid', label=f'Cluster {cluster_id}', color = color)
        ax.fill(angles, values, color=color, alpha=0.25)

    #Configuración de los ejes y etiquetas
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    
    # Eje y (escala de 0 a 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
    ax.set_ylim(0, 1)
    
    plt.title(title, size=16, y=1.1)
    plt.legend(loc='lower left', bbox_to_anchor=(1.05, 0.05), fontsize=10)
    plt.tight_layout()
    plt.savefig('./Figures/segmentacion_radar_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

create_radar(df_norm, f'Perfiles de Clientes - {len(df_norm.index)} Clusters (Estandarizado)', paleta_color_dict)


#Heatmap de medias normalizadas 
variables = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfDependents",
    "age"
]
cluster_means = df_resultado.groupby("Cluster")[variables].mean()
cluster_z = (cluster_means - cluster_means.mean()) / cluster_means.std()

plt.figure(figsize=(10, 6))
sns.heatmap(
    cluster_z.T,               
    cmap="BuPu",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Heatmap – Medias normalizadas por variable y cluster", fontsize=16)
plt.xlabel("Cluster")
plt.ylabel("Variable")
plt.yticks(rotation=0)        # nombres de variables horizontales
plt.tight_layout()
plt.savefig("./Figures/heatmap_clusters.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()