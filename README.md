# Segmentación de Clientes por Perfil de Endeudamiento
### Proyecto final – Introducción a la Ciencia de Datos  
**Debany Jazmín Hernández Camacho** · Maestría en Matemáticas Aplicadas · CIMAT  

---

##  Descripción general

Este repositorio contiene el proyecto final del curso **Introducción a la Ciencia de Datos**, cuyo objetivo es construir una **segmentación no supervisada** de clientes a partir de sus características financieras.

Se utiliza la base de datos **Give Me Some Credit** (Kaggle), que incluye información de ~150,000 clientes con variables de:

- ingreso,
- endeudamiento,
- utilización del crédito,
- comportamiento de pago (morosidad),
- y características demográficas simples.

El enfoque combina **preprocesamiento cuidadoso**, **PCA** y **K-Means** para identificar perfiles de endeudamiento interpretables.

---

## 📂 Estructura del repositorio

```text
ProyectoFinal-CdD/
│
├── Articulo/
│   ├── reporte.tex
│   └── reporte.pdf
│
├── Presentacion/
│   ├── presentacion.tex
│   └── presentacion.pdf
│
├── Codigos/
│   ├── exploracion_inicial.py
│   ├── modelado.py
│   ├── Kmeans.py
│   └── perfilamiento.py
│
├── data/
│   ├── cs_training.csv
│   ├── df_limpio_para_clustering.csv
│   ├── df_pca_para_clustering.csv
│   └── df_resultado_clusters.csv
│
├── Figures/
│   ├── matriz_correlacion.png
│   ├── varianza_explicada_pca.png
│   ├── metodo_del_codo_kmeans.png
│   ├── silhouette_vs_k_kmeans.png
│   ├── clusters_kmeans_pca.png
│   ├── heatmap_clusters.png
│   ├── segmentacion_radar_clusters.png
│   └── variables_clave_exploracion.png
│
└── README.md
