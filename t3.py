import pandas as pd
import plotly.express as px

# Đọc dữ liệu PCA từ file CSV
pca_data_path = "visualization_data/pca_clusters.csv"
df_pca = pd.read_csv(pca_data_path)

# Vẽ biểu đồ 3D với Plotly
fig = px.scatter_3d(df_pca, x="PC1", y="PC2", z="PC3", 
                     color=df_pca["Cluster"].astype(str),  
                     title="3D PCA Clustering",
                     opacity=0.8)

fig.show()
