import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

preprocessed_dff = pd.read_csv("preprocessed_cardekho.csv")
preprocessed_dff = preprocessed_dff.sample(n=500, random_state=42)

# Giả sử preprocessed_df đã được load vào môi trường
# Đổi tên cột "mileage(km/ltr/kg)" thành "mileage" để thuận tiện
preprocessed_dff.rename(columns={"mileage(km/ltr/kg)": "mileage"}, inplace=True)

# Chọn các thuộc tính số để tính độ tương đồng: selling_price, km_driven, mileage, engine, max_power
features = preprocessed_dff[["selling_price", "km_driven", "mileage", "max_power"]]

# Chuẩn hóa dữ liệu để các thuộc tính có cùng quy mô
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Tính ma trận cosine similarity giữa các mẫu xe
similarity_matrix = cosine_similarity(features_scaled)

# Xây dựng đồ thị: mỗi xe là một nút với nhãn là 'name'
G = nx.Graph()
for idx, row in preprocessed_dff.iterrows():
    G.add_node(idx, label=row["name"])

# Thêm cạnh nếu độ tương đồng vượt qua ngưỡng nhất định (ví dụ: threshold = 0.8)
threshold = 0.8
n = len(preprocessed_dff)
for i in range(n):
    for j in range(i + 1, n):
        sim = similarity_matrix[i, j]
        if sim > threshold:
            G.add_edge(i, j, weight=sim)

# Tính vị trí nút bằng thuật toán Fruchterman-Reingold (spring_layout)
pos = nx.spring_layout(G)

# Vẽ đồ thị
plt.figure(figsize=(10, 10))
nx.draw(G, pos,
        with_labels=True,
        labels=nx.get_node_attributes(G, 'label'),
        node_color='skyblue',
        edge_color='gray',
        node_size=100,
        font_size=8)
plt.title("FR Layout Graph từ Preprocessed Cardekho Dataset")
plt.show()


