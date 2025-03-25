# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# 1. Load Data
df = pd.read_csv('cardekho.csv')
print("Dataset Shape:", df.shape)
print(df.head())

# 2. Exploratory Data Analysis (EDA)
# Kiểm tra missing values
print("Missing Values:\n", df.isnull().sum())

# Phân tích cơ bản
print("Summary Statistics:\n", df.describe())

# Ma trận tương quan
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
plt.title('Correlation Matrix')
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='Blues')
plt.show()

# 3. Data Preprocessing
# Tạo bản sao để xử lý
preprocessed_df = df.copy()

# Xử lý missing values (điền trung bình cho số, loại bỏ dòng nếu cần)
preprocessed_df['mileage(km/ltr/kg)'].fillna(preprocessed_df['mileage(km/ltr/kg)'].mean(), inplace=True)
preprocessed_df['engine'].fillna(preprocessed_df['engine'].mean(), inplace=True)
preprocessed_df['seats'].fillna(preprocessed_df['seats'].mean(), inplace=True)
preprocessed_df['max_power'] = pd.to_numeric(preprocessed_df['max_power'], errors='coerce')  # Chuyển thành số
preprocessed_df['max_power'].fillna(preprocessed_df['max_power'].mean(), inplace=True)
preprocessed_df.dropna(inplace=True)  # Loại bỏ các dòng còn lại nếu có NaN

# Encode categorical columns
categorical_columns = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    preprocessed_df[col] = le.fit_transform(preprocessed_df[col])
    label_encoders[col] = le

# Chia dữ liệu thành X và y
X = preprocessed_df.drop(columns=['selling_price'])
y = preprocessed_df['selling_price']

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training and Evaluation
# Huấn luyện Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Random Forest): {mse:.2f}")

# So sánh giá trị thực tế và dự đoán
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Đường tham chiếu
plt.title('Actual vs Predicted Selling Price')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

# 5. Feature Importance
importances = rf_model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
print("Feature Importance:\n", feature_importance_df)

# Vẽ feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.show()

# 6. Save Model and Encoders
joblib.dump(rf_model, 'random_forest_model.sav')
joblib.dump(label_encoders, 'label_encoders.sav')
print("Model and encoders saved successfully.")