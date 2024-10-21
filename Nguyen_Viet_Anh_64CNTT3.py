from vnstock import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Lấy dữ liệu từ thư viện vnstock cho công ty Vinamilk (VNM)
df = stock_historical_data(symbol="VNM", start_date="2023-01-10", end_date="2024-01-10", resolution="1D", type="stock", beautify=True, decor=False, source='DNSE')

# 2. Xoá những cột không cần thiết và xử lý giá trị không phải số
dc = df.drop(columns=['ticker', 'time'])

# 3. Xử lý giá trị thiếu cho các cột số
numeric_cols = dc.select_dtypes(include=[np.number]).columns
dc[numeric_cols] = dc[numeric_cols].fillna(dc[numeric_cols].mean())

# 4. Tách biến độc lập (X) và biến phụ thuộc (y)
X = dc[['open', 'high', 'low', 'volume']]  # Sử dụng 'close' để dự đoán giá
y = dc['close']  # Dự đoán giá đóng cửa

# 5. Tính Z-score và loại bỏ các outliers
z_scores = np.abs(stats.zscore(X))
filtered_entries = (z_scores < 3).all(axis=1)
X_filtered = X[filtered_entries]
y_filtered = y[filtered_entries]

# 6. Chuẩn hóa các biến độc lập
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# 7. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)

## ================== Mô hình Hồi quy tuyến tính ==================
# 8. Khởi tạo và huấn luyện mô hình hồi quy tuyến tính
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 9. Dự đoán trên tập kiểm tra
y_pred_test_linear = linear_model.predict(X_test)

# 12. Đánh giá mô hình cho tập kiểm tra
mse_test_linear = mean_squared_error(y_test, y_pred_test_linear)
r2_test_linear = r2_score(y_test, y_pred_test_linear)
mae_test_linear = mean_absolute_error(y_test, y_pred_test_linear)

# 7. Nhập dữ liệu từ người dùng
open_price = 64970  # Giá mở cửa
high_price = 64970   # Giá cao nhất
low_price = 64300    # Giá thấp nhất
volume = 3006600     # Khối lượng giao dịch

# 8. Tạo mảng dữ liệu mới từ đầu vào của người dùng
user_input = np.array([[open_price, high_price, low_price, volume]])

# 9. Chuẩn hóa dữ liệu đầu vào
user_input_scaled = scaler.transform(user_input)

# 10. Dự đoán giá đóng cửa
predicted_close = linear_model.predict(user_input_scaled)

print("Mô hình Hồi quy tuyến tính:")
print(f'Giá dự đoán đóng cửa cho VNM: {predicted_close[0]} VND')
print(f"MSE: {mse_test_linear}, R²: {r2_test_linear}, MAE: {mae_test_linear}")
