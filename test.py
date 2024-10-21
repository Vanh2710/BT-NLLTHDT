from vnstock import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Lấy dữ liệu từ thư viện vnstock cho công ty Vinamilk (VNM)
df = stock_historical_data(symbol="VNM", start_date="2023-01-10", end_date="2024-01-10", resolution="1D", type="stock", beautify=True, decor=False, source='DNSE')

# 2. Xoá những cột không cần thiết
dc = df.drop(columns=['ticker', 'time'])

# 3. Xử lý giá trị thiếu cho các cột số
numeric_cols = dc.select_dtypes(include=[np.number]).columns
dc[numeric_cols] = dc[numeric_cols].fillna(dc[numeric_cols].mean())

# 4. Chọn cột giá đóng cửa và chuẩn hóa dữ liệu
data = dc[['close']].values  # Chỉ lấy giá đóng cửa
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 5. Tạo dữ liệu huấn luyện cho LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Thiết lập số lượng time_step
time_step = 10
X, y = create_dataset(scaled_data, time_step)

# 6. Reshape dữ liệu để phù hợp với đầu vào của LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# 7. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Tạo mô hình LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

# 9. Biên dịch mô hình
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# 10. Huấn luyện mô hình LSTM
lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# 11. Dự đoán với mô hình LSTM
y_pred = lstm_model.predict(X_test)

# 12. Chuyển đổi giá dự đoán về giá thực tế
y_pred_inverse = scaler.inverse_transform(y_pred)
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

# 13. Đánh giá mô hình LSTM
mse = mean_squared_error(y_test_inverse, y_pred_inverse)
mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
r2 = r2_score(y_test_inverse, y_pred_inverse)

print("\nMô hình LSTM:")
print(f"MSE: {mse}, MAE: {mae}, R²: {r2}")

# 14. Nhập dữ liệu từ người dùng
open_price = 64970  # Giá mở cửa
high_price = 64970   # Giá cao nhất
low_price = 64300    # Giá thấp nhất
volume = 3006600     # Khối lượng giao dịch

# 15. Tạo mảng dữ liệu mới từ đầu vào của người dùng
user_input = np.array([[open_price, high_price, low_price, volume]])

# 16. Tạo scaler mới cho các đặc trưng đầu vào
input_scaler = MinMaxScaler(feature_range=(0, 1))
input_scaler.fit(user_input)  # Huấn luyện scaler với dữ liệu đầu vào

# 17. Chuẩn hóa dữ liệu đầu vào
user_input_scaled = input_scaler.transform(user_input)

# 18. Tạo dữ liệu cho mô hình LSTM (sử dụng thời điểm gần nhất trong dữ liệu huấn luyện)
last_sequence = scaled_data[-time_step:].reshape((1, time_step, 1))  # Lấy 10 giá cuối cùng từ dữ liệu

# 19. Dự đoán giá đóng cửa với mô hình LSTM
predicted_close_lstm = lstm_model.predict(last_sequence)

# 20. Chuyển đổi giá dự đoán về giá thực tế
predicted_close_lstm_inverse = scaler.inverse_transform(predicted_close_lstm)

print(f'Giá dự đoán đóng cửa cho VNM bằng mô hình LSTM: {predicted_close_lstm_inverse[0][0]} VND')
