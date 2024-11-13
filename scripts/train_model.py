import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from preprocess import load_data  

# Tải dữ liệu từ preprocess.py
X_train, y_train = load_data('data/train')
X_val, y_val = load_data('data/val')

# Kiểm tra kích thước dữ liệu
print("Kích thước dữ liệu huấn luyện:", X_train.shape)  # Expected: (số lượng ảnh, 128, 128, 3)
print("Kích thước nhãn huấn luyện:", y_train.shape)    # Expected: (số lượng ảnh,)
print("Kích thước dữ liệu validation:", X_val.shape)   # Expected: (số lượng ảnh, 128, 128, 3)
print("Kích thước nhãn validation:", y_val.shape)      # Expected: (số lượng ảnh,)

# Chuẩn bị mô hình CNN
model = Sequential([
    Input(shape=(128, 128, 3)),  
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') 
])

# Compile mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16
)

# Lưu mô hình
model.save('cnn_model.h5')
print("Mô hình đã được lưu.")
