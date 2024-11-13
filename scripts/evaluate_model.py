import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Load dữ liệu
train_data = np.load('data/train_data.npz')
val_data = np.load('data/val_data.npz')

# Lấy dữ liệu và nhãn
X_train, y_train = train_data['data'], train_data['labels']
X_val, y_val = val_data['data'], val_data['labels']

# Kiểm tra kích thước dữ liệu
print("Kích thước dữ liệu huấn luyện:", X_train.shape)  # Expected: (số lượng ảnh, 128, 128, 3)
print("Kích thước nhãn huấn luyện:", y_train.shape)    # Expected: (số lượng ảnh,)
print("Kích thước dữ liệu validation:", X_val.shape)   # Expected: (số lượng ảnh, 128, 128, 3)
print("Kích thước nhãn validation:", y_val.shape)      # Expected: (số lượng ảnh,)

# Đảm bảo dữ liệu có kích thước đúng
if X_train.ndim != 4 or X_train.shape[1:] != (128, 128, 3):
    print("Reshaping lại dữ liệu huấn luyện...")
    X_train = X_train.reshape(-1, 128, 128, 3)

if X_val.ndim != 4 or X_val.shape[1:] != (128, 128, 3):
    print("Reshaping lại dữ liệu validation...")
    X_val = X_val.reshape(-1, 128, 128, 3)

# Đảm bảo rằng dữ liệu có kiểu float32 và chuẩn hóa (chia cho 255)
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# Kiểm tra lại kích thước dữ liệu sau khi reshape
print("Kích thước dữ liệu huấn luyện sau khi reshape:", X_train.shape)
print("Kích thước dữ liệu validation sau khi reshape:", X_val.shape)

# Chuẩn bị mô hình CNN
model = Sequential([
    Input(shape=(128, 128, 3)),  # Đảm bảo layer đầu tiên có đúng shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output binary classification (Chó hoặc Mèo)
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
