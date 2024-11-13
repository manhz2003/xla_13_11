import numpy as np
from preprocess import load_data  # Sử dụng hàm load_data từ preprocess.py

# Load dữ liệu từ các thư mục huấn luyện và validation
X_train, y_train = load_data('data/train')
X_val, y_val = load_data('data/val')

# Lưu dữ liệu vào file npz
np.savez('data/train_data.npz', data=X_train, labels=y_train)
np.savez('data/val_data.npz', data=X_val, labels=y_val)

print("Đã lưu dữ liệu huấn luyện vào 'data/train_data.npz'")
print("Đã lưu dữ liệu validation vào 'data/val_data.npz'")