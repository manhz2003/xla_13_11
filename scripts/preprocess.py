import os
import numpy as np
from tensorflow.keras.preprocessing import image

# Định nghĩa hàm để load dữ liệu từ thư mục
def load_data(image_dir):
    data = []
    labels = []
    
    # In ra đường dẫn tuyệt đối của thư mục
    print(f"Đang kiểm tra thư mục: {os.path.abspath(image_dir)}")
    
    if not os.path.exists(image_dir):
        print(f"Thư mục {image_dir} không tồn tại!")
        return np.array([]), np.array([])

    # Kiểm tra các thư mục con 'cats' và 'dogs'
    for label, subdir in enumerate(['cats', 'dogs']):
        class_dir = os.path.join(image_dir, subdir)
        print(f"Đang kiểm tra thư mục con: {class_dir}")
        if not os.path.exists(class_dir):
            print(f"Thư mục {class_dir} không tồn tại!")
            continue
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            try:
                img = image.load_img(img_path, target_size=(128, 128))
                img_array = image.img_to_array(img)
                data.append(img_array)
                labels.append(label)  # 0 cho cats, 1 cho dogs
            except Exception as e:
                print(f"Không thể đọc file {img_name}: {e}")
                continue

    return np.array(data), np.array(labels)

# Load dữ liệu huấn luyện và validation
X_train, y_train = load_data('data/train')
X_val, y_val = load_data('data/val')

# Kiểm tra kích thước dữ liệu
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
