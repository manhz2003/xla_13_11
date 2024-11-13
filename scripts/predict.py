import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load mô hình đã lưu
model = tf.keras.models.load_model('cnn_model.h5', compile=False)

# Đường dẫn tới ảnh cần test
img_path = 'data/cat.1.jpg'

# Load ảnh và tiền xử lý
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) 
img_array = img_array.astype('float32') / 255.0

# Dự đoán
prediction = model.predict(img_array)
class_label = 'Dog' if prediction[0][0] >= 0.5 else 'Cat'
print(f"Kết quả dự đoán: {class_label}")
