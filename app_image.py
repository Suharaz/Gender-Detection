import cv2
import cvlib as cv
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Load model
model = load_model('my_model.h5')

classes = ['man', 'woman']

def upload_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        detect_gender(image_path)

def detect_gender(image_path):
    frame = cv2.imread(image_path)
    face = cv.detect_face(frame)[0]
    count_male = 0
    count_female = 0
    for idx, f in enumerate(face):
        startX, startY, endX, endY = f
        face_crop = np.copy(frame[startY:endY, startX:endX])
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        face_crop = cv2.resize(face_crop, (64, 64))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0]
        print(conf)
        idx = np.argmax(conf)
        label = classes[idx]
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        color = (0, 255, 0)
        if label.startswith('man'):
            color = (255, 0, 0)  # mau xanh
            count_male += 1
        elif label.startswith('woman'):
            color = (0, 0, 255)  # màu đỏ
            count_female += 1

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Thông tin hiện thị
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    text = "Male: {}\nFemale: {}".format(count_male, count_female)
    plt.text(10, 10, text, fontsize=12, color='white', backgroundcolor='black', verticalalignment='top')
    plt.show()


# Tạo giao diện
window = tk.Tk()
window.title("Phát hiện giới tính và đếm số lượng từng nhãn")
canvas = tk.Canvas(window, width=700, height=700)
canvas.pack()
upload_btn = tk.Button(window, text="Tải lên ảnh", command=upload_image)
upload_btn.pack(pady=10)

window.mainloop()
