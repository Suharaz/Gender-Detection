import cv2
import cvlib as cv
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog

# Load model
model = load_model('my_model.h5')

classes = ['man', 'woman']

def upload_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if video_path:
        detect_objects(video_path)

def detect_objects(video_path):
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        face, temp = cv.detect_face(frame)
        count_male_frame = 0
        count_female_frame = 0
        
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
            idx = np.argmax(conf)
            label = classes[idx]

            if label.startswith('man'):
                count_male_frame += 1
            elif label.startswith('woman'):
                count_female_frame += 1

            color = (0, 255, 0)
            if label.startswith('man'):
                color = (255, 0, 0)  # Blue color for male
            elif label.startswith('woman'):
                color = (0, 0, 255)  # Red color for female

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        label_info = "Male: {} | Female: {}".format(count_male_frame, count_female_frame)
        cv2.putText(frame, label_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


window = tk.Tk()
window.title("Phát hiện đối tượng và đếm số lượng từng nhãn")
canvas = tk.Canvas(window, width=700, height=700)
canvas.pack()
upload_btn = tk.Button(window, text="Tải lên video", command=upload_video)
upload_btn.pack(pady=10)

window.mainloop()
