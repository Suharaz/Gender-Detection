import cv2
import cvlib as cv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Load model
model = load_model('my_model.h5')

# Chech status cam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Không thể mở webcam")
    exit()

classes = ['man', 'woman']

# Tạo giao diện app

window = tk.Tk()
window.title("Phát hiện giới tính và đếm số lượng")
canvas = tk.Canvas(window, width=700, height=700)
canvas.pack()
button_frame = tk.Frame(window)
button_frame.pack(pady=10)


def start_detection():
    start_btn.config(state=tk.DISABLED)
    quit_btn.config(state=tk.NORMAL)
    
    def detect_gender():
        while True:
            status, frame = webcam.read()

            if not status:
                print("Không thể nhận được khung hình từ webcam")
                break

            
            face= cv.detect_face(frame)[0]
            label_counts = {label: 0 for label in classes}
            for idx, f in enumerate(face):
                startX, startY, endX, endY = f
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                face_crop = np.copy(frame[startY:endY, startX:endX])
                if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                    continue

                # Chuẩn hoá dữ liệu để đúng vs model
                face_crop = cv2.resize(face_crop, (64, 64))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                conf = model.predict(face_crop)[0]
                idx = np.argmax(conf)
                label = classes[idx]
                label = "{}: {:.2f}%".format(label, conf[idx] * 100)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                label_counts[label.split(':')[0].strip()] += 1

            y_offset = 30
            for label, count in label_counts.items():
                label_text = '{}: {}'.format(label, count)
                cv2.putText(frame, label_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 20

            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.image = img

        

        webcam.release()
        cv2.destroyAllWindows()
    
    detection_thread = threading.Thread(target=detect_gender)
    detection_thread.start()

def quit_application():
    window.destroy()
    webcam.release()
    cv2.destroyAllWindows()


start_btn = tk.Button(button_frame, text="Start", command=start_detection)
start_btn.pack(side=tk.LEFT, padx=5)
quit_btn = tk.Button(button_frame, text="Quit", command=quit_application, state=tk.DISABLED)
quit_btn.pack(side=tk.LEFT, padx=5)

window.mainloop()
