import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from time import sleep
from keras_preprocessing.image import img_to_array
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Tải bộ phân loại khuôn mặt và mô hình
face_classifier = cv2.CascadeClassifier(r'C:\Users\DELL\BTL_DACN\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\DELL\BTL_DACN\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Cài đặt fone chữ hiển thị
font_path = "arial.ttf"
font = ImageFont.truetype(font_path, 32)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    if len(faces) == 0:  # Không phát hiện thấy khuôn mặt nào
        # Tạo hình ảnh PIL từ khung hình
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        text = "Không phát hiện được khuôn mặt"

        # Lấy kích thước của hộp văn bản
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Đặt chữ ở giữa màn hình
        text_x = (frame.shape[1] - text_width) // 2  # Căn giữa theo chiều ngang
        text_y = (frame.shape[0] - text_height) // 2  # Căn giữa theo chiều dọc

        # Vẽ chữ lên hình
        draw.text((text_x, text_y), text, font=font, fill=(0, 255, 0))

        # Chuyển đổi lại thành định dạng OpenCV
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Không có khuôn mặt nào', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
