import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model.h5')  


class_names = ['Acne', 'Eczema', 'Psoriasis', 'Healthy']  


IMG_SIZE = 128


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow('Webcam - Press p to Predict', frame)

    key = cv2.waitKey(1)

    if key == ord('p'):
        photo = frame.copy()
        gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1) 
        img = np.expand_dims(img, axis=0)  
        prediction = model.predict(img, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence < 0.5:
            result = "Healthy"
        else:
            result = f"{class_names[predicted_class]}"

        print(f"Prediction: {result} ({confidence*100:.2f}%)")
        result_text = f'{result} ({confidence*100:.2f}%)'
        (text_width, text_height), _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        x, y = 10, 40 + text_height
        cv2.rectangle(photo, (x, y - text_height - 10), (x + text_width + 10, y + 10), (0, 0, 0), -1)
        cv2.putText(photo, result_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow('Prediction Result', photo)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
