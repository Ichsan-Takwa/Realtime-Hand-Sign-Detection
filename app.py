from flask import Flask, Response, render_template, request, jsonify
import cv2
import mediapipe as mp
import pickle
import numpy as np
import base64
from PIL import Image
from io import BytesIO

# Memuat model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 
               14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/process_image", methods=['POST'])
def opencam():
    data = request.get_json()
    image_data = data['image_data']
    image_data = image_data.split(',')[1]  # Remove the "data:image/png;base64," part
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    data_aux = []
    x_ = []
    y_ = []


    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    H, W, _ = img.shape
    results = hands.process(img)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        if len(data_aux) != 84:
            data_aux.extend([0] * (84 - len(data_aux)))  # Padding dengan nilai 0 jika kurang dari 84 fitur

        data_aux = np.asarray(data_aux).reshape(1, -1)

        prediction = model.predict(data_aux)
        prediction_character = labels_dict[int(prediction[0])]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(img, prediction_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255,), 3,
                    cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(jpeg).decode('utf-8')
    return jsonify({'result': img_str})

if __name__ == '__main__':
    app.run(host="localhost", port=8080, debug=True, threaded=False)   #ganti dengan alamat ipmu

