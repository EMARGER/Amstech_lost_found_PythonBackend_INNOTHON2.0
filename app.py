import os
import numpy as np
from PIL import Image
import cv2
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

print("Python executable:", sys.executable)
print("cv2 location:", cv2.__file__)
print("cv2 version:", cv2.__version__)
print("cv2 has 'face':", hasattr(cv2, 'face'))


@app.route('/api/generate-dataset', methods=['POST'])
# @cross_origin(origin='http://localhost:4200')
def generate_dataset():
    data = request.json
    id = 1  # For demo purposes

    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            return img[y:y+h, x:x+w]
        return None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({'status': 'error', 'message': 'Camera not accessible'})
    
    img_id = 0

    while True:
        ret, frame = cap.read()
        cropped = face_cropped(frame)

        if cropped is not None:
            img_id += 1
            face = cv2.resize(cropped, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/user.{id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            if img_id == 50:
                break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'status': 'done', 'images': img_id})

@app.route('/api/train', methods=['GET'])
def train():
    try:
        print("[INFO] Starting training...")
        data_dir = "data"
        if not os.path.exists(data_dir):
            print("[ERROR] Data directory does not exist.")
            return jsonify({'status': 'error', 'message': 'Data directory does not exist'}), 400

        path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"[INFO] Found {len(path)} images")

        faces = []
        ids = []

        for image_path in path:
            try:
                img = Image.open(image_path).convert('L')
                image_np = np.array(img, 'uint8')
                filename = os.path.split(image_path)[1]
                print(f"[INFO] Processing {filename}")
                id = int(filename.split(".")[1])
                faces.append(image_np)
                ids.append(id)
            except Exception as e:
                print(f"[WARNING] Skipping {image_path}: {e}")

        if not faces:
            print("[ERROR] No valid training images found.")
            return jsonify({'status': 'error', 'message': 'No valid training images found'}), 400

        ids = np.array(ids)

        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")

        print("[INFO] Training complete. Model saved as classifier.xml")
        return jsonify({'status': 'trained'})

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# Route to detect faces and recognize the person
@app.route('/api/detect', methods=['GET'])
def detect():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id, pred = clf.predict(gray[y:y+h, x:x+w])
            confidence = int(100 * (1 - pred / 300))
            label = "Unknown"
            if confidence > 75:
                if id == 1:
                    label = "Atul"
                elif id == 2:
                    label = "Manish"
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.imshow("Face Detection", img)
        if cv2.waitKey(1) == 13:  # Press 'Enter' to stop
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'status': 'done'})

if __name__ == '__main__':
    app.run(debug=True)
