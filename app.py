from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import cv2
from PIL import Image
import mysql.connector
import logging

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app, supports_credentials=True,resources={r"/api/*": {"origins": "*"}}, allow_headers="*", methods=["GET", "POST", "OPTIONS"])

# Dataset generation with database insert
@app.route('/api/generate-dataset', methods=['POST'])
def generate_dataset():
    data = request.json
    name = data.get('name')
    age = data.get('age')
    email = data.get('email')
    # Add additional fields for gender, id_proof, etc. (even if they are empty)
    gender = data.get('gender', '')  # Default to empty string if not provided
    id_proof = data.get('id_proof', '')  # Default to empty string if not provided
    description = data.get('description', '')  # Default to empty string if not provided
    mobile_number = data.get('mobile_number', '')
    if not name or not age or not email:
        return jsonify({'status': 'error', 'message': 'Incomplete user details'}), 400

    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="lostfound"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM missing_person")
    id = len(mycursor.fetchall()) + 1

    sql = "INSERT INTO missing_person (name, id_proof, age, gender, description, mobile_number, email) VALUES (%s, %s, %s, %s, %s, %s, %s);"
    val = (name, id_proof, age, gender, description, mobile_number, email)
    mycursor.execute(sql, val)
    mydb.commit()

    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            return img[y:y + h, x:x + w]
        return None

    cap = cv2.VideoCapture(0)
    img_id = 0
    os.makedirs("data", exist_ok=True)

    while True:
        ret, frame = cap.read()
        cropped = face_cropped(frame)

        if cropped is not None:
            img_id += 1
            face = cv2.resize(cropped, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/user.{id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)

        if cv2.waitKey(1) == 13 or img_id == 200:
            break

    cap.release()
    cv2.destroyAllWindows()

    return jsonify({'status': 'done', 'images': img_id})

# Train classifier
@app.route('/api/train', methods=['GET'])
def train():
    try:
        data_dir = "data"
        if not os.path.exists(data_dir):
            return jsonify({'status': 'error', 'message': 'Data directory does not exist'}), 400

        path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        faces = []
        ids = []

        for image_path in path:
            try:
                img = Image.open(image_path).convert('L')
                image_np = np.array(img, 'uint8')
                id = int(os.path.split(image_path)[1].split(".")[1])
                faces.append(image_np)
                ids.append(id)
            except Exception as e:
                print(f"[WARNING] Skipping {image_path}: {e}")

        if not faces:
            return jsonify({'status': 'error', 'message': 'No valid training images found'}), 400

        ids = np.array(ids)
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")

        return jsonify({'status': 'trained'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Detect a specific user
@app.route('/api/detect', methods=['POST'])
def detect():
    data = request.json
    name = data.get('name')

    if not name:
        return jsonify({'status': 'error', 'message': 'Name is required'}), 400

    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="lostfound"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT id FROM missing_person WHERE LOWER(name) = %s", (name.lower(),))
    result = mycursor.fetchone()

    if not result:
        return jsonify({'status': 'error', 'message': 'User not found in database'}), 404

    target_id = result[0]
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    found = False

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 6)

        for (x, y, w, h) in faces:
            id, pred = clf.predict(gray[y:y+h, x:x+w])
            confidence = int(100 * (1 - pred / 300))

            if confidence > 70 and id == target_id:
                found = True

                # Consume result properly
                mycursor.execute("SELECT MAX(id) FROM found_data")
                result = mycursor.fetchone()  # Important line to consume result
                max_id = result[0] if result and result[0] is not None else 0
                next_id = max_id + 1

                mycursor.execute(
                    "INSERT INTO found_data (id, missing_person_id, location) VALUES (%s, %s, %s)",
                    (next_id, target_id, 1)
                )
                mydb.commit()
                cap.release()
                cv2.destroyAllWindows()
                return jsonify({'status': 'found', 'user': name})

        # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # cv2.putText(img, f"ID: {id} Conf: {confidence}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Detecting Face", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'status': 'not_found'})

if __name__ == '__main__':
    app.run(debug=True)
