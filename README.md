# Face Recoginaigation

# 👤 Missing Person Face Recognition System

This is a Python-based application to **detect and track missing persons** using **OpenCV's face recognition** and **MySQL database integration**. It allows storing face datasets, training a recognizer, and detecting faces in real-time using webcam input.

---

## 🚀 Features

- Add new missing person data along with photo samples.
- Automatically crop and store facial images from live webcam.
- Train a facial recognition model (LBPH algorithm).
- Detect and recognize missing persons in real-time.
- Record detection events with timestamp and camera source.
- Fully integrated with MySQL for backend data management.

---

## 🧰 Tech Stack

| Technology | Description |
|------------|-------------|
| Python | Backend logic and OpenCV integration |
| OpenCV | Face detection & recognition (LBPH algorithm) |
| MySQL | Database to store person details & detection logs |
| Tkinter | GUI popups for user interaction |
| PIL (Pillow) | Image processing |
| NumPy | Numerical image array conversion |

---

## 🗃️ Folder Structure

FaceReco/
├── data/ # Collected face images (user.ID.imgNo.jpg)
├── classifier.xml # Trained face recognition model
├── haarcascade_frontalface_default.xml
├── detect_face.py # Detection script
├── generate_dataset.py # Script to collect image samples
├── train_classifier.py # Training script
└── README.md


---

## ⚙️ Setup Instructions

1. **Clone this repo:**
   ```bash
   git clone https://github.com/your-username/face-recognition-missing-person.git
   cd face-recognition-missing-person
🖼️ Sample Image Naming Format
user.<id>.<img_number>.jpg
Example: user.3.45.jpg


📸 Detection Success Criteria
Face must match trained data with >80% confidence.

Upon success, face is marked with a red rectangle and the name is displayed.

Detection details are stored in the found_data table.

🛡️ Security Note
Make sure to secure your database credentials and validate user input before deploying publicly.

🙌 Contribution
Pull requests and suggestions are welcome! Please fork this repo and create a PR.

📄 License
This project is open-source and available under the MIT License.

💬 Contact
For queries or feedback, feel free to reach out at:
📧  goutamdogyan123@gmail.com

📧  atulpatel6357@gmail.com
