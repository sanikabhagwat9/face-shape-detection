import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to classify face shape
def classify_face_shape(landmarks):
    points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

    jaw_width = np.linalg.norm(points[0] - points[16])
    cheek_width = np.linalg.norm(points[1] - points[15])
    face_height = np.linalg.norm(points[27] - points[8])
    forehead_width = np.linalg.norm(points[17] - points[26])

    face_ratio = face_height / cheek_width
    jaw_cheek_ratio = jaw_width / cheek_width

    if abs(cheek_width - face_height) < 20 and jaw_cheek_ratio < 1.1:
        return "Round"
    elif face_ratio > 1.5:
        return "Long"
    elif jaw_cheek_ratio > 1.2:
        return "Square"
    elif cheek_width > forehead_width and cheek_width > jaw_width:
        return "Diamond"
    elif forehead_width > jaw_width:
        return "Heart"
    else:
        return "Oval"

# Function to detect face shape
def detect_face_shape(image, gray, face):
    landmarks = predictor(gray, face)
    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    face_shape = classify_face_shape(landmarks)
    return image, face_shape

# Function to upload an image and find the face shape
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])

    if file_path:
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            image, face_shape = detect_face_shape(image, gray, face)
            cv2.putText(image, f"Face Shape: {face_shape}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Uploaded Image - Face Shape Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Function to capture video from webcam and detect face shape
def capture_image():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        cv2.putText(frame, "Press 'Space' to capture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Face Shape Detection - Real-Time", frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            for face in faces:
                frame, face_shape = detect_face_shape(frame, gray, face)
                cv2.putText(frame, f"Face Shape: {face_shape}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Captured Image - Face Shape Detected", frame)
            cv2.waitKey(0)
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI for uploading or capturing image with improved interface
def main():
    root = tk.Tk()
    root.title("Face Shape Detection")
    root.geometry("400x300")
    
    # Set the background color
    root.configure(bg="#f0f0f0")

    # Header label with custom styling
    header_label = tk.Label(root, text="Face Shape Detection", font=("Helvetica", 18, "bold"), bg="#4CAF50", fg="white", pady=10)
    header_label.pack(fill="x")

    # Add some padding around buttons and customize button appearance
    button_style = {"bg": "#2196F3", "fg": "white", "font": ("Helvetica", 14), "activebackground": "#0D47A1", "activeforeground": "white", "width": 25, "height": 2}

    upload_button = tk.Button(root, text="Upload Image", command=upload_image, **button_style)
    upload_button.pack(pady=20)

    capture_button = tk.Button(root, text="Capture Image from Webcam", command=capture_image, **button_style)
    capture_button.pack(pady=20)

    # Run the application
    root.mainloop()

if __name__ == "__main__":
    main()
