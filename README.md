# Face Recognition-Based Equipment Borrowing System

A web-based equipment borrowing management system that uses **Face Recognition**, **Computer Vision**, and **Machine Learning** to identify users and record equipment borrowing information.

The system allows users to register their information, capture face images through a webcam, train a recognition model, and automatically record the user's identity, classroom, and borrowing time.

## System Interface
<img width="1771" height="884" alt="image" src="https://github.com/user-attachments/assets/2bdceccc-9d40-4d87-8241-7f8e838f54db" />

## Features

### 1. User Registration
- Add new users with:
  - Full name
  - Student ID
  - Class
- Capture facial images using a webcam.
- Store facial images for model training.
  
### 2. Face Recognition
- Access the webcam to capture real-time video.
- Detect faces using OpenCV.
- Recognize registered users using a trained machine learning model.
- Automatically identify the user during the equipment borrowing process.

### 3. Classroom Management
- Enter the classroom where the equipment is being borrowed.
- Check whether a classroom is already occupied.
- Prevent multiple users from borrowing equipment from the same classroom at the same time.

### 4. Borrowing Records
The system automatically records:

- Student name
- Student ID
- Class
- Classroom
- Borrowing time

The records are stored in CSV files and displayed on the web interface.

### 5. User Management
- View registered users.
- Delete users from the system.
- Retrain the recognition model when the user database changes.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Web application framework |
| OpenCV | Face detection, image processing, and webcam access |
| NumPy | Numerical and array operations |
| Pandas | Data processing and CSV management |
| Scikit-learn | Machine learning and face recognition |
| Joblib | Model serialization and loading |
| HTML/CSS | Web interface |

## Project Structure

```text
face-recognition-attendance-system/
│
├── app.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
│
├── templates/
│   └── home.html
│
├── Attendance/
│   └── Attendance-YYYY-MM-DD.csv
│
├── static/
│   ├── faces/
│   └── face_recognition_model.pkl
