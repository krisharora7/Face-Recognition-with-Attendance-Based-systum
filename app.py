import streamlit as st
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# ---------------------- FACE REGISTRATION ----------------------
def register_face(name: str, max_samples: int = 100):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    faces_data = []
    i = 0
    stframe = st.empty()  # live frame in Streamlit

    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) < max_samples and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, f"Samples: {len(faces_data)}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

        # show in Streamlit instead of cv2.imshow
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if len(faces_data) >= max_samples:
            break

    video.release()

    faces_data = np.asarray(faces_data).reshape(len(faces_data), -1)

    os.makedirs("data", exist_ok=True)

    # Save names
    if 'names.pkl' not in os.listdir('data/'):
        names = [name] * len(faces_data)
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * len(faces_data)
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    # Save faces
    if 'faces_data.pkl' not in os.listdir('data/'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)

    return f"✅ Face data for {name} registered successfully!"


# ---------------------- ATTENDANCE MARKING ----------------------
def mark_attendance():
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    attendance_dir = "attendance_records"
    os.makedirs(attendance_dir, exist_ok=True)

    COL_NAMES = ['NAME', 'TIME']
    marked_names = set()
    results = []

    stframe = st.empty()

    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.putText(frame, str(output), (x, y-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if output not in marked_names:
                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                csv_file = os.path.join(attendance_dir, f"Attendance_{date}.csv")
                exist = os.path.isfile(csv_file)

                attendance = [str(output), str(timestamp)]
                with open(csv_file, "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if not exist:
                        writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

                marked_names.add(output)
                results.append({"name": output, "time": timestamp})

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if len(results) > 0:  # stop after first set of recognitions
            break

    video.release()
    return results


# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="Face Recognition Attendance", page_icon="🧑‍💻", layout="centered")

st.title("📸 Face Recognition Attendance System")

menu = st.sidebar.radio("Choose Action", ["Register Face", "Mark Attendance"])

if menu == "Register Face":
    st.subheader("Register a new person")
    name = st.text_input("Enter Name:")
    if st.button("Start Registration"):
        if name.strip() == "":
            st.error("⚠️ Please enter a name first.")
        else:
            msg = register_face(name)
            st.success(msg)

elif menu == "Mark Attendance":
    st.subheader("Mark Attendance")
    if st.button("Start Attendance"):
        results = mark_attendance()
        if results:
            st.success("✅ Attendance Marked")
            st.table(results)
        else:
            st.warning("No faces detected.")
