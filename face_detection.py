import cv2
import numpy as np
import urllib.request
import face_recognition
import os
from datetime import datetime

path = 'image_folder/'
image_paths = [os.path.join(path, img) for img in os.listdir(path) if img.endswith(('.jpg', '.png', '.jpeg'))]

url = 'http://192.168.0.145/stream'

attendance_file = 'Attendance.csv'
if attendance_file in os.listdir():
    print(f"El archivo '{attendance_file}' ya existe.")
else:
    with open(attendance_file, 'w') as f:
        f.write('Name,Time\n')
    print(f"El archivo '{attendance_file}' ha sido creado.")

images = []
classNames = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    images.append(image)
    classNames.append(os.path.splitext(os.path.basename(image_path))[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open(attendance_file, 'r+') as f:
        data = f.readlines()
        nameList = [line.split(',')[0] for line in data]
        if name not in nameList:
            now = datetime.now()
            timeString = now.strftime('%H:%M:%S')
            f.write(f'{name},{timeString}\n')
            print(f"Asistencia registrada para {name} a las {timeString}")

encodeListKnown = findEncodings(images)

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("No se pudo abrir el flujo de video.")
else:
    print("Flujo de video iniciado.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el marco del video.")
        break

    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(f"Persona detectada: {name}")  
            markAttendance(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('ESP32-CAM', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
