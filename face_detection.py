import firebase_admin
from firebase_admin import credentials, storage, db
import urllib.request
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import requests
import urllib.parse

cred = credentials.Certificate("firebase_credentials.json")  
firebase_admin.initialize_app(cred, {
    'storageBucket': 'sense-bell.firebasestorage.app',  
    'databaseURL': 'https://sense-bell-default-rtdb.firebaseio.com/' 
})

bucket = storage.bucket()

last_detection_time = {}

DETECTION_INTERVAL = 300 

def download_images_from_firebase():
    blobs = bucket.list_blobs(prefix="photos/")  
    images = []
    classNames = []

    for blob in blobs:
        if blob.name.endswith(('.jpg', '.png', '.jpeg')):
            encoded_name = urllib.parse.quote(blob.name, safe='')  # Mueve esto dentro del bucle
            url = f"https://firebasestorage.googleapis.com/v0/b/sense-bell.firebasestorage.app/o/{encoded_name}?alt=media"
            try:
                resp = urllib.request.urlopen(url)
                img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                images.append(img)
                classNames.append(blob.name.split("/")[-1].split(".")[0]) 
            except Exception as e:
                print(f"❌ Error al descargar {blob.name}: {e}")

    return images, classNames

print("🔹 Descargando imágenes desde Firebase...")
images, classNames = download_images_from_firebase()
print(f"🔹 Se cargaron {len(images)} imágenes para reconocimiento.")

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  
            encodeList.append(encode[0])  
    return encodeList

encodeListKnown = findEncodings(images)
print("🔹 Codificación facial completada.")

url = 'http://192.168.0.145/stream'  
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ No se pudo abrir el flujo de video de la ESP32-CAM.")
else:
    print("✅ Transmisión de ESP32-CAM iniciada.")

attendance_file = 'Attendance.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Time\n')
    print("🔹 Archivo de asistencia creado.")

def markAttendance(name):
    now = datetime.now()
    if name in last_detection_time:
        time_elapsed = now - last_detection_time[name]
        if time_elapsed.total_seconds() < DETECTION_INTERVAL:
            return

    timeString = now.strftime('%H:%M:%S')
    with open(attendance_file, 'a') as f:
        f.write(f'{name},{timeString}\n')
    print(f"✅ Asistencia registrada para {name} a las {timeString}")
    markAttendanceInFirebase(name, timeString)

    last_detection_time[name] = now

def markAttendanceInFirebase(name, timestamp):
    url = "https://sense-bell-default-rtdb.firebaseio.com/attendance.json"
    data = {
        "name": name,
        "timestamp": timestamp,
        "message": f"{name} está en la puerta!"
    }
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print(f"✅ Asistencia subida a Firebase para {name}")
    else:
        print(f"❌ Error subiendo asistencia a Firebase: {response.text}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer el marco del video.")
        break

    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis) if faceDis.size > 0 else None

        if matchIndex is not None and matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(f"👤 Persona detectada: {name}")  
            markAttendance(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('ESP32-CAM Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
